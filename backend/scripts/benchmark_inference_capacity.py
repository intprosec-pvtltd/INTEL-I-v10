"""Repeatable 10/20/30/40/50-camera central-inference load driver.

This tool actively submits virtual camera frames; it is not a telemetry-only
poller. For final capacity claims use real representative CCTV sources on the
target L40S deployment and retain the JSON/CSV output as test evidence.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import cv2
import httpx

DEFAULT_STAGES=(10,20,30,40,50)


def _frames_from_source(source: str, count: int, jpeg_quality: int) -> list[bytes]:
    capture=cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError("benchmark source could not be opened")
    frames=[]
    try:
        attempts=0
        while len(frames)<count and attempts<count*20:
            attempts+=1; ok,frame=capture.read()
            if not ok or frame is None:
                capture.set(cv2.CAP_PROP_POS_FRAMES,0); continue
            ok,encoded=cv2.imencode('.jpg',frame,[int(cv2.IMWRITE_JPEG_QUALITY),jpeg_quality])
            if ok: frames.append(encoded.tobytes())
    finally:
        capture.release()
    if not frames: raise RuntimeError("benchmark source produced no decodable frames")
    return frames


async def _health(client: httpx.AsyncClient, base: str) -> dict[str,Any]:
    response=await client.get(base+'/health/ready'); response.raise_for_status(); return response.json()


async def _virtual_camera(
    client:httpx.AsyncClient, base:str, camera_id:str, frames:list[bytes], fps:float,
    duration:float, latencies:list[float], counters:dict[str,int], lock:asyncio.Lock,
) -> None:
    interval=1.0/max(0.1,fps); deadline=time.monotonic()+duration; seq=0; next_at=time.monotonic()
    while time.monotonic()<deadline:
        wait=next_at-time.monotonic()
        if wait>0: await asyncio.sleep(wait)
        next_at+=interval; payload=frames[seq%len(frames)]; seq+=1; started=time.perf_counter()
        try:
            response=await client.post(
                base+'/v1/infer',
                data={
                    'camera_id':camera_id,'user_id':'0','timestamp':str(time.time()),
                    'source_type':'upload','frame_count':str(seq),'priority':'5',
                    'timestamp_source':'BENCHMARK_REPLAY','timestamp_quality':'SIMULATED',
                },
                files={'frame':('frame.jpg',payload,'image/jpeg')},
            )
            elapsed=(time.perf_counter()-started)*1000.0
            async with lock:
                latencies.append(elapsed)
                if response.status_code==200: counters['ok']+=1
                elif response.status_code==429: counters['shed']+=1
                else: counters['error']+=1
        except Exception:
            async with lock: counters['error']+=1


async def _run_stage(args, stage:int, frames:list[bytes]) -> dict[str,Any]:
    headers={}
    if args.token: headers['X-Intel-I-Inference-Token']=args.token
    timeout=httpx.Timeout(max(10.0,args.request_timeout),connect=5.0)
    limits=httpx.Limits(max_connections=max(100,stage*3),max_keepalive_connections=max(50,stage*2))
    latencies=[]; counters={'ok':0,'shed':0,'error':0}; lock=asyncio.Lock(); health_samples=[]
    async with httpx.AsyncClient(headers=headers,timeout=timeout,limits=limits) as client:
        async def sampler():
            end=time.monotonic()+args.duration
            while time.monotonic()<end:
                try: health_samples.append(await _health(client,args.base_url))
                except Exception: pass
                await asyncio.sleep(args.health_interval)
        tasks=[asyncio.create_task(_virtual_camera(client,args.base_url,f'BENCH-{stage}-{idx+1:03d}',frames,args.fps,args.duration,latencies,counters,lock)) for idx in range(stage)]
        tasks.append(asyncio.create_task(sampler()))
        await asyncio.gather(*tasks)
        final=await _health(client,args.base_url)
    sorted_lat=sorted(latencies)
    def pct(p):
        if not sorted_lat:return None
        idx=min(len(sorted_lat)-1,max(0,int(round((len(sorted_lat)-1)*p))))
        return round(sorted_lat[idx],3)
    gpu_utils=[float((h.get('gpu') or {}).get('utilization_percent')) for h in health_samples if (h.get('gpu') or {}).get('utilization_percent') is not None]
    gpu_mem=[float((h.get('gpu') or {}).get('memory_used_mb')) for h in health_samples if (h.get('gpu') or {}).get('memory_used_mb') is not None]
    queue=[int(h.get('queue_depth') or 0) for h in health_samples]
    row={
        'timestamp':time.time(),'camera_stage':stage,'target_ai_fps_per_camera':args.fps,'duration_seconds':args.duration,
        'requests_ok':counters['ok'],'requests_shed':counters['shed'],'requests_error':counters['error'],
        'client_p50_ms':pct(.50),'client_p95_ms':pct(.95),'client_p99_ms':pct(.99),
        'gpu_utilization_avg':round(statistics.fmean(gpu_utils),2) if gpu_utils else None,
        'gpu_utilization_max':round(max(gpu_utils),2) if gpu_utils else None,
        'gpu_memory_used_mb_max':round(max(gpu_mem),2) if gpu_mem else None,
        'queue_depth_max':max(queue) if queue else None,
        'server_avg_inference_ms':final.get('average_inference_latency_ms'),'server_dropped_frames':final.get('dropped_frames'),
        'average_batch_size':final.get('average_batch_size'),'current_batch_size':final.get('current_batch_size'),
        'nvdec_utilization':(final.get('gpu') or {}).get('decoder_utilization_percent'),
        'cpu_percent':final.get('cpu_percent'),'ram_percent':final.get('ram_percent'),
        'model_metrics':final.get('model_metrics'),'decoder':final.get('decoder'),'runtime':final.get('runtime'),
        'test_kind':'central-inference-active-replay',
        'capacity_claim_valid':False,
        'capacity_claim_note':'Set only after representative end-to-end RTSP test meets your acceptance thresholds.',
    }
    return row


def _write_csv(path:str, rows:list[dict[str,Any]]):
    flat=[]
    excluded={'model_metrics','decoder','runtime'}
    keys=[k for k in rows[0] if k not in excluded]
    for row in rows: flat.append({k:row.get(k) for k in keys})
    exists=os.path.exists(path)
    with open(path,'a',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys)
        if not exists: writer.writeheader()
        writer.writerows(flat)


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--base-url',default='http://127.0.0.1:9300')
    parser.add_argument('--source',required=True,help='Representative video file/RTSP source; never written to output')
    parser.add_argument('--stages',default=','.join(map(str,DEFAULT_STAGES)))
    parser.add_argument('--fps',type=float,default=5.0)
    parser.add_argument('--duration',type=float,default=60.0)
    parser.add_argument('--health-interval',type=float,default=2.0)
    parser.add_argument('--request-timeout',type=float,default=20.0)
    parser.add_argument('--sample-frames',type=int,default=48)
    parser.add_argument('--jpeg-quality',type=int,default=90)
    parser.add_argument('--token',default=os.getenv('INFERENCE_INTERNAL_TOKEN',''))
    parser.add_argument('--output-json',default='capacity-results.json')
    parser.add_argument('--output-csv',default='capacity-results.csv')
    args=parser.parse_args()
    stages=[int(v.strip()) for v in args.stages.split(',') if v.strip()]
    if any(stage not in DEFAULT_STAGES for stage in stages): raise SystemExit('stages must be among 10,20,30,40,50')
    frames=_frames_from_source(args.source,max(1,args.sample_frames),max(50,min(100,args.jpeg_quality)))
    rows=[]
    for stage in stages:
        print(f'Running active {stage}-camera replay stage...',flush=True)
        row=asyncio.run(_run_stage(args,stage,frames)); rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in {'model_metrics','decoder','runtime'}},indent=2))
    Path(args.output_json).write_text(json.dumps(rows,indent=2,default=str),encoding='utf-8')
    _write_csv(args.output_csv,rows)
    print(f'Results: {args.output_json}, {args.output_csv}')
    return 0

if __name__=='__main__': raise SystemExit(main())
