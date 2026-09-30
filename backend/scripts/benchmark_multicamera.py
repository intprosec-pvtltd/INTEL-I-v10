"""INTEL-I 5/10/25/50 camera acceptance benchmark collector.

Run against the real deployment. It does not synthesize benchmark claims.
Example:
  python scripts/benchmark_multicamera.py --api http://127.0.0.1:3000 --token "$TOKEN" --camera-ids-file cameras.txt --levels 5,10,25,50 --seconds 300
"""
from __future__ import annotations
import argparse, csv, json, re, statistics, time
from pathlib import Path
import httpx

METRICS = ("camera_fps", "camera_queue_depth", "camera_dropped_frames_total", "gpu_utilization_percent", "gpu_memory_used_bytes", "camera_frame_latency_seconds_sum", "camera_frame_latency_seconds_count", "alert_processing_seconds_sum", "alert_processing_seconds_count")

def parse(text):
    out={k:[] for k in METRICS}
    for line in text.splitlines():
        if not line or line.startswith('#'): continue
        name=line.split('{',1)[0].split(' ',1)[0]
        base=name
        for target in METRICS:
            if base==target:
                try: out[target].append(float(line.rsplit(' ',1)[1]))
                except Exception: pass
    return out

def mean(v): return statistics.fmean(v) if v else 0.0

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--api',required=True); ap.add_argument('--token',required=True); ap.add_argument('--camera-ids-file',required=True); ap.add_argument('--levels',default='5,10,25,50'); ap.add_argument('--seconds',type=int,default=300); ap.add_argument('--sample-seconds',type=float,default=5); ap.add_argument('--output',default='benchmark_results.csv'); a=ap.parse_args()
    cams=[x.strip() for x in Path(a.camera_ids_file).read_text().splitlines() if x.strip() and not x.startswith('#')]
    levels=[int(x) for x in a.levels.split(',')]
    headers={'Authorization':f'Bearer {a.token}'}; rows=[]
    with httpx.Client(base_url=a.api.rstrip('/'),headers=headers,timeout=20) as c:
        for n in levels:
            if len(cams)<n: raise SystemExit(f'Need at least {n} camera IDs')
            selected=cams[:n]
            for cam in selected: c.post(f'/camera/{cam}/start').raise_for_status()
            time.sleep(10)
            samples=[]; started=time.time()
            first=parse(c.get('/metrics').text)
            while time.time()-started<a.seconds:
                samples.append(parse(c.get('/metrics').text)); time.sleep(a.sample_seconds)
            last=parse(c.get('/metrics').text)
            fps=[mean(x['camera_fps']) for x in samples]; q=[max(x['camera_queue_depth'] or [0]) for x in samples]; gpu=[mean(x['gpu_utilization_percent']) for x in samples]; vram=[sum(x['gpu_memory_used_bytes']) for x in samples]
            dropped=max(0.0,sum(last['camera_dropped_frames_total'])-sum(first['camera_dropped_frames_total']))
            lat_count=sum(last['camera_frame_latency_seconds_count'])-sum(first['camera_frame_latency_seconds_count']); lat_sum=sum(last['camera_frame_latency_seconds_sum'])-sum(first['camera_frame_latency_seconds_sum'])
            alert_count=sum(last['alert_processing_seconds_count'])-sum(first['alert_processing_seconds_count']); alert_sum=sum(last['alert_processing_seconds_sum'])-sum(first['alert_processing_seconds_sum'])
            row={'camera_count':n,'duration_seconds':a.seconds,'avg_camera_fps':round(mean(fps),3),'max_queue_depth':round(max(q or [0]),3),'dropped_frames':round(dropped,0),'avg_frame_latency_ms':round(1000*lat_sum/lat_count,3) if lat_count>0 else None,'avg_gpu_percent':round(mean(gpu),3),'max_vram_bytes':round(max(vram or [0]),0),'avg_alert_latency_ms':round(1000*alert_sum/alert_count,3) if alert_count>0 else None}
            rows.append(row); print(json.dumps(row,indent=2))
            for cam in selected: c.post(f'/camera/{cam}/stop')
            time.sleep(5)
    with open(a.output,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f'Wrote {a.output}')
if __name__=='__main__': main()
