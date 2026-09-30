"""Failure-recovery acceptance recorder for a deployed INTEL-I environment.

This script records readiness before/during/after an operator-induced failure.
It intentionally does not stop production services itself.
"""
import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path
import httpx

def get(c,path):
    try:
        r=c.get(path); return {'status_code':r.status_code,'body':r.json() if 'json' in r.headers.get('content-type','') else r.text[:1000]}
    except Exception as e: return {'error':type(e).__name__}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--api',required=True); p.add_argument('--scenario',choices=['camera_outage','db_restart','kafka_outage','ollama_outage','gpu_overload','frontend_restart','websocket_disconnect'],required=True); p.add_argument('--observe-seconds',type=int,default=60); p.add_argument('--output',default='failure_acceptance.json'); a=p.parse_args()
    result={'scenario':a.scenario,'started_at':datetime.now(timezone.utc).isoformat(),'before':None,'during':[],'after':None}
    with httpx.Client(base_url=a.api.rstrip('/'),timeout=10) as c:
        result['before']=get(c,'/ready')
        print(f'Induce scenario NOW: {a.scenario}. The harness will observe /ready for {a.observe_seconds}s.')
        start=time.time()
        while time.time()-start<a.observe_seconds:
            result['during'].append({'t':round(time.time()-start,1),'ready':get(c,'/ready')}); time.sleep(5)
        print('Restore the dependency now; waiting 20 seconds for recovery...'); time.sleep(20)
        result['after']=get(c,'/ready')
    Path(a.output).write_text(json.dumps(result,indent=2,default=str)); print(f'Wrote {a.output}')
if __name__=='__main__': main()
