"""Real deployed face-watchlist uploaded-video E2E acceptance.
Requires a test account/token, reference image and video containing that person.
"""
import argparse, json, time
from pathlib import Path
import httpx

def main():
    p=argparse.ArgumentParser(); p.add_argument('--api',required=True); p.add_argument('--token',required=True); p.add_argument('--reference',required=True); p.add_argument('--video',required=True); p.add_argument('--name',default='INTEL-I E2E Test Person'); p.add_argument('--timeout',type=int,default=180); a=p.parse_args()
    h={'Authorization':f'Bearer {a.token}'}
    with httpx.Client(base_url=a.api.rstrip('/'),headers=h,timeout=600) as c:
        with open(a.reference,'rb') as f:
            r=c.post('/api/intelligence/person-watchlist/enroll',data={'full_name':a.name,'category':'OTHER','status':'ACTIVE','description':'automated E2E acceptance','source':'P0_ACCEPTANCE'},files={'image':(Path(a.reference).name,f,'image/jpeg')}); r.raise_for_status(); enrolled=r.json()
        print('ENROLLED',json.dumps(enrolled,default=str)[:1000])
        with open(a.video,'rb') as f:
            r=c.post('/video-upload',data={'camera_name':'FACE_E2E_UPLOAD','location_name':'P0 acceptance'},files={'videoFile':(Path(a.video).name,f,'video/mp4')}); r.raise_for_status(); upload=r.json()
        cam=str(upload.get('camID')); print('UPLOAD',json.dumps(upload,default=str)[:1000])
        deadline=time.time()+a.timeout; hit=None
        while time.time()<deadline:
            rr=c.get('/alerts',params={'limit':500}); rr.raise_for_status()
            for alert in rr.json().get('alerts',[]):
                if str(alert.get('cam_id') or alert.get('camera_id'))==cam and (alert.get('rule') in {'PERSON_WATCHLIST_MATCH','PERSON_WATCHLIST_WANTED','PERSON_WATCHLIST_MISSING'} or alert.get('event')=='PERSON_WATCHLIST_MATCH'):
                    hit=alert; break
            if hit: break
            time.sleep(3)
        if not hit: raise SystemExit('FAIL: no persisted person-watchlist alert before timeout')
        required=['id','cam_id']; missing=[x for x in required if not hit.get(x)]
        if missing: raise SystemExit(f'FAIL: alert missing {missing}')
        print('PASS: persisted face watchlist alert',json.dumps(hit,default=str)[:1500])
        print('NEXT: verify matching snapshot, Incident, IncidentEvidence and WebSocket in the application/staging DB. This script never bypasses tenant APIs or mutates DB directly.')
if __name__=='__main__': main()
