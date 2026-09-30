import os,time
from dataclasses import dataclass
from threading import Lock
@dataclass
class State:
    traffic_at:float=0; crowd_at:float=0; traffic_level:str='NONE'; crowd_level:str='NONE'
_states={}; _lock=Lock()
T_INFO=int(os.getenv('TRAFFIC_INFO_THRESHOLD','35')); T_LOW=int(os.getenv('TRAFFIC_LOW_THRESHOLD','55')); T_MED=int(os.getenv('TRAFFIC_MEDIUM_THRESHOLD','80')); T_HIGH=int(os.getenv('TRAFFIC_HIGH_THRESHOLD','120'))
C_INFO=int(os.getenv('CROWD_INFO_THRESHOLD','25')); C_LOW=int(os.getenv('CROWD_LOW_THRESHOLD','50')); C_MED=int(os.getenv('CROWD_MEDIUM_THRESHOLD','90')); C_HIGH=int(os.getenv('CROWD_HIGH_THRESHOLD','150'))
COOLDOWN=max(10,int(os.getenv('BEHAVIOR_ALERT_COOLDOWN_SECONDS','60')))
def evaluate_behavior(camera_id,person_count,vehicle_count,timestamp=None):
    now=float(timestamp or time.time()); st=_states.setdefault(str(camera_id),State()); out=[]
    def lvl(n,a,b,c,d): return 'HIGH' if n>=d else 'MEDIUM' if n>=c else 'LOW' if n>=b else 'INFO' if n>=a else 'NONE'
    tl=lvl(vehicle_count,T_INFO,T_LOW,T_MED,T_HIGH); cl=lvl(person_count,C_INFO,C_LOW,C_MED,C_HIGH)
    if tl!='NONE' and (tl!=st.traffic_level or now-st.traffic_at>=COOLDOWN):
        out.append({'event_type':'TRAFFIC_DENSITY','rule':'HEAVY_TRAFFIC' if tl=='INFO' else 'TRAFFIC_DENSITY_ESCALATION','level':tl,'vehicle_count':vehicle_count,'person_count':person_count,'confidence':min(1,vehicle_count/max(1,T_HIGH)),'message':f'High traffic density: {vehicle_count} vehicles detected'}) ; st.traffic_at=now
    if cl!='NONE' and (cl!=st.crowd_level or now-st.crowd_at>=COOLDOWN):
        out.append({'event_type':'CROWD_DENSITY','rule':'CROWD_BUILDUP' if cl=='INFO' else 'CROWD_DENSITY_ESCALATION','level':cl,'vehicle_count':vehicle_count,'person_count':person_count,'confidence':min(1,person_count/max(1,C_HIGH)),'message':f'Crowd buildup: {person_count} persons detected'}); st.crowd_at=now
    st.traffic_level=tl; st.crowd_level=cl
    return out
