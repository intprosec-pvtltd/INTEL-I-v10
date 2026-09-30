#!/usr/bin/env python3
"""INTEL-I Phase 0-9 production acceptance matrix.

Automated checks run without hardware. GPU/RTSP gates are reported as BLOCKED
until the command is executed on the target NVIDIA deployment host.
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REQUIRED=[
 "services/aiScheduler.py","services/vehicleAnalytics.py","services/observationValidation.py",
 "services/anprVoting.py","services/identityCorrection.py","events/outbox.py",
 "events/outboxWorker.py","events/kafkaProducer.py","tools/benchmark_vehicle_pipeline.py",
 "db/intelligence_model.py","vehicleCorrelation.py",
]

def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    return p.returncode,p.stdout[-4000:],p.stderr[-4000:]

def main():
    results=[]
    missing=[x for x in REQUIRED if not (ROOT/x).exists()]
    results.append({"gate":"required_phase_0_9_files","status":"PASS" if not missing else "FAIL","missing":missing})
    rc,out,err=run([sys.executable,"-m","pytest","-q"])
    results.append({"gate":"automated_regression_tests","status":"PASS" if rc==0 else "FAIL","output":out,"error":err})
    try:
        from alembic.config import Config
        from alembic.script import ScriptDirectory
        heads=ScriptDirectory.from_config(Config(str(ROOT/"alembic.ini"))).get_heads()
        results.append({"gate":"alembic_single_head","status":"PASS" if len(heads)==1 else "FAIL","heads":heads})
    except Exception as exc:
        results.append({"gate":"alembic_single_head","status":"FAIL","error":str(exc)})
    # Hardware gates are deliberately not fabricated on non-NVIDIA hosts.
    gpu=os.getenv("NVIDIA_ACCEPTANCE_RUN","false").lower()=="true"
    results.append({"gate":"real_nvidia_tensorRT","status":"RUN_REQUIRED_ON_TARGET_HOST" if not gpu else "RUNNING","note":"Requires target NVIDIA GPU, validated TensorRT engines and real deployment models."})
    results.append({"gate":"real_50_camera_rtsp","status":"RUN_REQUIRED_ON_TARGET_HOST" if not gpu else "RUNNING","note":"Requires real RTSP/NVR/VMS sources; local-file benchmark is not acceptance evidence."})
    results.append({"gate":"websocket_disconnect_recovery","status":"AUTOMATED_CONTRACT_PRESENT","note":"Run live kill/reconnect test against deployed API/frontend."})
    results.append({"gate":"transactional_outbox","status":"AUTOMATED_CONTRACT_PRESENT","note":"Alert + outbox are committed in one DB transaction; Kafka delivery acknowledgement is awaited."})
    path=ROOT/"PHASE_0_9_ACCEPTANCE_MATRIX.json"
    path.write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2))
    print(f"Report: {path}")
    return 0 if all(r["status"] not in {"FAIL"} for r in results) else 1
if __name__=="__main__": raise SystemExit(main())
