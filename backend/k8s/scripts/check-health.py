import json, sys
try:
    data = json.load(sys.stdin)
    ready = str(data.get("status", "")).upper() == "READY"
except (ValueError, TypeError, AttributeError):
    ready = False
sys.exit(0 if ready else 1)
