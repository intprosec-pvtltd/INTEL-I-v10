# Final submission recording runbook

Record two distinct 2–3 minute videos at 1920×1080 or the evaluator-required
resolution. Close unrelated windows and notifications. Never reveal passwords,
API tokens, RTSP URLs, encryption keys or browser developer tools containing
request headers.

## Government/Sentinel recording

1. Start recording before opening the INTEL-I login screen.
2. Log in and open **Camera Setup → Sentinel / VMS**.
3. Show the configured Sentinel integration (origin only), run **Discover** and
   **Sync now**, then show the imported camera ID, live status, codec,
   resolution and location. Do not expose a stream URL.
4. Start one real Government camera and show the live feed plus its camera
   health with `SOURCE_PTS`/`PTS_ALIGNED` timing.
5. Show a real vehicle detection, ANPR result, timestamp, watchlist decision,
   alert and snapshot evidence.
6. Open **Analytics Reports**, show the record and download PDF/CSV.
7. Open `/api/reports/government-e2e/readiness?hours=24` through the authorized
   UI/API workflow and show all evidence gates green.

## Own-feed recording

1. Upload or configure the approved own-feed clip without changing the existing
   workflow.
2. Show the camera starting, detections/ANPR/watchlist handling and evidence.
3. Show a report filtered to that camera, then stop the stream cleanly.

Linux:

```bash
./scripts/record_submission_demo.sh government 180
./scripts/record_submission_demo.sh own-feed 180
```

Windows PowerShell:

```powershell
.\scripts\record_submission_demo.ps1 -Mode government -DurationSeconds 180
.\scripts\record_submission_demo.ps1 -Mode own-feed -DurationSeconds 180
```

Play each output from beginning to end before submission. Confirm readable text,
smooth playback, no secrets and no unrelated personal data. Record the SHA-256
in the package manifest.
