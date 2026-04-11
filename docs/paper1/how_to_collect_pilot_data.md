# How To Collect Pilot Data

```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python scripts\check_local_devices.py
python -m src.data.collect_pilot_session --project-root . --session-name paper1_anchor_demo --duration-seconds 3 --frame-interval-ms 250
```

Outputs are written under:
- `data/pilot/sessions/paper1_anchor_demo/`
