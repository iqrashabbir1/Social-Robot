# How To Run Real-Anchor CS2 and CS3

## CS2 real-anchor synchronization
```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python -m src.data.run_cs2 --project-root . --config configs/cs2/real_anchor_video_audio.yaml
```

## CS3 pilot real-anchor baseline
```powershell
python -m src.models.classical.train_classical --project-root . --config configs/cs3/real_anchor_baseline.yaml
```

## Notes
- CS2 real-anchor is a pilot demonstration.
- CS3 real-anchor currently demonstrates baseline inference only.
- These runs do not claim generalizable benchmark performance.
