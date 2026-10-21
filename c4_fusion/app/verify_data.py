"""Fast local preflight (does not import Streamlit or disclose private stage values)."""
from pathlib import Path
from data_service import load_workspace, SourceDataError, repo_root_from_app

if __name__ == '__main__':
    root = repo_root_from_app()
    print('FloodRisk C4 dashboard preflight')
    print('Repository found:', root)
    try:
        ws = load_workspace(root)
    except (SourceDataError, FileNotFoundError, ValueError, KeyError) as exc:
        print('\nFAILED:', exc)
        raise SystemExit(1)
    print('PASS — Station map:', len(ws.stations), 'private WGS84 points')
    print('PASS — Private source:', len(ws.observations), 'labelled records')
    print('PASS — Models:', ', '.join(ws.available_models))
    print('PASS — Predictions:', len(ws.predictions), 'model/station-day rows')
    print('PASS — Common TEST set: 351 station-days, 39 gauges, 9 dates')
    print('PASS — Gauge-specific alert/minor/major thresholds, identical comparison records')
    if ws.aligned_index is None:
        print('OPTIONAL — Aligned QA index missing; detailed acquisition provenance will be hidden')
    else:
        print('PASS — Optional acquisition and spatial QA index loaded')
    if ws.missing_models:
        print('WARNING — Some models are not present:', ', '.join(ws.missing_models))
    print('READY — Run: .\\.venv\\Scripts\\python.exe -m streamlit run c4_fusion\\app\\app.py')
