#!/usr/bin/env python3
"""C1 PP1 evidence pack. Reads existing results; never retrains/tunes models.

Expected to run at repository root:
  python c1_satellite/scripts/15_pp1_evidence.py
Outputs under c1_satellite/outputs/pp1_evidence/
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
DEST = OUT / 'pp1_evidence'


def safe_json(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run Phase 12 and Phase 13 first.')
    return json.loads(path.read_text(encoding='utf-8'))


def read_model_metrics(path):
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run Phase 14 first.')
    df = pd.read_csv(path)
    required = {'model_id','split','n','MAE_m','RMSE_m'}
    if missing := required - set(df.columns):
        raise ValueError(f'Missing validation metrics columns: {sorted(missing)}')
    df = df[df.split == 'VALIDATION'].copy()
    if df.empty:
        raise ValueError('No validation rows; refusal to plot TEST as validation')
    if df.model_id.duplicated().any():
        raise ValueError('Duplicate model IDs in validation data')
    for k in ['n','MAE_m','RMSE_m']:
        df[k] = pd.to_numeric(df[k], errors='raise')
    if df[['MAE_m','RMSE_m']].isna().any().any() or (df[['MAE_m','RMSE_m']] < 0).any().any():
        raise ValueError('Invalid metric')
    if df.n.nunique() != 1:
        raise ValueError('Cannot call results fair/same-cohort: n differs by model')
    return df


def plot_validation(df, metric, output):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    vals = df[metric].to_numpy(dtype=float)
    labels = df.model_id.astype(str).tolist()
    bars = ax.bar(labels, vals)
    ax.set_title(f'C1 two-band sensitivity: VALIDATION {metric.replace("_m", "")} (lower is better)')
    ax.set_ylabel('Prediction error (metres)')
    ax.set_xlabel('Model')
    ax.set_ylim(0, max(vals) * 1.22 if vals.max()>0 else 1)
    ax.grid(axis='y', alpha=0.25)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.025, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    fig.text(0.5,0.015, f'Same {int(df.n.iloc[0])} validation station-day records; not a flood-warning accuracy claim', ha='center', fontsize=9)
    fig.tight_layout(rect=[0,0.04,1,1])
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_swi_coverage(summary, output):
    splits = ('TRAIN','VALIDATION','TEST')
    coverage = summary['split_coverage']
    names = ('SWI001','SWI005','SWI010','All three')
    data = []
    for s in splits:
        r = coverage[s]
        n = int(r['labelled_n'])
        if n <= 0: raise ValueError(f'Invalid labelled_n for {s}')
        by = r['complete_7_by_band']
        data.append([100*int(by['001'])/n, 100*int(by['005'])/n,
                     100*int(by['010'])/n,100*int(r['complete_7_all_three'])/n])
    arr = np.array(data)
    fig, ax = plt.subplots(figsize=(9,5.1))
    x=np.arange(len(splits))
    for i,name in enumerate(names):
        ax.bar(x + (i-1.5)*0.19, arr[:,i], width=0.19, label=name)
    ax.set_xticks(x, splits)
    ax.set_ylabel('Complete seven-day histories (%)')
    ax.set_ylim(0,112)
    ax.set_title('Per-band observed seven-day histories (provisional QFLAG threshold)')
    ax.legend(loc='upper center', ncol=4, fontsize=9)
    ax.grid(axis='y',alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output,dpi=170)
    plt.close(fig)


def plot_stage_regime(summary,output):
    d={v['split']: v for v in summary['daily_stage_change_by_split']}
    splits=('TRAIN','VALIDATION','TEST')
    values=[float(d[s]['mean_abs_stage_change_m']) for s in splits]
    fig,ax=plt.subplots(figsize=(7.2,4.9))
    bars=ax.bar(splits,values)
    ax.set_title('Observed daily river-stage changes across chronological splits')
    ax.set_ylabel('Mean absolute 24-hour stage change (m)')
    ax.set_ylim(0,max(values)*1.25)
    ax.grid(axis='y',alpha=.25)
    ax.set_axisbelow(True)
    for bar,v in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(values)*.03,f'{v:.3f} m',ha='center')
    fig.tight_layout()
    fig.savefig(output,dpi=170)
    plt.close(fig)


def md_report(swi, diagnostic, df):
    splits=('TRAIN','VALIDATION','TEST')
    lines=['# C1 PP1 evidence: satellite depth prototype','',
      '**Scope:** retrospective river-stage-change prediction, using official DMC station-day labels, actual Sentinel-1-derived numerical SAR features, and Copernicus ASCAT SWI.','',
      '## Research objective','',
      'Compare latest vs multi-temporal Sentinel-1 data and latest vs seven-day ASCAT soil-wetness features using controlled regression experiments. The revised output is next-day river stage in metres, not a flood-extent mask. Supervisor approval for this change is needed.','',
      '## Seven-day ASCAT quality coverage','',
      '| Chronological split | Labelled records | SWI001 full 7 days | SWI005 full 7 days | SWI010 full 7 days | 3 bands full 7 days |',
      '|---|---:|---:|---:|---:|---:|']
    for s in splits:
        r=swi['split_coverage'][s]; b=r['complete_7_by_band']
        lines.append(f"| {s} | {r['labelled_n']} | {b['001']} | {b['005']} | {b['010']} | {r['complete_7_all_three']} |")
    lines+=['','**Important:** The new A/B/C/D experiments are a **two-band SWI005+SWI010 sensitivity study**; not the original three-band architecture. QFLAG ≥50 and the 2-day lag remain provisional and should not be presented as operational verification.','',
        '## Validation results: identical eligible station-days','',
        '| Model | n | MAE (m) | RMSE (m) |','|---|---:|---:|---:|']
    for r in df.itertuples(index=False):
        lines.append(f'| {r.model_id} | {int(r.n)} | {r.MAE_m:.4f} | {r.RMSE_m:.4f} |')
    persistence=df[df.model_id=='PERSISTENCE']
    best=df.sort_values('MAE_m').iloc[0]
    if not persistence.empty:
        p=float(persistence.iloc[0].MAE_m)
        lines+=['',f'**Observed conclusion:** Lowest validation MAE is {best.model_id} ({best.MAE_m:.4f} m). Persistence MAE is {p:.4f} m. Satellite input has **not yet shown improvement over persistence** on this validation cohort.' if best.model_id=='PERSISTENCE' else f'**Observed conclusion:** {best.model_id} has lowest validation MAE {best.MAE_m:.4f} m; persistence is {p:.4f} m. This needs confirmation on a fresh event before operational claims.']
    lines+=['','## Distribution differences (descriptive, not a causal explanation)','',
        '| Split | Number of labels | Mean abs stage change (m) | Median abs change (m) |','|---|---:|---:|---:|']
    for r in diagnostic['daily_stage_change_by_split']:
        lines.append(f"| {r['split']} | {r['n']} | {r['mean_abs_stage_change_m']:.4f} | {r['median_abs_stage_change_m']:.4f} |")
    lines+=['','## Real implementation evidence to demonstrate','',
        '- Show a real downloaded VV/VH `.tif` and its image preview; explain TIFF is a scientific raster, not a text file.',
        '- Show station-day keyed real SAR summary statistics and ASCAT daily/lagged values with masks, rather than asserting full-image CNN training.',
        '- Explain that each arm uses the same eligible rows, chronological train/validation/test boundaries, and a validation-chosen Ridge hyperparameter.',
        '- Explain why persistence is an appropriate benchmark for next-day river stage.',
        '', '## Unresolved limitations and next scientific steps','',
        '- Same satellite acquisitions and coarse ASCAT grid cells contribute to multiple station-day rows; observations are correlated.',
        '- Data cover a short historical window, and the held-out period is not independent evidence of unseen major-flood performance.',
        '- The full U-Net-style image encoder is **not** implemented in this feature-based Ridge proof of concept.',
        '- Source observation time is not necessarily public product availability time; two-day lag is a provisional assumption.',
        '- Confirm upstream catchment geometry quality. Model target is gauge level change, not pixel-level flood depth.',
        '- Use TRAIN and VALIDATION to plan improvements; prior TEST outcomes have been inspected. A new independent period is required for stronger final testing.',
        '', '## PP1 speaker summary (Tanglish)','',
        '> Naan C1 satellite temporal-depth component implement panren. Real Sentinel-1 SAR VV/VH scenes and ASCAT SWI features collect panni 39 gauges-oda DMC station-day labels-oda join pannirukken. Four controlled feature-based regression experiments conduct pannom. SWI001 QFLAG quality gap nala two-band SWI005+SWI010 seven-day sensitivity analysis pannom. Current VALIDATION-la persistence baseline better perform pannuthu; satellite advantage innum prove aagala. Ithu preliminary retrospective proof-of-concept; final U-Net encoder and independent event validation future work.',
        '', '## Figures','',
        '1. `01_validation_MAE.png`', '2. `02_validation_RMSE.png`', '3. `03_swi_history_coverage.png`', '4. `04_stage_regime_comparison.png`','']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default=None,help='Optional report destination; default c1_satellite/outputs/pp1_evidence')
    opts=p.parse_args()
    dest=Path(opts.output_dir) if opts.output_dir else DEST
    swi=safe_json(OUT/'c1_swi_bandwise_7day_coverage.json')
    diagnostic=safe_json(OUT/'c1_phase12_diagnostics'/'phase12_summary.json')
    df=read_model_metrics(OUT/'c1_bandwise_validation_metrics.csv')
    dest.mkdir(parents=True,exist_ok=True)
    plot_validation(df,'MAE_m',dest/'01_validation_MAE.png')
    plot_validation(df,'RMSE_m',dest/'02_validation_RMSE.png')
    plot_swi_coverage(swi,dest/'03_swi_history_coverage.png')
    plot_stage_regime(diagnostic,dest/'04_stage_regime_comparison.png')
    (dest/'C1_PP1_EVIDENCE.md').write_text(md_report(swi,diagnostic,df),encoding='utf-8')
    print('C1 PP1 EVIDENCE GENERATED — READ ONLY INPUTS')
    print('Output directory:',dest)
    print('Validation models:',', '.join(df.model_id.astype(str)))
    print('Identical validation n:',int(df.n.iloc[0]))
    print('Files: 4 PNG figures + C1_PP1_EVIDENCE.md')
    print('No models were trained, scores altered, or TEST results used for decisions.')

if __name__=='__main__': main()
