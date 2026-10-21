"""FloodRisk Sri Lanka — C4 historical river-stage research explorer.

Run from repository root:
    python -m streamlit run c4_fusion/app/app.py

This deliberately DOES NOT run live inference, Earth Engine, or official warning logic.
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import timedelta
import html
from contextlib import nullcontext

import altair as alt

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from dotenv import dotenv_values

# Streamlit starts the script in a module-like environment, so import siblings reliably.
from data_service import (
    MODELS, MODEL_NAMES, C4Workspace, SourceDataError,
    classify_stage, load_workspace, repo_root_from_app,
)
from map_service import (
    build_station_map, extract_station_id, load_candidate_polygon, tile_options,
)

ROOT = repo_root_from_app()
APP = Path(__file__).resolve().parent

st.set_page_config(
    page_title='FloodRisk Sri Lanka · Historical C4 Explorer',
    page_icon='🌊',
    layout='wide',
    initial_sidebar_state='expanded',
    menu_items={
        'About': 'FloodRisk Sri Lanka — C4 retrospective research prototype. Not an official warning service.'
    },
)
css_file = APP / 'theme.css'
if css_file.is_file():
    st.markdown(f'<style>{css_file.read_text(encoding="utf-8")}</style>', unsafe_allow_html=True)


def get_file_fingerprint(root: Path) -> tuple[tuple[str, int, int], ...]:
    """Invalidate data cache when model predictions or source files change."""
    checked = [
        root / 'c4_fusion/private_data/stations_40_PRIVATE.geojson',
        root / 'c4_fusion/private_data/observations/FloodRisk_DMC_StationDay_LABELLED_1429.csv',
        root / 'data/processed/c4/multimodal/c4_aligned_index_1429_PRIVATE.csv',
        *(root / f'results/pp1/c4_neural/M{i}_test_predictions.csv' for i in range(1, 8)),
    ]
    return tuple((str(p.relative_to(root)), p.stat().st_mtime_ns if p.exists() else -1,
                  p.stat().st_size if p.exists() else -1) for p in checked)


@st.cache_data(show_spinner='Loading validated historical predictions…')
def load_validated_workspace(root_as_str: str, fingerprint) -> C4Workspace:
    return load_workspace(Path(root_as_str))


@st.cache_data(show_spinner=False)
def read_catchment_polygon(root_as_str: str, relative: str, mtime: int):
    return load_candidate_polygon(Path(root_as_str), relative)


def format_model(model: str) -> str:
    return MODEL_NAMES.get(model, model)


def metres(v: float, decimals: int = 2) -> str:
    return f'{float(v):.{decimals}f} m'


def threshold_class(status: str) -> str:
    return 'fr-normal' if status == 'NORMAL' else 'fr-warn' if status == 'ALERT' else 'fr-critical'


def status_html(status: str) -> str:
    css = threshold_class(status)
    return f'<span class="{css}">{html.escape(status.title())}</span>'


def show_stage_chart(ws: C4Workspace, station_id: str, model: str):
    """Readable, responsive light-theme chart of nine held-out target dates."""
    frame = ws.station_series(station_id, model).copy()
    if frame.empty:
        st.info('No saved test predictions for this station.')
        return
    frame['target_date'] = pd.to_datetime(frame['forecast_date']) + pd.Timedelta(days=1)
    frame = frame.sort_values('target_date')
    frame['Date'] = frame['target_date'].dt.strftime('%d %b')
    series = [
        ('target_stage_m', 'Actual observed'),
        ('pred_stage_m', 'Model prediction'),
        ('current_stage_m', 'Persistence baseline'),
    ]
    tidy = pd.concat([
        frame[['Date', col]].rename(columns={col:'Stage (m)'}).assign(Series=name)
        for col, name in series
    ], ignore_index=True)
    order = frame['Date'].tolist()
    names = [label for _, label in series]
    chart = (
        alt.Chart(tidy)
        .mark_line(point=alt.OverlayMarkDef(filled=True, size=54), strokeWidth=2.7)
        .encode(
            x=alt.X('Date:N', sort=order,
                    axis=alt.Axis(title='Next-day observation (09:00 SLST)',
                                  labelAngle=0, labelPadding=9, grid=False)),
            y=alt.Y('Stage (m):Q', scale=alt.Scale(zero=False, nice=True),
                    axis=alt.Axis(title='River stage (m)', tickCount=5)),
            color=alt.Color('Series:N', sort=names,
                            scale=alt.Scale(domain=names,
                                            range=['#102B46','#1683A8','#8798AC']),
                            legend=alt.Legend(orient='bottom', title=None,
                                              labelFontSize=12, symbolSize=100)),
            tooltip=[alt.Tooltip('Date:N'), alt.Tooltip('Series:N'),
                     alt.Tooltip('Stage (m):Q', format='.3f')],
        )
        .properties(height=290, background='#FFFFFF')
        .configure_view(stroke=None, fill='#FFFFFF')
        .configure_axis(labelColor='#334F66', titleColor='#102B46',
                        gridColor='#E4ECF3', labelFontSize=12, titleFontSize=12)
        .configure_legend(labelColor='#334F66', labelFontSize=12)
    )
    # st.altair_chart supports theme=None. st.line_chart()/st.bar_chart() do NOT.
    st.altair_chart(chart, theme=None, use_container_width=True)
    st.caption('Nine held-out historical predictions at 09:00 SLST. Dates on the horizontal axis are target observation dates (T+1). The lines join daily values for readability.')


def render_prediction_result(row: pd.Series | None, model: str, station_name: str):
    st.markdown('<div class="fr-section-title">Historical prediction</div>', unsafe_allow_html=True)
    if row is None:
        st.info('No held-out prediction for this station/date. The station remains available on the map.')
        return
    current, actual, predicted = (float(row[x]) for x in ('current_stage_m','target_stage_m','pred_stage_m'))
    error = abs(predicted-actual)
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Stage at origin · T 09:00', metres(current))
    with col2:
        st.metric('Predicted stage · T+1 09:00', metres(predicted))
    col3, col4 = st.columns(2)
    with col3:
        st.metric('Actual observed · T+1', metres(actual))
    with col4:
        st.metric('This prediction’s error', metres(error, 3))
    st.caption(f'{format_model(model)}. Error is calculated against the later historical reading at {station_name}; it is not a forecast confidence interval.')
    alert, minor, major = (float(row[n]) for n in ('alert_threshold_m','minor_flood_threshold_m','major_flood_threshold_m'))
    predicted_status = classify_stage(predicted, alert, minor, major)
    observed_status = classify_stage(actual, alert, minor, major)
    st.markdown('**River-level threshold categories**')
    c1, c2 = st.columns(2)
    with c1:
        st.caption('Model-implied level')
        st.markdown(status_html(predicted_status), unsafe_allow_html=True)
    with c2:
        st.caption('Observed historical level')
        st.markdown(status_html(observed_status), unsafe_allow_html=True)
    st.caption('Computed using this gauge’s three Irrigation Department-derived thresholds. These are NOT official warnings or inundation maps.')


def render_officer_comparison(ws: C4Workspace, station_id: str, forecast_date: str):
    frame = ws.model_comparison(station_id, forecast_date)
    if frame.empty:
        st.info('No held-out predictions are available for this station/date.')
        return
    data = frame[['model','pred_stage_m','target_stage_m','abs_error_m','signed_error_m']].copy()
    data['Model'] = data.model.map(format_model)
    data['Predicted stage (m)'] = data.pred_stage_m
    data['Absolute error (m)'] = data.abs_error_m
    data['Signed error (m)'] = data.signed_error_m
    st.dataframe(data[['Model','Predicted stage (m)','Absolute error (m)','Signed error (m)']],
                 hide_index=True, width='stretch',
                 column_config={
                     'Predicted stage (m)': st.column_config.NumberColumn(format='%.3f'),
                     'Absolute error (m)': st.column_config.NumberColumn(format='%.3f'),
                     'Signed error (m)': st.column_config.NumberColumn(format='%.3f'),
                 })
    ordered = data['Model'].tolist()
    error_chart = (
        alt.Chart(data)
        .mark_bar(size=20, cornerRadiusEnd=4, color='#1683A8')
        .encode(
            y=alt.Y('Model:N', sort=ordered, title=None,
                    axis=alt.Axis(labelLimit=220, labelFontSize=12)),
            x=alt.X('Absolute error (m):Q', title='Absolute error (m)',
                    scale=alt.Scale(zero=True), axis=alt.Axis(tickCount=5)),
            tooltip=['Model:N', alt.Tooltip('Absolute error (m):Q', format='.3f')],
        )
        .properties(height=310, background='#FFFFFF')
        .configure_view(stroke=None, fill='#FFFFFF')
        .configure_axis(labelColor='#334F66', titleColor='#102B46', gridColor='#E4ECF3')
    )
    st.altair_chart(error_chart, theme=None, use_container_width=True)
    st.caption('For this one historical record, lower absolute error is better. The comparison does not establish statistical superiority.')


def render_officer_metrics(ws: C4Workspace, selected_model: str):
    scores = ws.metrics.copy()
    scores['model_rank'] = scores.model.map({**{m:i for i,m in enumerate(MODELS)}, 'PERSISTENCE':7})
    scores['Model'] = scores.model.map(format_model)
    scores = scores.sort_values('model_rank')
    st.dataframe(scores[['Model','n','mae_m','rmse_m']].rename(columns={
        'n':'Test records','mae_m':'Test MAE (m)','rmse_m':'Test RMSE (m)'
    }), hide_index=True, width='stretch',
       column_config={
           'Test MAE (m)': st.column_config.NumberColumn(format='%.4f'),
           'Test RMSE (m)': st.column_config.NumberColumn(format='%.4f'),
       })
    selected_row = scores.loc[scores.model.eq(selected_model)]
    if not selected_row.empty:
        st.caption(f'{format_model(selected_model)} — test MAE = {selected_row.iloc[0].mae_m:.4f} m across {int(selected_row.iloc[0].n)} held-out records. This is different from the selected station-day error above.')
    st.caption('The TEST period is easier than the training period in terms of observed absolute next-day river-stage changes. Major-flood test records: 0. Do not interpret these scores as verified operational warning accuracy.')


def render_data_qa(ws: C4Workspace, row: pd.Series | None, station: pd.Series):
    st.markdown('**Source, time and quality evidence**')
    if row is None:
        st.info('This mapped gauge has no current held-out prediction record; it is not among the 39 labelled stations.')
        return
    rid = str(row['record_id'])
    origin = pd.to_datetime(row['forecast_date']).strftime('%d %b %Y')
    st.write(f'**Gauge:** {station.station_name} · **River basin:** {station.basin}')
    st.write(f'**Forecast origin:** {origin} 09:00 SLST (simulated historical origin)')
    st.write(f'**Target time:** {(pd.to_datetime(row["forecast_date"])+pd.Timedelta(days=1)).strftime("%d %b %Y")} 09:00 SLST')
    st.caption('DMC nominal public reports were approximately 09:30. This demonstration assumes the 09:00 stage is already known at its simulated origin; it does not reproduce real publication latency.')
    qa = row.get('catchment_qa_status', 'CANDIDATE (unverified)')
    st.write('**Catchment status:**', str(qa))
    st.caption('The highlighted polygon is a provisional upstream contributing area, not a predicted flood extent. Gauge coordinates and modelled outlet positions may differ.')
    provenance = ws.provenance(rid)
    if provenance is None:
        st.info('Aligned source index is optional. To display SAR/SWI acquisition provenance, keep c4_aligned_index_1429_PRIVATE.csv in data/processed/c4/multimodal/.')
        return
    info = [
        ('Sentinel-1 acquisition (UTC)', 'sar_acquisition_utc'),
        ('SAR age at simulated origin (hours)', 'sar_age_hours'),
        ('SAR valid-pixel fraction', 'sar_valid_pixel_fraction'),
        ('SAR scene', 'sar_scene_id'),
        ('SWI source day', 'swi_source_date'),
        ('SWI age (days)', 'swi_age_days'),
        ('SWI catchment valid fraction', 'area_valid_fraction'),
        ('Spatial source QA', 'catchment_qa_status'),
        ('Combined feature complete', 'feature_complete'),
    ]
    details = [{'Data quality field': label, 'Source value': str(provenance.get(field,''))}
               for label, field in info]
    st.dataframe(pd.DataFrame(details), hide_index=True, width='stretch')
    st.caption('ERA5-Land weather: fourteen consecutive historical 24-hour sequences using source records ending no later than the simulated 09:00 SLST origin. ASCAT SWI: latest quality-accepted past daily observation. Source completeness is not proof of operational product availability by 09:00.')


# Load existing local data, fail visibly and early if a source isn't where expected.
try:
    ws = load_validated_workspace(str(ROOT), get_file_fingerprint(ROOT))
except (SourceDataError, FileNotFoundError, ValueError, KeyError) as e:
    st.error('C4 historical research data could not be loaded.')
    st.code(str(e), language='text')
    st.info('Keep the files where Notebook 07 saved them; no CSVs need to be copied into the app folder. See the installation instructions shared with this app.')
    st.stop()

# Read the optional CARTO key from known *local* locations. Never show the key or
# put it in source control. In the user's existing repo it lives in notebooks/secrets.env.
def resolve_carto_key(root: Path) -> tuple[str, str]:
    key = os.getenv('FLOODRISK_CARTO_BASEMAP_KEY', '').strip()
    if key:
        return key, 'environment variable'
    candidates = (
        root / 'notebooks' / 'secrets.env',  # existing research notebooks location
        root / 'secrets.env',                # also support repo-root secrets
        root / 'c4_fusion' / 'notebooks' / 'secrets.env',
    )
    for candidate in candidates:
        if candidate.is_file():
            values = dotenv_values(candidate)
            key = str(values.get('FLOODRISK_CARTO_BASEMAP_KEY') or '').strip()
            if key:
                return key, candidate.relative_to(root).as_posix()
    return '', 'not configured'

carto_key, carto_key_location = resolve_carto_key(ROOT)

# Map clicks are handled BEFORE any keyed selection widgets are created.
if '_pending_map_station' in st.session_state:
    target_sid = st.session_state.pop('_pending_map_station')
    if target_sid in set(ws.stations.station_id):
        station_basin = ws.stations.loc[ws.stations.station_id.eq(target_sid), 'basin'].iloc[0]
        st.session_state['c4_river'] = station_basin
        st.session_state['c4_station'] = target_sid

# Visible controls fallback: Streamlit can remember a collapsed native sidebar.
# This button always works, even if its native reopen chevron is hard to find.
if 'c4_inline_controls' not in st.session_state:
    st.session_state['c4_inline_controls'] = False
inline = bool(st.session_state['c4_inline_controls'])
heading, utility = st.columns([5, 1.6], gap='medium', vertical_alignment='center')
with heading:
    st.markdown('<div class="fr-eyebrow">Hydrological intelligence · C4 research component</div>', unsafe_allow_html=True)
    st.title('FloodRisk Sri Lanka')
with utility:
    toggle_text = '↩ Use sidebar' if inline else '☰ Show controls here'
    if st.button(toggle_text, key='c4_controls_switch', use_container_width=True,
                 help='If the left sidebar is hidden, display all selections here instead.'):
        st.session_state['c4_inline_controls'] = not inline
        st.rerun()
st.markdown('<div class="fr-subtitle">River-stage prediction explorer · Satellite × Weather × Environment</div>', unsafe_allow_html=True)
st.markdown('<div class="fr-banner">Historical research demonstration · Not a live forecast or official flood-warning service · 17–25 December 2025 · 09:00 SLST origins</div>', unsafe_allow_html=True)

basins = ['All river basins'] + sorted(ws.stations.basin.unique().tolist())
if st.session_state.get('c4_river') not in basins:
    st.session_state['c4_river'] = 'All river basins'

# There is only ONE instance of each keyed widget, either in the left sidebar
# or in the clearly labelled fallback controls panel at the top of the page.
controls_host = st.container(border=True) if inline else st.sidebar
with controls_host:
    if inline:
        st.markdown('### Explore river predictions · On-page controls')
        st.caption('You can use these selections even when the Streamlit sidebar is collapsed. Click “Use sidebar” above to return them to the left.')
    else:
        st.markdown('<div class="fr-sidebar-brand">🌊 &nbsp; FLOODRISK</div>', unsafe_allow_html=True)
        st.markdown('<div class="fr-sidebar-desc">Choose a research workspace, select a gauge, and explore recorded predictions.</div>', unsafe_allow_html=True)
    st.markdown('<div class="fr-sidebar-step">01 &nbsp; WORKSPACE</div>', unsafe_allow_html=True)
    view = st.radio('Choose your workspace', ['Public Explorer', 'Officer / Research Console'],
                    key='c4_view', horizontal=inline,
                    help='Public: simple historical prediction. Officer: seven-model comparison, thresholds and provenance.')
    if view == 'Public Explorer':
        st.caption('Simple historical forecast explorer. Advanced research options are optional.')
    else:
        st.caption('Research console: model comparison, station thresholds and data-quality evidence.')

    st.markdown('<div class="fr-sidebar-step">02 &nbsp; LOCATION</div>', unsafe_allow_html=True)
    loc1, loc2 = st.columns(2) if inline else (None, None)
    with (loc1 if inline else nullcontext()):
        basin = st.selectbox('River basin', basins, key='c4_river',
                             help='Choose one basin, or keep all stations visible.')
    filtered = ws.stations if basin == 'All river basins' else ws.stations.loc[ws.stations.basin.eq(basin)]
    station_options = filtered.station_id.tolist()
    if not station_options:
        st.error('No gauges found in this river basin.'); st.stop()
    preferred = ('hanwella' if 'hanwella' in station_options and 'hanwella' in ws.available_station_ids()
                 else next((sid for sid in station_options if sid in ws.available_station_ids()), station_options[0]))
    if st.session_state.get('c4_station') not in station_options:
        st.session_state['c4_station'] = preferred
    with (loc2 if inline else nullcontext()):
        station_id = st.selectbox('Gauge station', station_options, key='c4_station',
                                  format_func=lambda sid: ws.stations.loc[
                                      ws.stations.station_id.eq(sid), 'station_name'].iloc[0])
    st.caption('Tip: you may also click a gauge marker on the map to change the selection.')

    st.markdown('<div class="fr-sidebar-step">03 &nbsp; HISTORICAL PREDICTION</div>', unsafe_allow_html=True)
    fc1, fc2 = st.columns(2) if inline else (None, None)
    dates = ws.dates(station_id)
    date_options = dates or sorted(ws.predictions.forecast_date.unique().tolist())
    if st.session_state.get('c4_date') not in date_options:
        st.session_state['c4_date'] = date_options[-1]
    with (fc1 if inline else nullcontext()):
        forecast_date = st.selectbox('Historical forecast date (T)', date_options, key='c4_date')
        st.caption('Fixed prediction origin · 09:00 Sri Lanka time')

    model_options = [m for m in MODELS if m in ws.available_models] + ['PERSISTENCE']
    if st.session_state.get('c4_model') not in model_options:
        st.session_state['c4_model'] = 'M7' if 'M7' in model_options else model_options[0]
    with (fc2 if inline else nullcontext()):
        if view == 'Public Explorer':
            advanced = st.checkbox('Explore other research models (advanced)',
                                   key='c4_public_advanced', value=False)
            if not advanced:
                if st.session_state['c4_model'] != ('M7' if 'M7' in model_options else model_options[0]):
                    st.session_state['c4_model'] = 'M7' if 'M7' in model_options else model_options[0]
                model = st.session_state['c4_model']
                st.caption(f'Current model: {format_model(model)} (research demo, not an endorsed best model)')
            else:
                model = st.selectbox('Research model', model_options,
                                     key='c4_model', format_func=format_model)
        else:
            model = st.selectbox('Research model', model_options,
                                 key='c4_model', format_func=format_model)

    st.markdown('<div class="fr-sidebar-step">04 &nbsp; MAP DISPLAY</div>', unsafe_allow_html=True)
    map1, map2 = st.columns(2) if inline else (None, None)
    basemap_choices = [t[0] for t in tile_options(carto_key)]
    if st.session_state.get('c4_basemap') not in basemap_choices:
        st.session_state['c4_basemap'] = basemap_choices[0]
    with (map1 if inline else nullcontext()):
        basemap = st.selectbox('Basemap', basemap_choices, key='c4_basemap')
    with (map2 if inline else nullcontext()):
        fit_catchment = st.checkbox('Fit the entire upstream catchment',
                                   value=False, key='c4_fit_polygon')
    if not carto_key:
        st.caption('CARTO Voyager needs your local basemap key; both Esri layers remain available.')
    if not inline:
        st.divider()
    st.caption('40 mapped gauges · 39 with historical test predictions · 9 supported origin dates.')
    if ws.missing_models:
        st.warning('Missing saved models: ' + ', '.join(ws.missing_models))
    st.caption('Personal/local research demonstration only; do not publish private observation data without permission.')

selected_station = ws.stations.loc[ws.stations.station_id.eq(station_id)].iloc[0]
row = ws.record(station_id, forecast_date, model)

st.markdown(f'### {html.escape(selected_station.station_name)} · {html.escape(selected_station.basin)}')
st.caption(f'{pd.to_datetime(forecast_date).strftime("%d %B %Y")} at 09:00 SLST → next-day 09:00 stage · {format_model(model)}')

# Optional saved watershed geometry; never dynamically redelineate or assert official certification.
catchment_rel = None
if row is not None:
    catchment_rel = str(row['catchment_polygon_path'])
else:
    subset = ws.observations.loc[ws.observations.station_id.eq(station_id), 'catchment_polygon_path'].dropna()
    if len(subset):
        catchment_rel = str(subset.iloc[0])
    else:
        # Thambuththegama has no 2025 DMC labels but does have a shared candidate polygon.
        matches = list((ROOT / 'shared' / 'spatial' / 'catchments').glob(
            f'*/polygons/{station_id}_catchment_polygon_CANDIDATE.geojson'
        ))
        if len(matches) == 1:
            catchment_rel = matches[0].relative_to(ROOT).as_posix()
polygon = None
polygon_warning = None
if catchment_rel:
    relpath = ROOT / catchment_rel.replace('\\', '/')
    mtime = relpath.stat().st_mtime_ns if relpath.is_file() else -1
    polygon, polygon_warning = read_catchment_polygon(str(ROOT), catchment_rel, mtime)

def render_station_overview():
    """Unified map and selected next-day stage for either audience."""
    left, right = st.columns([1.65, 1], gap='medium')
    with left:
        st.markdown('<div class="fr-section-title">River stations & upstream catchment</div>', unsafe_allow_html=True)
        folium_map = build_station_map(
            ws.stations, station_id, ws.available_station_ids(),
            carto_key=carto_key, polygon=polygon, fit_catchment=fit_catchment,
            preferred_basemap=basemap,
        )
        event = st_folium(
            folium_map, width=None, height=550,
            returned_objects=['last_object_clicked_tooltip'],
            key=f'c4_gauge_map_{station_id}_{basemap}',
            use_container_width=True,
        )
        if event and event.get('last_object_clicked_tooltip'):
            clicked = extract_station_id(event.get('last_object_clicked_tooltip'))
            if clicked and clicked in set(ws.stations.station_id) and clicked != station_id:
                st.session_state['_pending_map_station'] = clicked
                st.rerun()
        if polygon_warning:
            st.warning(polygon_warning)
        else:
            st.markdown('<div class="fr-caption">● Teal: selected gauge &nbsp; ● Blue: gauges with held-out predictions &nbsp; ● Grey: no labelled prediction. The teal polygon is an upstream contributing area, not a flood-inundation map. Click any gauge to switch.</div>', unsafe_allow_html=True)

    with right:
        render_prediction_result(row, model, selected_station.station_name)
        st.divider()
        if row is not None:
            mrow = ws.metrics.loc[ws.metrics.model.eq(model)]
            if not mrow.empty:
                st.metric('Overall held-out test MAE', metres(mrow.iloc[0].mae_m, 4))
            st.caption('This is the model’s average error across 351 test records, not the selected prediction’s error.')
        else:
            st.caption('No new model inference is run. Only saved held-out historical predictions can be viewed.')


if view == 'Public Explorer':
    render_station_overview()
    st.markdown('<div class="fr-view-separator"></div>', unsafe_allow_html=True)
    st.subheader('Historical water levels at this gauge')
    if row is not None:
        show_stage_chart(ws, station_id, model)
    else:
        st.info('Choose one of the 39 gauges with a saved historical test prediction.')
    st.markdown('**How to read these results**')
    st.write('The solid lines show the next-day observed stage, the model estimate and a simple persistence baseline (tomorrow equals today). These are historical observations, so the actual values are already known.')
    st.caption('The selected-day error is not an accuracy percentage or a confidence interval. Model-implied alert categories use the selected station’s own reference thresholds; they are not official warnings or flood-inundation maps.')
else:
    # Visible at the top, rather than hidden below an entire tall map.
    tabs = st.tabs(['🗺️ Station overview', '📊 Compare all models', '🔎 Data & QA'])
    with tabs[0]:
        render_station_overview()
        if row is not None:
            st.markdown('#### Gauge-specific reference thresholds')
            t1, t2, t3 = st.columns(3)
            t1.metric('Alert level', metres(row.alert_threshold_m))
            t2.metric('Minor flood level', metres(row.minor_flood_threshold_m))
            t3.metric('Major flood level', metres(row.major_flood_threshold_m))
            st.caption('These are reported gauge-stage reference levels, not predicted flood depth or official warnings.')
        st.markdown('#### Historical stage comparison')
        show_stage_chart(ws, station_id, model)
    with tabs[1]:
        st.subheader('Model comparison for the selected gauge and date')
        st.caption('M1–M7 and persistence use the same historical station-day. Their predictions were saved after held-out evaluation.')
        render_officer_comparison(ws, station_id, forecast_date)
        st.markdown('#### Model performance across all 351 held-out records')
        render_officer_metrics(ws, model)
    with tabs[2]:
        st.subheader('Source quality and model assumptions')
        render_data_qa(ws, row, selected_station)
        with st.expander('Independent C4 model design'):
            st.write('M1–M7 were trained independently on the same chronological split: 809 train / 269 validation / 351 test. The three independent encoders produce 128/16/16-dimensional embeddings for intermediate concatenation fusion. Learned modality attention has not been implemented.')
            st.write('Source inputs are quality-screened SAR, daily ASCAT SWI, ERA5-Land hourly weather and static upstream-catchment features. Source availability at a simulated 09:00 origin is a retrospective assumption, not a verified live data service.')

st.markdown('<div class="fr-footer">FloodRisk Sri Lanka · C4 BSc Data Science research proof of concept · Source measurements: DMC / Sri Lanka Irrigation Department records, interpreted for retrospective research · Open hydrological and Earth-observation datasets · Not affiliated with or replacing the official warning dashboard · Personal/local demo only until data-use and deployment rights are reviewed.</div>', unsafe_allow_html=True)
