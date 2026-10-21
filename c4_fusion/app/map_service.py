"""Folium map rendering for private C4 gauge points and candidate upstream watersheds."""
from __future__ import annotations

from html import escape
from pathlib import Path
import json
import re

import folium
from folium.plugins import Fullscreen

ESRI_SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_TOPO = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}"
CARTO_BASE = "https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
ATTR_ESRI_SAT = "Tiles &copy; Esri, Maxar, Earthstar Geographics, and GIS User Community"
ATTR_ESRI_TOPO = "Tiles &copy; Esri and its data providers"
ATTR_CARTO = ('&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
              'contributors, &copy; <a href="https://carto.com/attribution/">CARTO</a>')


def tile_options(carto_key: str | None):
    key = (carto_key or '').strip()
    # CARTO key is passed to the browser in the tile URL; don't confuse local .env privacy with
    # secrecy of a browser-visible query-string credential.
    options = [("Esri Topographic", ESRI_TOPO, ATTR_ESRI_TOPO, 19)]
    if key:
        options.insert(0, ("CARTO Voyager", CARTO_BASE + "?key=" + key, ATTR_CARTO, 20))
    options.append(("Esri Satellite", ESRI_SATELLITE, ATTR_ESRI_SAT, 19))
    return options


def _polygon_bounds(geometry: dict) -> tuple[float, float, float, float] | None:
    coords = []

    def walk(value):
        if isinstance(value, (list, tuple)) and len(value) >= 2 and all(isinstance(n, (int, float)) for n in value[:2]):
            coords.append((float(value[0]), float(value[1])))
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(geometry.get("coordinates", []))
    if not coords:
        return None
    return min(p[0] for p in coords), min(p[1] for p in coords), max(p[0] for p in coords), max(p[1] for p in coords)


def load_candidate_polygon(root: Path, relative_path: str | None) -> tuple[dict | None, str | None]:
    """Use exactly the established station polygon; no geoprocessing or regeneration."""
    if not relative_path or str(relative_path).lower() in ('nan', 'none'):
        return None, 'No catchment path recorded for the selected station.'
    relative = str(relative_path).replace('\\', '/')
    if not relative.startswith('shared/spatial/catchments/') or not relative.endswith('_catchment_polygon_CANDIDATE.geojson'):
        return None, 'Unexpected candidate polygon path; overlay omitted for safety.'
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to((Path(root) / 'shared' / 'spatial' / 'catchments').resolve()):
        return None, 'Candidate polygon path resolves outside the shared catchment directory.'
    if not path.is_file():
        return None, f'Missing local catchment polygon: {relative}'
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except (UnicodeError, json.JSONDecodeError):
        return None, f'Invalid GeoJSON candidate polygon: {relative}'
    features = data.get('features', []) if data.get('type') == 'FeatureCollection' else [data]
    if len(features) != 1 or features[0].get('geometry', {}).get('type') not in {'Polygon','MultiPolygon'}:
        return None, f'Candidate file is not a single polygon: {relative}'
    return {"type": "FeatureCollection", "features": features}, None


def extract_station_id(clicked_tooltip: object) -> str | None:
    """Safe click-decoder: never use arbitrary lat/lon to guess a station."""
    if not isinstance(clicked_tooltip, str):
        return None
    match = re.search(r'\[gauge:([a-z0-9_]+)\]', clicked_tooltip)
    return match.group(1) if match else None


def build_station_map(stations, selected_station_id: str, available_station_ids: set[str],
                      *, carto_key: str = '', polygon: dict | None = None, fit_catchment: bool = False,
                      preferred_basemap: str = 'CARTO Voyager') -> folium.Map:
    """All 40 stations visible; selected station highlighted with its upstream polygon."""
    selected = stations.loc[stations.station_id.eq(selected_station_id)]
    if len(selected) != 1:
        raise ValueError(f'Unknown station_id {selected_station_id}')
    row = selected.iloc[0]
    m = folium.Map(location=(float(row.lat), float(row.lon)), zoom_start=11,
                   tiles=None, control_scale=True, prefer_canvas=True)
    layers = tile_options(carto_key)
    available_names = {name for name, _, _, _ in layers}
    if preferred_basemap not in available_names:
        preferred_basemap = layers[0][0]
    for name, url, attribution, max_zoom in layers:
        folium.TileLayer(tiles=url, name=name, attr=attribution,
                         max_zoom=max_zoom, overlay=False, control=True,
                         show=name == preferred_basemap).add_to(m)

    if polygon:
        folium.GeoJson(
            polygon,
            name='Selected upstream catchment (candidate)',
            style_function=lambda feature: {
                'fillColor': '#138577', 'color': '#08786D', 'weight': 2.7,
                'fillOpacity': 0.12, 'opacity': 0.90,
            },
            highlight_function=lambda feature: {'fillOpacity': 0.2, 'weight': 3.4},
            tooltip=folium.Tooltip('Upstream contributing area (candidate, NOT flood inundation)'),
        ).add_to(m)
        if fit_catchment:
            bounds = _polygon_bounds(polygon['features'][0]['geometry'])
            if bounds:
                lonmin, latmin, lonmax, latmax = bounds
                m.fit_bounds([[latmin, lonmin], [latmax, lonmax]], padding=(26,26))

    all_points = folium.FeatureGroup(name='Hydrometric stations', show=True)
    # Selected last so its marker stays visually above the others.
    ordered = stations.loc[~stations.station_id.eq(selected_station_id)]
    ordered = __import__('pandas').concat([ordered, selected], ignore_index=True)
    for item in ordered.itertuples():
        active = item.station_id == selected_station_id
        covered = item.station_id in available_station_ids
        color = '#087F76' if active else ('#1683A8' if covered else '#98A6B5')
        tooltip = f'Select {escape(item.station_name)} [gauge:{item.station_id}]'
        popup_html = (
            f'<div style="min-width:170px;font-family:Arial,sans-serif">'
            f'<b>{escape(item.station_name)}</b><br>'
            f'{escape(item.basin)}<br>'
            f'<small>{"Historical predictions available" if covered else "No held-out predictions"}</small>'
            f'</div>'
        )
        folium.CircleMarker(
            location=(float(item.lat), float(item.lon)),
            radius=9 if active else 5.5,
            color='#FFFFFF' if active else color,
            weight=2.2 if active else 1.2,
            fill=True, fill_color=color, fill_opacity=1 if active else 0.88,
            tooltip=tooltip,
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(all_points)
    all_points.add_to(m)
    Fullscreen(position='topleft', title='Expand map', title_cancel='Exit full screen').add_to(m)
    folium.LayerControl(collapsed=True, position='topright').add_to(m)
    return m
