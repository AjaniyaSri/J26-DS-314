# Kelani catchment candidates — FloodRisk Sri Lanka (J26-DS-314)

This folder contains **candidate**, not formally validated, MERIT Hydro v1.0.1
upstream catchments for seven Kelani gauge locations sourced from the public
Irrigation Department ArcGIS dashboard. HydroBASINS Level 12 is a geographic
cross-check rather than the primary delineation source.

- `polygons/`: irregular upstream watershed GeoJSON polygons (WGS84).
- `bounding_boxes/`: rectangular WGS84 bounding extents, **not** the watershed.
- `masks/`: native MERIT-grid GeoTIFF watershed masks, 1 inside and 0 outside.
  When resampling to another source grid, use a consistent categorical-mask
  procedure and preserve source raster NoData separately.
- `kelani_7_catchment_QA_registry.csv`: seven-station area/outlet/status audit.
- `metadata/`: outlet candidates, HydroBASINS and other supporting checks.
- `kelani_7_catchment_file_manifest.csv`: sizes and SHA256 hashes.

**Known limitations:** Kithulgala has about 11.38% difference from the published
drainage area; Nagalagam Street has about 9.01% and requires attention to
lowland/coastal river-network conditions. Internal traced-area agreement with
MERIT `upa` does NOT provide independent spatial validation. None of the seven
geometries is labelled VERIFIED.

**To share with the group:** Extract the team ZIP from `results/catchments/`
into the repository root. GeoTIFFs are intentionally ignored in Git, and must
be transferred separately. Do NOT upload the native regional MERIT raster,
intermediate tiles, or HTML maps with possible basemap/Earth Engine tokens.

**C1–C4:** All components may agree on these raw reference geometries and
station definitions. C4 independently builds and trains all three modality
pipelines, without C1/C2/C3 embeddings, features, models or predictions.
