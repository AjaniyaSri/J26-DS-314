# PP1 Shared Flood Label / Mask Definition — Draft

**Status:** DRAFT, not an approved mask protocol.

## Intended purpose

Provide a common binary flood-reference label for all four component models over an approved event/AOI and agreed grid. The label is a retrospective remote-sensing reference, corroborated by event/gauge records where available; DMC administrative impact reports are not pixel-level ground truth.

## Proposed workflow

1. Select a compatible pre-event and post-event Sentinel-1 GRD scene after reviewing the actual acquisition dates and local DMC chronology.
2. Inspect both scenes, coverage and valid-data intersection inside the approved AOI.
3. Calculate a documented bitemporal change measure (the proposal currently mentions log-ratio; a VV difference may be used only for a labelled exploratory prototype until the final method is confirmed).
4. Apply a documented thresholding rule such as Otsu only if the histogram and visual evidence support its use.
5. Apply limited morphological cleaning and document the parameters.
6. Remove persistent water using the agreed JRC Global Surface Water product/rule, recording the product version and threshold.
7. Perform independent QC against imagery and the event/gauge evidence.
8. Record the mask version, source scene IDs, acquisition dates, CRS/transform, value encoding, NoData value, QC status and SHA-256 checksum.

## Required label semantics

- `1`: pixel classified as flood candidate by the approved procedure.
- `0`: valid observed pixel classified as non-flood.
- NoData: pixel lacks valid observations or lies outside the valid analysis footprint; it must not be silently treated as non-flood.

The final encoding must be frozen before all components use the authoritative mask. Pilot masks are named `PILOT` or `DRAFT` and must never be consumed as approved labels without review.

## CRS/grid issue to resolve

The proposal mentions a common 30 m grid and EPSG:4326. EPSG:4326 coordinates are angular degrees, so 30 m must not be written as a 30-degree pixel size. The group must approve a projected CRS in metres or explicitly calculate an appropriate angular grid, then align every component's rasters to it.
