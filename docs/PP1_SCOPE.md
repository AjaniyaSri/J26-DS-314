# PP1 Scope and Shared Event Contract (v0.1)

**Status:** Draft, 6 October 2026. Not yet supervisor-approved.

## Purpose

Provide one shared operational contract for the PP1 proof of concept so four independently developed components can start in parallel. PP1 demonstrates end-to-end feasibility on two candidate historical events. It is not the final 8–10-event benchmark and must not be presented as proof of nationwide generalisation.

## Candidate events

| Event ID | Proposal reference date | Initial DMC seed window | Broad Sentinel discovery window | Status |
|---|---|---|---|---|
| `EV2_Nov_2025` | 2025-11-28 | 2025-11-25 to 2025-12-01 | 2025-11-10 to 2025-12-05 inclusive | CANDIDATE |
| `EV9_May_2021` | 2021-05-14 | 2021-05-11 to 2021-05-17 | 2021-04-25 to 2021-05-20 inclusive | CANDIDATE |

The proposal reference dates are preliminary inventory values, not verified hydrological or SAR peak dates. DMC search windows are seed windows that must be expanded backward and forward if the relevant reports indicate an earlier onset or later recession. Sentinel discovery windows are for acquiring candidate scene metadata only.

## Provisional AOI

AOI ID: `PP1_AOI_KELANI_01`

- West longitude: 79.82° E
- South latitude: 6.80° N
- East longitude: 80.28° E
- North latitude: 7.25° N
- Coordinate reference: WGS 84 longitude/latitude (`EPSG:4326`)
- Approximate geographic size: about 51 km east–west by 50 km north–south; final dimensions must be calculated in a suitable projected CRS for reporting.

GeoJSON path: `shared/events/pp1_aoi_candidate.geojson`.

The rectangle is a candidate study window, not a claimed flood extent. C1 must map administrative divisions intersecting the geometry, identify gauge locations within or hydrologically relevant to it, and inventory actual Sentinel-1 coverage. C2/C3 must establish whether DMC-reported impacts for their events are relevant to this window. If the window is unsuitable for either event, propose a revised or event-specific AOI and record the decision before final exports/masks.

## Candidate target dates and temporal rule

Do not set final target dates solely from the preliminary reference dates. For each event:

1. Reconstruct the local documentary timeline from DMC Situation Reports and relevant river-level/flood-warning reports.
2. Extract actual water levels for selected relevant gauge(s), preserving timestamps, units and thresholds as stated in the source.
3. C1 inventories actual Sentinel-1 acquisitions and proposes compatible pre/post pairs.
4. The group and supervisor decide which flood condition is the intended target and whether the SAR label acquisition can represent it credibly.
5. Record separate fields for event onset, hydrological peak, impact peak (if comparably documented), warning peak, event recession/end, target date `T`, input cutoff `T-1`, and SAR label-acquisition date.
6. If the SAR label date differs materially from `T`, document the lag and do not claim strict one-day-ahead prediction of that observed label without a justified protocol.

## Shared contracts to freeze

- Event IDs and selected AOI IDs.
- Shared flood-reference mask and label encoding.
- CRS, pixel size, grid alignment and NoData semantics.
- Event-level PP1 development/held-out demonstration split.
- Prediction cutoff and target/label-date convention.
- Metrics schema and result column definitions.
- Data-source IDs, acquisition metadata and file manifests.

**CRS warning:** a 30 m grid cannot be implemented as 30-degree pixels in EPSG:4326. Before producing canonical rasters, the group must choose either a suitable projected CRS with units in metres, or an explicitly calculated angular grid whose ground spacing is about 30 m at the study latitude. Do not silently mix these conventions.

## PP1 validation boundary

For a two-event demonstration, the proposed starting split is one event used for model development and the other held out for demonstration. Choose the development event only after source/label feasibility checks. Do not use held-out-event outcomes to tune the model. This is a proof-of-concept demonstration, not a stable estimate of out-of-event generalisation.

## Acceptance conditions before final data exports

- Local DMC impacts and relevant gauges checked against the AOI.
- Candidate Sentinel-1 pre/post scenes inventoried and their coverage assessed.
- Feasibility of weather and environmental sources checked for both event windows.
- A defensible temporal-label relationship recorded or the candidate event/AOI replaced.
- Shared grid/label/NoData conventions approved.
