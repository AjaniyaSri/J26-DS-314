# PP1 Recovery Plan — 6–19 October 2026

**Implementation target:** end of 18 October 2026  
**Contingency and rehearsal:** 19 October 2026  
**Progress Presentation 1:** 20 October 2026

## Goal

Deliver four independently runnable component proof-of-concepts on two quality-screened candidate events. Each component should demonstrate source data → preprocessing → model/training → prediction → test/evaluation → small working UI. The full 8–10-event benchmark, full cross-validation and extensive statistical tests remain later work.

## Team ownership

- **C1 Satellite:** provisional AOI map, DS/district/province overlay, reference gauge map, actual Sentinel-1 acquisition inventory and candidate pre/post pairs; then its own satellite data pipeline/model/UI.
- **C2 Weather:** complete EV2 DMC event reconstruction (Tables A/B/C), then its own CHIRPS/ERA5 component.
- **C3 Environment:** complete EV9 DMC event reconstruction (Tables A/B/C), then its own environmental component.
- **C4 coordinator (you):** common scope/contracts, supervise AOI/date decisions, prototype and coordinate shared SAR masks, review shared outputs, build independent fusion component and CED/log evidence.

Event owners must provide filled draft evidence tables, not just report links. C2/C3 may ask for help interpreting a PDF but remain responsible for a first complete pass of their assigned event.

## Day-by-day plan

### 6 October — establish shared structure and parallel work
- C1: run the AOI and Sentinel-1 metadata inventory; deliver the scene CSV, AOI/gauge map and preliminary administrative overlay.
- C2: reconstruct EV2 chronology beginning 25 Nov–1 Dec 2025; fill Table A and Table C; expand the date search in two-day blocks as warranted; prepare a draft Table B.
- C3: reconstruct EV9 chronology beginning 11–17 May 2021; fill Table A and Table C; expand the date search in two-day blocks as warranted; prepare a draft Table B.
- C4: commit the initial event/AOI contract and templates; start a small mask-processing test only after discovering actual candidate scenes; request supervisor review.
- End-of-day result: all members deliver a real artifact (CSV, map, source-backed finding, code or log), not merely a list of URLs.

### 7 October — cross-check and supervisor review 1
- C1 checks whether the candidate AOI covers useful observations for both events and reports actual acquisition dates and candidate pairs.
- C2/C3 expand timelines backward toward earliest local impact/commencement and forward toward recession/withdrawal. Record gaps explicitly.
- C4 reviews both envelope drafts and creates a list of date/AOI/label contradictions or gaps.
- Meet the supervisor if available. Ask about the two-event PP1 scope, whether one common AOI is justified, the temporal label protocol and appropriate claims if SAR acquisition date differs from the target date.
- End-of-day result: recommendation to keep/revise AOI and retain/replace each event; no invented date certainty.

### 8 October — freeze the contract or log unresolved gates
- Finalise the event/AOI selection recommendation from evidence.
- Approve a shared box only if both event records and data coverage support it. Otherwise use event-specific boxes using the same grid/label rules.
- Freeze the CRS/grid/NoData/mask conventions only after resolving the 30 m versus EPSG:4326 issue.
- Decide the event-level PP1 split and target-date selection rule. Specific dates may remain provisional if required evidence is unavailable; record the remaining gate and fallback.
- C2/C3 deliver draft Tables A, B and C with sources and uncertainties.

### 9 October — source retrieval
- All components retrieve the data they need for the approved candidate windows.
- Keep large rasters and intermediate artifacts on local storage; save source IDs, dates, manifests and checksums in the repository.
- Record export failures as Planner blockers rather than silently changing study dates.

### 10–11 October — preprocessing and mask candidates
- Each component produces model-ready inputs for both events or documents the exact blocker.
- C4 generates candidate masks using the selected actual scene IDs and the agreed procedure.
- C1/C3 independently check mask coverage, alignment and visual plausibility.

### 12–13 October — mask QA and common artifact release
- Complete mask checks (CRS, transform, dimensions, values, NoData, flood/non-flood balance and overlay review).
- Publish approved shared masks and metadata; retain rejected candidates with clear names/status only if needed as research evidence.
- Freeze the common label, split and metrics schema in GitHub.

### 14–15 October — model and evaluation milestone
- Every component completes at least one actual training/inference path and saves predictions.
- Produce per-event evaluation where the labels and split support it.
- For C4, attempt short runs for M1–M7; report exactly which configurations trained and which failed/skipped. Do not claim full ablation evaluation for unrun models.
- Supervisor review 2 on 15 October: show outputs, metrics, UI progress and the remaining critical issues.

### 16–17 October — UI and evidence pack
- Complete a minimal component UI displaying event, input/target date information, output, model/version and relevant metrics/status.
- Test reproducibility, missing-input handling, output file saving and UI load.
- Prepare system diagram, event-selection evidence, mask QA evidence, metrics, screenshots, risk/limitation statement and logbook/CED references.

### 18 October — freeze the implementation
- Run end-to-end smoke tests on the chosen PP1 demonstration path.
- Confirm the repository has scripts/configs, shared artifacts, component outputs and evidence.
- Freeze new scope; record incomplete items honestly and list as post-PP1 work.

### 19 October — rehearsal and contingency
- Rehearse the presentation and the live/demo recording.
- Fix only blocking bugs and data-path issues.
- Keep a reproducible backup of the approved demo inputs/outputs and screenshots.

## Daily progress format

Each member updates Planner and their own logbook with:

- date, component and Planner task ID;
- time spent;
- work completed;
- evidence/file/commit link;
- pass/partial/fail result;
- blocker and next action;
- decision made and its evidence, if any.

## Scope controls

Defer: nationwide event reconstruction, all 8–10 event processing, full five-fold cross-validation, extensive hyperparameter search, production deployment and UI polish. Prioritise a working, reproducible proof-of-concept path per component and a defensible explanation of what was and was not established.
