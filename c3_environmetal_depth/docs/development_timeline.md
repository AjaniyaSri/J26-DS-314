# C3 Environmental Depth ? Evidence-based Development Timeline

Prepared on **21 October 2026** for supervisor review.

This is a retrospective reconstruction from saved local evidence, not a contemporaneous daily work diary. Dates below describe what the records support. Experiment execution, artifact preparation, Git commit creation and GitHub publication are separate events.

## Summary for the supervisor

> The saved C3 records document local Model 1 and Model 2 experiment execution on 19 October 2026. Prediction analysis and consolidated feature-importance artifacts were present in the dated final-report manifest by 20 October. The existing implementation and research evidence were subsequently organized into eight Git commits on 20?21 October. The short interval between those commits reflects repository organization; it does not establish the duration of development.

The inspected records do **not** establish specific C3 development milestones from **9?18 October**. This is an evidence gap, not proof that no work occurred. Earlier work should be added only when dated supporting records are available.

## How dates were interpreted

- All displayed times use **Asia/Colombo (+0530)**. Explicit UTC timestamps in saved metadata were converted by adding 5 hours 30 minutes.
- A model's `frozen_utc` records the saved freeze event; a completion marker records an evaluation completion event. Neither establishes when its source code was first written.
- Logs without timezone offsets corroborate the dated runs, but explicit UTC metadata supplies the time basis.
- The final report's `created_utc` and recorded source hashes provide a documented ?present by? boundary for earlier analysis artifacts. They do not establish each analysis's exact original creation time.
- File modification times were not used to reconstruct work dates because this project was copied between locations.
- These are locally recorded timestamps and hashes, not independently certified timestamps or proof of individual authorship.

## Recorded research and development events

| Date/time (+0530) | Event supported by the records | Evidence | Interpretation |
|---|---|---|---|
| 9?18 October 2026 | No specific C3 work milestone established in the inspected records | C3 artifact dates and Git history reviewed during this reconstruction | Unverified period; no daily activities or work hours assigned |
| 19 October, 13:07:39 | Original absolute-stage Model 1 frozen | [Original metadata](../models/model1_static/model1_metadata.json), `frozen_utc=20261019T073739_480886Z` | A recorded model-freezing event, not the start of implementation |
| 19 October, 13:07:43 | Original Model 1 TEST evaluation completion recorded | [Completion marker](../models/model1_static/test_evaluation_complete.json), `completed_utc=20261019T073743_499866Z` | Existing saved assessment; no new evaluation was run for this timeline |
| 19 October, 13:36:19 | Revised Model 1 selection frozen | [Revised metadata](../experiments/model1_diagnostics_v2/models/frozen_experiment.json), `frozen_utc=20261019T080619_946557Z` | Selected `delta_small` stage-change model |
| 19 October, 13:36:42 | Revised Model 1 descriptive TEST assessment completion recorded | [Revised completion marker](../experiments/model1_diagnostics_v2/models/test_evaluation_complete.json), `completed_utc=20261019T080642_090018Z` | The project TEST period was already examined |
| 19 October, 17:21:46 | Completed Model 2 run's selection frozen | [Model 2 metadata](../experiments/model2_dynamic_v1/runs/20261019T115140_557496Z/models/model2_metadata.json), `frozen_utc=20261019T115146_416096Z` | Primary absolute and secondary change experiments; selected secondary model `model2_delta_mae_loss` |
| 19 October, 17:22:47 | Model 2 descriptive TEST assessment completion recorded | [Model 2 completion marker](../experiments/model2_dynamic_v1/runs/20261019T115140_557496Z/models/test_evaluation_complete.json), `completed_utc=20261019T115247_958122Z` | Saved comparison includes persistence and matched no-weather controls |
| By 20 October, 11:35:04 | Prediction-analysis report and consolidated feature-importance report were present as sources in the final-report manifest | [Dated final-report manifest](../reports/final_c3_research_report_v1/source_artifact_checksums.json), `created_utc=2026-10-20T06:05:04.136484+00:00` | Both reports' current SHA-256 hashes were checked against its `pre_existing_artifact_sha256` entries during this reconstruction |
| 20 October, 11:35:04 | Final research-report deliverables recorded by their checksum manifest | [Final report](../reports/final_c3_research_report_v1/final_c3_research_report.md) and [manifest](../reports/final_c3_research_report_v1/source_artifact_checksums.json) | Recorded manifest creation time; not a claim about the exact start/end of report writing |
| By 21 October, 08:56:33 | Historical inference script, guide and tests preserved in Git | Commit `edb4bfc`; [demo guide](pp1_inference_demo.md) | This establishes their presence at commit time; their first coding date is not assigned from that commit |

Supporting interpretation: [revised Model 1 comparison](../experiments/model1_diagnostics_v2/reports/model1_final_comparison.md), [prediction analysis](../experiments/model2_dynamic_v1/runs/20261019T115140_557496Z/prediction_analysis/prediction_analysis_report.md), and [feature-importance report](../experiments/c3_feature_importance_v1/feature_importance_report.md).

Some linked model metadata, logs and run artifacts are intentionally ignored by Git and remain local. Their links require the complete local project and may not resolve on GitHub. No datasets or model weights are published by this document.

## Git commit creation timeline

These are the actual recorded commit events. The table does not assign each milestone's entire development effort to its commit date. All eight commits have matching author and committer instants.

| Commit | Creation time in Sri Lanka (+0530) | Milestone |
|---|---|---|
| `395fc97` | 20 October 2026, 23:48:06 | Environment and local exclusions |
| `b208cf3` | 20 October 2026, 23:57:13 | Shared data contracts, configuration and validation |
| `73b5bac` | 21 October 2026, 08:49:53 | Static Model 1 and revised diagnostics |
| `c228d12` | 21 October 2026, 08:56:00 | Model 2 and matched comparisons |
| `edb4bfc` | 21 October 2026, 08:56:33 | Offline historical inference demo |
| `2a3e6d1` | 21 October 2026, 08:57:42 | Experiment audits, findings and research figures |
| `790831d` | 21 October 2026, 08:58:10 | Consolidated feature-importance analysis |
| `81f36e1` | 21 October 2026, 08:58:46 | Final C3 research synthesis |

Commit `395fc97` stores `2026-10-20T18:18:06Z`; the table converts that same instant to Sri Lanka time. Its existing UTC offset was explicitly accepted and was not rewritten. The other commits store +0530 offsets (Git's ISO display uses +05:30).

The exact GitHub push/publication time was not independently checked. Git commit dates do not establish when a push occurred. The local branch reviewed for this timeline is `feature/c3-environmental-depth`, at `81f36e1946e3cbdad1f7235bbaf34b5eb91b9dfb`.

## Research meaning and limits

The work predicts next-day gauge stage, not flood-susceptibility or inundation maps. The selected stage-change Model 2 slightly improves pooled error relative to persistence, but its matched no-weather control has slightly lower overall MAE. The timeline does not change those results or establish a weather benefit, causal effect, seasonal generalization or operational readiness.

No model was retrained, dataset altered, old report overwritten or Git history rewritten to prepare this timeline. This document is a new review artifact; it has not been staged, committed or pushed by this step.

## What is needed to document 9?18 October

Dated research notes, notebook revision history, earlier commits, exported analysis logs, source-control backups or supervisor meeting records could support additional entries. Each addition should state the specific activity, its date, the supporting record and any uncertainty. A remembered date can be labelled a retrospective recollection, but should not be presented as independently verified activity.

Until such records are supplied, the appropriate account is ?earlier development dates not verified from the available C3 evidence,? rather than distributing recent commits across earlier dates.
