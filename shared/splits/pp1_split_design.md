# PP1 Event-Level Split — Draft

**Status:** Proposed, awaiting data feasibility review and supervisor confirmation.

## Proposed proof-of-concept split

- One event is assigned as the development/training event.
- The other event is held out for a demonstration prediction.
- The held-out event must not be used for model tuning, preprocessing-statistic fitting or threshold selection.
- A final selection of which event is development vs held-out is made only after the AOI, input scenes and mask feasibility are reviewed.

## Limitation

Two events cannot provide independent train, validation and test event groups simultaneously. This is a small proof-of-concept design, not the final 8–10-event validation strategy. Do not report strong generalisation claims from the two-event demonstration.
