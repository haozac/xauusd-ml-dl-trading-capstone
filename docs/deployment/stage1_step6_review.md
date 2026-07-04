# Stage 1 Step 6 v1.1 Internal Review

## Review summary

The v1.1 patch addresses the concern that Step 6 v1.0 could be described too strongly as a full bar-by-bar live runtime emulator.

The revised gate is now more precise:

- It remains an offline-only simulation.
- It does not use MT5 and does not create orders.
- It rebuilds features using the audited historical feature pipeline.
- It verifies streaming event readiness with a rolling 48-row chronological buffer.
- It then feeds the resulting events into stateful Model A and Model B ledgers.

## Strengthened check

New functions added to `offline_simulation.py`:

- `streaming_endpoint_positions_from_features`
- `verify_streaming_event_materialisation`
- `StreamingEventMaterialisationReport`

These verify that a live-like rolling sequence-readiness buffer emits exactly the same event timestamps, endpoint rows, target directions and forward returns as the audited batch sequence plan.

## Council judgement

Approved for Stage 1 Step 6 because the remaining batch component is the already audited feature builder. The new streaming event materialisation layer is sufficient to validate the runtime event interface before MT5 shadow mode.

## Important limitation

This does not prove broker execution correctness. Broker connectivity, completed-bar polling, symbol metadata, quote freshness, position reconciliation and order-disabled shadow logging belong to Stage 2.
