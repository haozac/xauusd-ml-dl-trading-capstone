# Stage 2 Step 2A v1.1 Internal Review

## Change summary

v1.1 fixes timestamp handling for Dukascopy MT5 server time. The v1.0 watch run proved live polling and logging, but showed a negative latest-bar age because MT5 server timestamps were treated as UTC. v1.1 adds explicit broker-server-time normalisation before all feature generation and signal logging.

## Design decision

Dukascopy documents MT4/MT5 server time as GMT+3 in summer and GMT+2 in winter. The capstone research dataset and frozen feature contracts are UTC-aligned. Therefore, live MT5 server timestamps must be shifted back by the configured offset before they enter the feature builder.

For July 2026, the runtime config uses:

```yaml
time:
  mt5_server_time_offset_hours: 3
```

## Files changed

- `config/mt5_shadow_runtime_template.yaml`
- `src/capstone_trading/runtime/mt5_shadow.py`
- `scripts/deployment/run_mt5_shadow_logger.py`
- `tests/unit/test_mt5_shadow.py`
- `docs/deployment/stage2_step2a_runbook.md`
- `docs/deployment/stage2_step2a_review.md`
- `docs/deployment/stage2_step2a_patch_manifest.json`

## Safety review

The patch remains read-only with respect to MT5. It does not expose or call `order_send`, `order_check`, or other trading functions. It continues to use completed bars only and leaves `orders_enabled=false`.

## Timestamp review

The new `time_normalisation` report records raw server-time values, canonical UTC values, configured offset, bar age before conversion, bar age after conversion, tick conversion, and future-bar guard status.

The feature pipeline receives only canonical UTC bars after conversion.

## Deprecation cleanup

The TensorFlow warning from using `.cpu()` on an EagerTensor is cleaned by only using `.cpu()` after a PyTorch-style `.detach()` path. TensorFlow tensors are converted directly through `.numpy()`.

## Known limitation

This patch uses a configured fixed offset. For Dukascopy, use `3` during summer-time periods and `2` during winter-time periods. If this project is run across a daylight-saving changeover, the offset should be reviewed before formal evidence collection.

## Local static review

The files were compiled with Python syntax checks. Unit tests are provided for the time-normalisation logic, but full project tests must be run in the user's repository because the complete capstone package and MT5 environment are local to the user.
