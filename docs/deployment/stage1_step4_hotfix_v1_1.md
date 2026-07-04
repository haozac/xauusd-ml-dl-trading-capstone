# Stage 1 Step 4 hotfix v1.1

This hotfix corrects the Model A replay semantics after the first Step 4 local run exposed a real trading-path mismatch.

Corrections:
1. A non-contiguous/gap prediction row is always non-tradable. If a position is open it performs a gap exit; if already flat it remains flat even when the raw signal is long or short.
2. `min_hold_bars = 3` is interpreted as three completed bars after entry before another position change is allowed. With the internal hold counter starting at 1 on the entry row, the engine blocks changes while `hold_bars <= 3`.
3. After a normal policy/risk exit to flat, the flat state must also complete the three-bar hold/cooldown before a new policy entry. Forced gap exits are exempt.

These are parity corrections only. They do not change Notebook 7, Model A, Model B, thresholds, costs, artefacts, or Stage 0.
