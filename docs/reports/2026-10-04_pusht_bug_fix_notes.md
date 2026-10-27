# PushT bug fix notes

Recorded October 4, 2026, America/Los_Angeles.

## Corrected 50 demo runs
The corrected 50-demo comparison consists of a retrained BC, ST k1 uniform, and ST k16 uniform / flat600 / ramp200→600. Results and checkpoint sweeps are in the W&B report. Older random BLQ50 evaluations are excluded from the current report because they use the legacy evaluation protocol.

- **EMA noise calibration:** noise-scale statistics and initialization flags were not copied into EMA. EMA now copies non-parameter buffers; older affected evaluation checkpoints are repaired in memory.
- **Stale/mismatched ST context:** older training reused generated context across four inner passes, with collection/training crop and dropout differences. `fresh_context=true, inner_epochs=1` generates context inside each loss with the current model, shared crop/features/corruption, and training mode. This matters for k16; k1 has no prior-candidate context.
- **Evaluation RNG:** candidate draws now use `paired_decision_v3`, independent of N and reset per decision; noisy observations use per-episode RNG streams. This removes N-dependent sampling seeds and batch-coupled corruption.
- **Interrupted training:** fresh ST checkpoints restore Python/NumPy/Torch RNG and minibatch position; update RNG is keyed by global step.
- **Terminal scoring:** terminal-success handling is explicitly enabled for the ST training verifier and corrected evaluations. This is a protocol correction alongside the implementation fixes.

These changes were bundled; the reruns do not isolate the effect of each fix. Dataset audit confirms 50 train / 19 validation / 50 test, with the same validation/test episodes as BLQ137. Provenance: `pusht_far50_20261003/retrain_manifest.json` and `pusht_ema_audit_20261003/audit_findings.json` (copies saved with report data).

