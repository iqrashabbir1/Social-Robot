# Digital Twin Predictive Validation

This note captures the manuscript-facing updates required after upgrading the digital twin from a passive state buffer to a predictive and safety-auditable subsystem.

## Section 3.3 update for Equation (4)

The synchronization-latency equation should remain

`epsilon_dt = |t_now - max(t_visual, t_audio, t_physio)|`

but the narrative should now include the empirical validation result obtained from the replay-oriented digital-twin benchmark:

- mean synchronization latency: `124.0 ms`
- standard deviation: `67.0 ms`
- `98.7%` of updates within the `500 ms` design threshold
- `p99 = 501.5 ms`
- `p98.7 = 499.9 ms`

For reviewer-facing clarity, the manuscript should avoid pairing `p99 = 487 ms` with `98.7%` within `500 ms`, because those two statements are not jointly consistent for the same distribution. The cleaner wording is:

> The empirical synchronization error was `124.0 +/- 67.0 ms`, with `98.7%` of updates remaining within the `500 ms` design threshold. The observed `p99` latency was `501.5 ms`, while the `98.7th` percentile remained below the threshold at `499.9 ms`.

## Table 2 update for M7

The `M7` row should now reflect the measured synchronization and audit capability rather than only a nominal threshold statement.

Recommended wording:

| Module | Function | Empirical status |
|---|---|---|
| `M7` | Digital twin synchronization and audit replay | Synchronization latency `124.0 +/- 67.0 ms`; `98.7%` of updates within `500 ms`; append-only safety audit and replay enabled |

## Section 4.10 manuscript insert

The following subsection is ready to insert after the missing-modality analysis.

### 4.10 Digital Twin Predictive Validation

The digital twin was further evaluated as a predictive and safety-auditable subsystem rather than as a passive synchronization buffer only. A recurrent predictor was trained over simulation-derived twin histories using a two-layer LSTM with `384`-dimensional input state vectors. The objective of this experiment was to test whether short-horizon forecasting and replay-grounded audit reconstruction could be added without violating the existing synchronization constraints of the framework.

Under a `10 s` prediction horizon, the learned predictor achieved mean squared error `0.0029` on held-out simulated twin trajectories. This indicates that the twin state remained locally predictable over short horizons in the current replay-oriented setting. The anomaly detector, which compared predicted and observed future state trajectories, achieved precision `1.000` and recall `1.000` on the current synthetic perturbation benchmark. These values should be interpreted as technical validation under simulation-grounded conditions rather than as evidence of prospective clinical anomaly-detection performance.

Synchronization behavior remained compatible with the original design target. The measured digital-twin synchronization error was `124.0 +/- 67.0 ms`, with `98.7%` of updates remaining within the `500 ms` threshold. The observed `p99` latency was `501.5 ms`, indicating a narrow tail beyond the nominal limit, whereas the `98.7th` percentile remained below threshold at `499.9 ms`. This pattern is consistent with a system that is usually compliant with the design target but still exhibits occasional tail-latency excursions that should be tracked during future real-time deployment studies.

The safety-audit layer complemented predictive validation by providing append-only incident logging with chained SHA256 signatures and pre-incident state replay. In the current benchmark, a fall-detection incident was reconstructed with a `5 s` pre-incident replay window, demonstrating that the digital twin can support post hoc review of perception state, synchronization status, and contextual signals immediately before a safety-critical event. This capability is particularly relevant for human oversight, failure analysis, and later compliance-oriented audit workflows.

Overall, the digital twin should now be interpreted as a hybrid synchronization, forecasting, and audit subsystem. The current evidence supports technical feasibility for replay-grounded predictive monitoring, but not yet prospective validation in real caregiving deployments.
