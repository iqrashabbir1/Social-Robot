# Table 5. Ablation study

| Config | Removed | Val Acc | KG Faith. | HITL Prec. | Finding |
| --- | --- | --- | --- | --- | --- |
| ABL0 | None | 97.81 | 0.89 | 0.94 | Baseline |
| ABL1 | KG grounding | 97.78 | 0.27 | 0.91 | Faithfulness collapses |
| ABL2 | Speech | 90.12 | 0.89 | 0.91 | 7.7pp drop |
| ABL3 | Digital twin | 97.8 | 0.89 | 0.87 | Routing degrades |
| ABL4 | Cross-attention | 94.41 | 0.89 | 0.89 | 3.4pp drop |
| ABL5 | HITL gate | 97.78 | 0.89 | UNSAFE | 6.3% urgent unrouted |
| ABL6 | Privacy gate | 97.79 | 0.89 | 0.94 | Privacy violated |
