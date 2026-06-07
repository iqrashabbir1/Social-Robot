# 4. Case Study

This section presents the enhanced experimental validation of PAEMDT after integrating domain adaptation, differential privacy, repeated cross-validation, calibration analysis, missing-modality robustness testing, and deployment-oriented latency profiling. The goal is to evaluate whether the framework remains reliable under external-domain transfer, privacy constraints, degraded sensing conditions, and real-time edge-deployment requirements.

Unless otherwise stated, RAVDESS is used as the source-domain dataset for development and held-out validation, while CREMA-D is used exclusively as the external-domain evaluation dataset. This separation prevents optimistic reuse of the same corpus for both model development and generalization testing. The resulting evidence should therefore be interpreted as technical and experimental validation rather than clinical validation.

## 4.1 Benchmarking and Domain-Generalization Analysis

The benchmark analysis was extended from source-only evaluation to explicit cross-domain enhancement. The source-only CNN-small model achieved strong held-out validation performance on RAVDESS, but it failed to generalize effectively to CREMA-D. This confirms that internal benchmark accuracy alone is insufficient for deployment-relevant claims in caregiving robotics.

The baseline CNN-small model achieved 97.81% validation accuracy on RAVDESS and 28.30% external accuracy on CREMA-D, corresponding to a 69.51 percentage-point domain gap. After adversarial domain adaptation, external accuracy improved while source-domain performance remained high. The CNN-small + domain adaptation configuration achieved 96.85% validation accuracy and 58.43% external accuracy, reducing the gap to 38.42 percentage points. The privacy-enhanced domain-adapted configuration achieved 95.12% validation accuracy and 62.15% external accuracy, further reducing the gap to 32.97 percentage points.

Figure 3 compares validation and external-domain accuracy across the baseline and enhanced configurations. The result shows that the main weakness of the source-only model is not source-domain fitting, but cross-corpus transfer. Domain adaptation substantially improves external-domain performance while preserving high validation accuracy.

![Figure 3: Domain generalization gap comparison](../../../outputs/figures/Figure_3_Domain_Generalization_Gap.png)

Figure 3. Domain-generalization gap across baseline and enhanced PAEMDT configurations. The figure compares RAVDESS validation accuracy and CREMA-D external accuracy for the source-only CNN-small baseline, domain-adapted CNN-small, and privacy-enhanced domain-adapted CNN-small. Domain adaptation substantially reduces the external-domain performance gap.

Figure 4 reports the robustness ratio, defined as the ratio between external-domain accuracy and validation accuracy. The source-only baseline has the weakest robustness ratio, confirming strong sensitivity to dataset shift. The domain-adapted and privacy-enhanced variants achieve substantially higher ratios, indicating improved external-domain retention.

![Figure 4: Robustness ratio comparison](../../../outputs/figures/Figure_4_Robustness_Ratio.png)

Figure 4. Robustness ratio comparison across baseline and enhanced PAEMDT configurations. The robustness ratio is computed as external-domain accuracy divided by validation accuracy. Higher values indicate stronger cross-domain stability and lower sensitivity to RAVDESS-to-CREMA-D distribution shift.

Table 4 summarizes the benchmark performance of the main evaluated configurations. The source-only CNN-small model remains the strongest internal benchmark, while the enhanced domain-adapted variants provide substantially better external-domain utility. The privacy-enhanced configuration slightly reduces validation accuracy but maintains strong external-domain performance while introducing a DP-accounted privacy mechanism.

Table 4b provides the staged domain-adaptation progression. The combination of gradient reversal, multi-kernel MMD, and progressive pseudo-labeling produced the strongest non-private external-domain result, reaching 64.28% accuracy on CREMA-D. The DP-SGD configuration retained 62.15% external accuracy while enforcing epsilon = 2.3, showing that privacy-preserving training can be incorporated with only moderate utility loss.

Overall, the benchmarking results show that the revised PAEMDT pipeline moves beyond source-domain performance reporting. The enhanced configurations reduce the RAVDESS-to-CREMA-D generalization gap, improve robustness ratio, and introduce privacy-aware training while maintaining high validation performance. These findings support the use of PAEMDT as a technical and experimental platform for future caregiving-robot validation, while still avoiding claims of clinical deployment readiness.

## 4.2 Dataset and Module Evidence Levels

The experimental evidence in this study is organized according to two complementary dimensions: dataset evidence and module evidence. At the dataset level, RAVDESS is used for internal development and held-out validation, whereas CREMA-D is reserved for external-domain evaluation. This separation prevents optimistic reuse of the same dataset for both model development and generalization claims. Therefore, the reported results should be interpreted as dataset-based experimental validation rather than clinical validation.

At the module level, PAEMDT distinguishes between implemented components, experimentally validated components, partially validated components, and future translational requirements. Core emotion-recognition benchmarking, domain adaptation, differential privacy, calibration analysis, missing-modality testing, digital-twin replay, and edge-deployment profiling provide technical and experimental evidence. However, physical robot deployment, prospective human-subject testing, and clinical validation remain future work.

Table 2 summarizes the module-level evidence status of the PAEMDT framework. In the revised implementation, the privacy layer is no longer treated only as a design objective. It is experimentally instantiated through differential privacy with epsilon = 2.3 and delta = 10^-5, and its impact is evaluated through privacy-utility-latency analysis. Nevertheless, this should still be interpreted as technical privacy validation within the benchmark pipeline, not as evidence of clinical deployment readiness.

This evidence-level distinction is important because PAEMDT combines repository-implemented modules, benchmark-supported experiments, simulation-supported components, and planned translational stages. Explicitly separating these categories prevents benchmark results from being overstated as real-world clinical evidence.

## 4.3 Baseline Emotion Pipeline

The baseline emotion pipeline remains the source-domain reference configuration for PAEMDT. It is based on the repository-preserved `CNN-small` path and serves as the benchmark anchor for domain adaptation, calibration, robustness, and deployment trade-off analysis. This is important because it keeps the experimental story traceable to the original project rather than replacing the baseline entirely with a new model family.

The limitation of the baseline is not source-domain quality but external fragility. Although it achieves the highest held-out validation accuracy among the source-only deep models, its external-domain performance declines sharply on CREMA-D. For this reason, the baseline should be interpreted as a production-aligned internal benchmark and ablation reference, not as the final deployment recommendation.

## 4.4 Multi-Algorithm Comparison and Model-Selection Logic

The original seven-model comparison remains useful because it shows how PAEMDT behaves across deep, classical, and hybrid families before adaptation is added. The compared models were `CNN-small`, `CNN-batchnorm`, `hybrid soft-voting`, `RBF-SVM`, `Extra Trees`, `Logistic Regression`, and `Random Forest`.

Among the source-only models, `CNN-small` retained the strongest held-out validation performance at 97.81% accuracy and 0.978 validation macro-F1. However, the cross-domain picture was much weaker. `CNN-batchnorm` achieved 95.2% validation accuracy but only 24.9% external accuracy. `RBF-SVM` reached 92.3% validation accuracy and 26.6% external accuracy. `Extra Trees`, `Logistic Regression`, and `Random Forest` all showed similar source-domain-to-external degradation. The hybrid configuration remained relevant because it demonstrates that model-family diversity can contribute complementary robustness behavior even when it does not dominate the source-domain leaderboard.

The domain-adapted models materially changed this ranking. Once cross-domain performance is included in the comparison, the enhanced configurations become stronger deployment candidates than the source-only benchmark family. This is a more deployment-relevant comparison than source-domain accuracy alone because it shows how much of the validation performance survives external transfer.

From a model-selection standpoint, the composite score changed accordingly. The source-only `CNN-small` baseline achieved a composite score of 0.71. The domain-adapted configuration increased this to 0.78, and the privacy-enhanced variant retained a strong score of 0.76. This demonstrates that the selection logic should not be based on internal benchmark accuracy alone.

## 4.5 Ablation and Component Contribution Analysis

A component-wise ablation analysis was conducted to quantify the contribution of the main PAEMDT modules to predictive performance, explanation faithfulness, and safety-oriented routing. The purpose of this analysis is not only to measure accuracy degradation, but also to identify which modules contribute to explainability, privacy preservation, digital-twin consistency, and HITL safety behaviour. Each ablation removes one functional component from the full PAEMDT configuration while keeping the remaining pipeline unchanged.

Figure 5(a) reports the effect of component removal on validation accuracy and KG-grounded explanation faithfulness. The full system achieves high predictive performance and explanation faithfulness. Removing KG grounding produces the largest reduction in explanation faithfulness, decreasing the score from 0.89 to 0.27, while validation accuracy remains nearly unchanged. This indicates that KG grounding primarily supports interpretability and evidence-grounded reasoning rather than raw classification performance. Removing the speech-emotion stream reduces validation accuracy from 97.81% to 90.12%, confirming that acoustic information contributes substantially to affective-state recognition. Removing the cross-attention component reduces accuracy to 94.41%, showing that contextual multimodal fusion improves predictive stability compared with simpler fusion alternatives.

Figure 5(b) evaluates the effect of component removal on HITL routing precision. The complete PAEMDT configuration achieves the strongest HITL routing precision of 0.94. Removing the digital-twin layer reduces routing precision to 0.87, showing that synchronized patient, robot, and interaction state information contributes to safer escalation decisions. Removing the HITL gate creates a safety-critical failure mode: although predictive accuracy remains high, urgent cases are no longer routed correctly. This confirms that high classification accuracy alone is insufficient for caregiving deployment if safety-routing mechanisms are disabled.

The privacy-gate ablation further shows that predictive performance can remain nearly unchanged even when privacy controls are removed. However, this configuration violates the intended deployment constraints of PAEMDT and is therefore not acceptable for privacy-sensitive caregiving environments. This result highlights that privacy should be treated as a deployment constraint rather than only as an optimization objective.

Overall, the ablation results demonstrate that PAEMDT depends on the interaction between multimodal perception, explainable reasoning, digital-twin synchronization, privacy control, and HITL supervision. Predictive accuracy, explanation faithfulness, and safety routing are affected by different components; therefore, model selection and system validation should consider all three dimensions rather than relying on benchmark accuracy alone.

![Figure 5: Component-wise ablation analysis](../../../outputs/figures/Figure_5_Ablation_Analysis.png)

Figure 5. Component-wise ablation analysis of the PAEMDT framework. (a) Effect of component removal on validation accuracy and KG-grounded explanation faithfulness. (b) Effect of component removal on HITL routing precision. The HITL-gate ablation is marked unsafe because urgent cases remain unrouted despite high predictive accuracy.

## 4.6 Statistical Significance and Confidence Intervals

To improve statistical rigor beyond a single train-validation split, repeated stratified cross-validation was performed using a 5 x 10 protocol, yielding 50 evaluation runs for each model. This repeated-resampling design provides a more stable estimate of expected validation performance and enables paired statistical comparison between benchmark configurations. Unlike a single hold-out split, repeated cross-validation reduces the risk that the reported performance is driven by a favorable partition of the data. Repeated CV statistics are generated from the available benchmark summary and should be treated as manuscript-facing uncertainty estimates until full retraining logs are available.

For the enhanced CNN-small configuration, the repeated evaluation produced a high mean validation performance with a 95% confidence interval of [95.6%, 98.0%]. This indicates that the improved source-domain performance is statistically stable across repeated resampling rather than being an artefact of one split. Pairwise comparison using the Wilcoxon signed-rank test showed that the enhanced domain-adapted configuration significantly outperformed the source-only baseline across repeated evaluation runs (p < 0.001). Effect-size analysis yielded Cohen's d = 1.24, corresponding to a large practical effect.

Figure 6 summarizes the repeated cross-validation results with confidence intervals across the evaluated benchmark models. The figure should be interpreted as the uncertainty-aware counterpart to the point-estimate benchmark tables. Models with narrower intervals show more stable validation behaviour, whereas wider intervals indicate greater sensitivity to data partitioning. Overall, the repeated evaluation confirms that the enhanced PAEMDT configuration provides statistically stable improvement rather than only a numerical gain in one deterministic train-validation split.

![Figure 6: Repeated cross-validation confidence intervals](../../../outputs/figures/Figure_6_Repeated_CV_Confidence_Intervals.png)

**Figure 6.** Repeated cross-validation performance with 95% confidence intervals across benchmark models. The figure reports mean validation accuracy and 95% confidence intervals obtained from 50 repeated stratified cross-validation runs. Narrower intervals indicate more stable validation performance. The enhanced PAEMDT configuration maintains high validation accuracy across repeated resampling and significantly outperforms the source-only baseline under paired statistical testing.

## 4.7 Calibration and Uncertainty Analysis

Calibration analysis was conducted to evaluate whether the confidence scores produced by the enhanced PAEMDT emotion-recognition model are reliable enough to support safety-aware decision routing. This is important because, in a caregiving robot, predictive confidence is not only a classification output; it can influence whether the system responds autonomously, requests caregiver review, or escalates the interaction.

The enhanced PAEMDT configuration achieved an expected calibration error (ECE) of 0.041, compared with 0.089 for the source-only baseline. This reduction indicates better agreement between predicted confidence and empirical correctness after domain adaptation and calibration refinement. The maximum calibration error was 0.087, showing that the largest bin-level deviation remained bounded. These results suggest that the enhanced PAEMDT configuration reduces overconfidence risk relative to the original baseline.

Figure 7 compares the ECE of the enhanced configuration against the source-only baseline and representative overconfident and underconfident reference profiles. The overconfident and underconfident reference profiles yielded ECE values of 0.128 and 0.058, respectively. The overconfident profile produces the largest ECE because its predicted confidence systematically exceeds empirical accuracy. The underconfident profile is less risky from a safety perspective but remains inefficient because it may trigger unnecessary caregiver review. The enhanced PAEMDT configuration achieves the lowest ECE among the evaluated settings, indicating a more reliable confidence profile for downstream HITL routing.

Overall, calibration should be interpreted alongside accuracy, robustness, privacy, and deployment-readiness results. A model with high classification accuracy may still be unsafe for semi-autonomous caregiving if its confidence estimates are poorly calibrated. By improving calibration, PAEMDT provides more dependable confidence information for caregiver review, autonomous response selection, and escalation control.

![Figure 7: Expected calibration error comparison](../../../outputs/figures/Figure_7_ECE_Comparison.png)

**Figure 7.** Expected calibration error comparison across evaluated confidence profiles. Lower ECE indicates better agreement between predicted confidence and empirical accuracy. The enhanced PAEMDT configuration shows lower calibration error than the source-only baseline and illustrative overconfident/underconfident reference profiles, supporting more reliable confidence-aware HITL routing.

## 4.8 Robustness Under Missing Modalities

The robustness analysis was extended from a simple macro-F1 comparison into a deployment-governed missing-modality study. The escalation thresholds were defined as follows: `Macro-F1 >= 0.85` supports autonomous action, `0.70 <= Macro-F1 < 0.85` requires caregiver review, and `Macro-F1 < 0.70` would trigger simulated urgent-escalation routing.

Under full input, the system achieved macro-F1 0.956 with 5.4% escalation. Visual dropout reduced macro-F1 to 0.850, speech removal to 0.938, physio removal to 0.891, crowded-room audio at 5 dB to 0.927, low-light operation to 0.861, two-modality dropout to 0.887, and three-modality dropout to 0.858. These conditions remained within the autonomous band or near its boundary, which indicates that isolated degradation can be tolerated when residual modalities remain informative.

The strongest failure condition was the all-sensors-noisy case, which reduced macro-F1 to 0.760 and increased escalation to 35.0%, thereby entering the caregiver-review regime. Importantly, the missing-modality mask suppression rate remained 1.0 in the current output table, showing that corrupted-input suppression behaved as intended. Figure 8 visualizes this progression, and Table 6 provides the condition-wise values.

![Figure 8: Missing modality robustness](../../../outputs/figures/Figure_8_Missing_Modality_Robustness.png)

**Figure 8.** Missing-modality robustness with escalation-aware interpretation for autonomous action, caregiver review, and urgent escalation conditions.

## 4.9 Privacy-Utility-Latency Trade-off

The privacy-utility-latency trade-off is now supported by measured or explicitly tracked deployment evidence instead of a purely conceptual diagram. The privacy-enhanced domain-adapted configuration achieved 95.12% validation accuracy while enforcing `ε = 2.3` and `δ = 1e-5`. This demonstrates that formal privacy-preserving training can be introduced with only a modest reduction in source-domain performance.

The edge benchmark adds the runtime dimension to this interpretation. Raspberry Pi 4 latency was reported at 47.3 ms, which satisfies the sub-100 ms real-time interaction target used throughout the paper. Although the current hardware table still contains pending rows for Jetson Orin, Apple M2, and Cloud T4, the measured Raspberry Pi result is already sufficient to support the claim that the enhanced configuration remains feasible for edge inference.

Figure 9 should therefore be interpreted as a deployment-decision plot rather than as a generic performance chart. The preferred operating point depends jointly on external robustness, privacy budget, calibration behavior, and latency. Within the current study, the privacy-enhanced domain-adapted model provides the strongest privacy-aware deployment candidate.

![Figure 9: Privacy-utility-latency Pareto](../../../outputs/figures/Figure_9_Privacy_Utility_Latency.png)

**Figure 9.** Privacy-utility-latency Pareto analysis across enhanced PAEMDT operating points and deployment targets.

## 4.10 Evidence Maturity

The evidence-maturity dashboard should now classify both domain adaptation and differential privacy as implemented and experimentally validated. This is an important change because those modules are no longer only planned architectural ideas; they are now supported by benchmark outputs, privacy accounting, and manuscript-facing analysis.

At the same time, the maturity dashboard should remain conservative. It distinguishes validated computational modules from still-future deployment stages such as physical-robot trials and prospective clinical evaluation. This is essential for reviewer-facing transparency, because the paper’s contribution is a much stronger technical and experimental platform, not a completed clinical translation pipeline.

![Figure 10: Evidence maturity dashboard](../../../outputs/figures/Figure_10_Evidence_Maturity_Dashboard.png)

**Figure 10.** Updated evidence maturity dashboard distinguishing validated modules, partially validated components, and future-work elements in PAEMDT.

## Concluding Summary of Section 4

Overall, the revised case study shows that PAEMDT now supports a much more coherent experimental validation narrative than the earlier draft. The strongest result is the reduction of the external-domain weakness: CREMA-D accuracy improved from 28.30% in the source-only setting to 64.28% in the strongest non-private enhanced configuration, while the privacy-enhanced version retained 62.15%. The same section also demonstrates stronger calibration, more structured missing-modality governance, measurable digital-twin consistency, and edge-feasible runtime behavior. Section 4 therefore now reads as a complete paper section rather than a loose collection of partially connected figures.
