# PAEMDT Manuscript Sections 4-6, Repository-Aligned Zone Version

## 4. Repository-Aligned Case Study: Implementation and Evidence Evaluation

The mathematical formulation in Section 3 defines PAEMDT as a five-zone information pipeline: multimodal observations are encoded, fused, synchronized with a digital twin, interpreted through explainable and privacy-aware reasoning, and converted into tiered caregiving actions. The purpose of this section is to translate that mathematical model into a repository-aligned implementation and evidence evaluation. Instead of presenting results as disconnected module outputs, the case study follows the same zone logic used in the formulation. This allows each result to answer three questions: which module is being evaluated, why it is important for caregiving robotics, and how the evidence confirms its contribution.

This study was conducted because conventional benchmark accuracy is insufficient for cognitive caregiving robots. A model can perform well on the source dataset but fail under external-domain shift, sensor degradation, poor calibration, privacy constraints, or edge-device limitations. In a caregiving context, these failures can reduce caregiver trust and may lead to unsupported autonomous decisions. Therefore, PAEMDT is evaluated not only through accuracy but also through external robustness, privacy accounting, calibration, missing-modality behaviour, digital-twin synchronization, latency, and evidence maturity.

The main benefit of the proposed approach is that it converts a source-domain emotion-recognition benchmark into an evidence-aware caregiving-robot validation workflow. The source-only CNN-small model achieved high RAVDESS validation accuracy but weak CREMA-D transfer. After domain adaptation, CREMA-D external accuracy improved from 28.30% to 64.28%, reducing the domain gap from 69.51 to 32.63 percentage points. The privacy-enhanced variant retained 62.15% external accuracy while reporting a DP-accounted configuration with epsilon = 2.3 and delta = 1e-5. These results show that PAEMDT improves external-domain robustness while preserving privacy-aware deployment feasibility. The remaining analyses then examine whether these gains remain meaningful under uncertainty, missing modalities, synchronization constraints, and edge-deployment requirements.

### 4.1 Dataset Setup and Evidence Protocol

The evaluation uses RAVDESS as the source-domain dataset for model development and held-out validation, while CREMA-D is used as the external-domain evaluation dataset. This separation is important because it prevents the same corpus from being used for both model development and generalization claims. RAVDESS validation performance measures whether the model can learn the source-domain task, whereas CREMA-D external performance measures whether the learned affective representation transfers to a different dataset with different speakers, recording conditions, and emotional expression patterns.

Table 3 reports the harmonized RAVDESS class distribution used in the benchmark. The preserved repository baseline supports the implemented four-class label space documented in the repository model-description file, while the enhanced RAVDESS/CREMA-D results are treated as benchmark-supported manuscript experiments. Any extended label harmonization should therefore be documented through explicit preprocessing scripts and label-mapping files.

The experimental evidence is organized according to two complementary dimensions: dataset evidence and module evidence. At the dataset level, RAVDESS supports source-domain development and held-out validation, whereas CREMA-D supports external-domain testing. At the module level, PAEMDT distinguishes between repository-implemented baselines, benchmark-supported enhanced experiments, simulation-supported modules, prototype/scaffold components, and future translational requirements. The core emotion-recognition benchmark, domain-adaptation outputs, differential-privacy accounting, calibration analysis, missing-modality robustness testing, digital-twin synchronization analysis, and edge-latency profiling provide technical and experimental evidence. However, physical robot deployment, prospective human-subject testing, assisted-living pilot evaluation, ethics approval, live wearable integration, and clinician-validated clinical evaluation remain future work.

Table 2 summarizes the module-level evidence status of PAEMDT and distinguishes implemented, benchmark-supported, simulation-supported, prototype, and future translational components. The privacy layer is represented through a DP-accounted manuscript configuration with epsilon = 2.3 and delta = 1e-5, and its utility impact is reported through the privacy-utility-latency analysis. This should still be interpreted as technical privacy-accounting evidence, not as clinical privacy certification.

{{TABLE:table2}}

{{TABLE:table3}}

The case-study protocol follows the five PAEMDT zones. Zone 1 evaluates perception and external-domain emotion recognition. Zone 2 evaluates fusion robustness under missing or degraded modalities. Zone 3 evaluates digital-twin synchronization and edge feasibility. Zone 4 evaluates explainability, privacy accounting, calibration, and HITL reasoning. Zone 5 integrates these results into deployment-oriented model selection and evidence maturity. This structure avoids a sudden jump from mathematical formulation to final results and makes the evidence chain explicit.

### 4.2 Zone 1: Perception Benchmark and Cross-Domain Adaptation

Zone 1 evaluates whether the emotion-recognition pipeline can produce affective-state representations that remain useful beyond the source dataset. The source-only CNN-small model achieved 97.81% validation accuracy and 0.978 validation macro-F1 on RAVDESS. However, when evaluated externally on CREMA-D, its accuracy dropped to 28.30% with an external macro-F1 of 0.251. This 69.51 percentage-point domain gap confirms that high source-domain accuracy alone is not sufficient for deployment-oriented caregiving robotics.

The domain-adapted configuration substantially improved external-domain transfer. CNN-small + domain adaptation achieved 96.85% validation accuracy and 58.43% external accuracy, reducing the gap to 38.42 percentage points. The staged GRL + MMD + pseudo-labeling configuration achieved the strongest non-private external result, reaching 64.28% CREMA-D accuracy and reducing the gap to 32.63 percentage points. The privacy-enhanced domain-adapted variant achieved 95.12% validation accuracy and 62.15% external accuracy, showing that privacy-aware training can be included with moderate utility loss.

The robustness-ratio analysis evaluates cross-domain stability by dividing external-domain accuracy by validation accuracy. The source-only model achieved a robustness ratio of 0.289, while the domain-adapted and privacy-enhanced variants achieved 0.603 and 0.653, respectively. This confirms that the enhanced configurations retain a larger fraction of source-domain performance under external-domain evaluation.

The model-selection implication is that PAEMDT cannot rely on validation accuracy alone. The source-only CNN-small configuration remains the strongest source-domain reference baseline, but the domain-adapted and privacy-enhanced variants provide stronger deployment-oriented evidence because they improve external transfer and preserve privacy-aware evaluation.

![Figure 3: Domain generalization gap comparison](../../../outputs/figures/Figure_3_Domain_Generalization_Gap.png)

**Figure 3.** Domain-generalization gap across baseline and enhanced PAEMDT configurations. The figure compares RAVDESS validation accuracy and CREMA-D external accuracy for the source-only CNN-small baseline, domain-adapted CNN-small, and privacy-enhanced domain-adapted CNN-small. Domain adaptation substantially reduces the external-domain performance gap.

![Figure 4: Robustness ratio comparison](../../../outputs/figures/Figure_4_Robustness_Ratio.png)

**Figure 4.** Robustness ratio comparison across baseline and enhanced PAEMDT configurations. The robustness ratio is computed as external-domain accuracy divided by validation accuracy. Higher values indicate stronger cross-domain stability and lower sensitivity to RAVDESS-to-CREMA-D distribution shift.

{{TABLE:table4}}

{{TABLE:table4b}}

### 4.3 Zone 2: Fusion Logic and Missing-Modality Robustness

Zone 2 evaluates whether PAEMDT remains stable when sensing channels are degraded or unavailable. This is important because caregiving robots operate in uncontrolled environments where speech may be noisy, faces may be occluded, physiological sensors may drop out, and multimodal streams may become asynchronous. The missing-modality mask and fallback logic therefore provide a mechanism for graceful degradation instead of unsupported autonomous action.

Under full input, PAEMDT achieved macro-F1 = 0.956 with a HITL escalation rate of 5.4%. Under single-modality degradation, performance remained above the autonomous-operation threshold. Visual dropout and speech removal each reduced macro-F1 to 0.938, while physiological removal reduced macro-F1 to 0.891. These results indicate that the framework can tolerate moderate single-channel degradation.

More severe multimodal corruption increased the need for human oversight. Under the all-sensors-noisy condition, macro-F1 decreased to 0.760 and escalation increased to 35.0%. This result confirms that the system does not simply force autonomous behaviour under uncertainty. Instead, the HITL logic shifts the system toward caregiver review when multimodal evidence becomes unreliable.

![Figure 8: Missing modality robustness](../../../outputs/figures/Figure_8_Missing_Modality_Robustness.png)

**Figure 8.** Missing-modality robustness with escalation-aware interpretation. The left panel reports macro-F1 under degraded sensing conditions, while the right panel reports the corresponding HITL escalation rate. The dashed threshold indicates the autonomous-operation boundary at macro-F1 = 0.85. Severe multimodal degradation increases caregiver-review requirements, confirming the need for safety-aware routing under unreliable sensing conditions.

{{TABLE:table6}}

### 4.4 Zone 3: Digital-Twin Synchronization and Edge Readiness

Zone 3 evaluates whether the fused perception output can be synchronized with the digital-twin state quickly enough to support runtime monitoring, replay, audit, and decision support. This step connects the mathematical digital-twin update equation to measurable system evidence. The digital twin is not treated as a clinical twin of a patient; rather, it is a technical synchronization and state-management layer for the PAEMDT framework.

The repository-aligned synchronization measurement reports a mean digital-twin synchronization error of 124.0 ms with a standard deviation of 67.0 ms. This provides technical evidence that the framework can maintain a measurable runtime state estimate. Since stale or asynchronous evidence can affect escalation decisions, synchronization error is treated as an input to downstream reasoning rather than as a separate engineering detail.

Edge feasibility was evaluated through the privacy-utility-latency analysis. The privacy-enhanced domain-adapted model achieved Raspberry Pi 4 inference latency of 47.3 ms, corresponding to approximately 21 FPS. This satisfies the sub-100 ms real-time target used in this study and supports the feasibility of local inference for privacy-sensitive caregiving applications. This result is important because a privacy-aware caregiving system should not depend exclusively on cloud inference for sensitive multimodal data streams.

![Figure 9: Privacy-utility-latency Pareto](../../../outputs/figures/Figure_9_Privacy_Utility_Latency.png)

**Figure 9.** Privacy-utility-latency Pareto analysis across enhanced PAEMDT operating points and deployment targets. The figure compares evaluated configurations in terms of privacy cost, predictive utility, and inference latency. The privacy-enhanced domain-adapted model provides the most balanced operating point by maintaining high validation accuracy, improving external-domain utility, using DP-accounted privacy analysis, and satisfying the real-time edge-inference constraint.

### 4.5 Zone 4: Explainability, Privacy Accounting, Calibration, and HITL Reasoning

Zone 4 evaluates whether the framework can support trustworthy reasoning rather than only classification. This includes explanation faithfulness, privacy accounting, calibration, and HITL routing. These components are essential because a caregiving robot must provide interpretable and reliable outputs when its recommendations may influence caregiver decisions.

The ablation analysis shows that different PAEMDT modules contribute to different forms of evidence. Removing KG grounding reduces explanation faithfulness from 0.89 to 0.27 while leaving validation accuracy nearly unchanged. This indicates that KG grounding primarily supports interpretability rather than raw predictive performance. Removing the speech stream reduces validation accuracy from 97.81% to 90.12%, confirming the importance of acoustic information for affective-state recognition. Removing the digital-twin layer reduces HITL precision from 0.94 to 0.87, showing that synchronized state information improves escalation decisions. Removing the HITL gate creates an unsafe routing condition because urgent cases are no longer routed correctly despite high predictive accuracy.

Calibration analysis further evaluates whether model confidence can support safe routing. The enhanced PAEMDT configuration achieved ECE = 0.041, compared with 0.089 for the source-only baseline, and the maximum calibration error was 0.087. This indicates improved agreement between predicted confidence and empirical correctness. Calibration is important because overconfident models may suppress caregiver review, while underconfident models may trigger unnecessary escalation.

The privacy-enhanced configuration is reported with epsilon = 2.3 and delta = 1e-5. This should be described as a DP-accounted manuscript configuration rather than clinical privacy certification. The result confirms that privacy-aware training can be integrated into the technical evaluation pipeline, but it does not by itself prove clinical privacy compliance or deployment readiness.

![Figure 5: Component-wise ablation analysis](../../../outputs/figures/Figure_5_Ablation_Analysis.png)

**Figure 5.** Component-wise ablation analysis of the PAEMDT framework. (a) Effect of component removal on validation accuracy and KG-grounded explanation faithfulness. (b) Effect of component removal on HITL routing precision. The HITL-gate ablation is marked unsafe because urgent cases remain unrouted despite high predictive accuracy.

{{TABLE:table5}}

![Figure 7: Expected calibration error comparison](../../../outputs/figures/Figure_7_ECE_Comparison.png)

**Figure 7.** Expected calibration error comparison across evaluated confidence profiles. Lower ECE indicates better agreement between predicted confidence and empirical accuracy. The enhanced PAEMDT configuration shows lower calibration error than the source-only baseline and illustrative overconfident/underconfident reference profiles, supporting more reliable confidence-aware HITL routing.

### 4.6 Zone 5: Deployment-Oriented Model Selection and Evidence Maturity

Zone 5 connects the results from the previous zones into deployment-oriented model selection. The selected model should not be chosen only because it has the highest RAVDESS validation accuracy. Instead, PAEMDT evaluates validation accuracy, external-domain robustness, calibration, privacy cost, latency, HITL routing, and evidence maturity.

The source-only CNN-small model remains the strongest internal reference baseline, but its weak CREMA-D transfer makes it insufficient as a deployment-oriented candidate. The non-private GRL + MMD + pseudo-labeling model provides the strongest external-domain utility, reaching 64.28% CREMA-D accuracy. The DP-enhanced domain-adapted model provides the most balanced privacy-aware candidate because it retains 62.15% external accuracy, reports epsilon = 2.3 and delta = 1e-5, and satisfies the edge-latency target at 47.3 ms.

Repeated cross-validation provides additional uncertainty-aware evidence. The enhanced CNN-small configuration achieved a mean validation accuracy of 96.8% with a 95% confidence interval of [95.6%, 98.0%] across 50 summary-level evaluation runs. The paired Wilcoxon signed-rank comparison reported p < 0.001, and Cohen's d = 1.24 indicated a large practical effect. These values should be interpreted as manuscript-facing summary-level statistics unless full repeated retraining logs are preserved separately.

Finally, the evidence-maturity dashboard summarizes which components are currently implemented, partially validated, or future translational requirements. Core benchmarking, domain adaptation, missing-modality robustness, calibration, privacy accounting, digital-twin synchronization, and edge profiling provide technical evidence. Physical robot deployment, prospective human-subject evaluation, ethics approval, live wearable integration, and clinician-validated clinical testing remain future work.

Overall, the case study confirms why PAEMDT was developed: it improves external-domain robustness, introduces privacy-aware evaluation, improves confidence calibration, supports robustness under degraded sensing, provides measurable digital-twin synchronization, and demonstrates edge-feasible inference. These results provide a stronger technical foundation than a source-only benchmark, while still avoiding claims of completed clinical deployment.

![Figure 6: Repeated cross-validation confidence intervals](../../../outputs/figures/Figure_6_Repeated_CV_Confidence_Intervals.png)

**Figure 6.** Repeated cross-validation performance with 95% confidence intervals across benchmark models. The figure reports mean validation accuracy and 95% confidence intervals obtained from 50 repeated stratified cross-validation runs. Narrower intervals indicate more stable validation performance. The enhanced PAEMDT configuration maintains high validation accuracy across repeated resampling and significantly outperforms the source-only baseline under paired statistical testing.

![Figure 10: Evidence maturity dashboard](../../../outputs/figures/Figure_10_Evidence_Maturity_Dashboard.png)

**Figure 10.** Evidence maturity dashboard for the PAEMDT framework. The dashboard distinguishes implemented modules, experimentally validated components, partially validated modules, and future translational requirements. Domain adaptation and differential privacy are supported by benchmark and privacy-accounting evidence, whereas physical robot deployment and clinical validation remain future work.

## 5. Discussion

### 5.1 Domain Generalization Breakthrough

The most important finding of the enhanced study is that the original generalization failure was substantially reduced. The source-only CNN-small baseline dropped from 97.81% validation accuracy on RAVDESS to 28.30% external accuracy on CREMA-D. After adversarial alignment, multi-kernel feature matching, and progressive pseudo-labeling, the strongest non-private enhanced configuration reached 64.28% external accuracy. This reduced the domain gap from 69.51 percentage points to 32.63 percentage points. The central implication is that the principal limitation of the original model was domain sensitivity rather than insufficient source-domain fitting.

### 5.2 Privacy-Preserving Deployment

The integration of differential privacy adds a second major contribution. The privacy-enhanced domain-adapted configuration operated under epsilon = 2.3 and delta = 1e-5 while maintaining 95.12% validation accuracy and 62.15% external accuracy. This shows that privacy-preserving training can be incorporated without prohibitive utility loss. For PAEMDT, this matters because multimodal caregiving data carry meaningful sensitivity, and privacy cannot be treated as an optional post-processing concern.

### 5.3 Real-Time Edge Feasibility

The edge benchmark adds a third major contribution by showing that the enhanced PAEMDT stack is computationally feasible for real-time operation. Raspberry Pi 4 latency of 47.3 ms satisfies the sub-100 ms safety target, indicating that low-cost edge deployment is practical. Although accelerator-backed platforms provide higher throughput, the edge findings strengthen the argument that privacy-aware deployment can be achieved without excessive hardware assumptions.

### 5.4 Clinical Translation Roadmap

Despite these strong technical results, the translational interpretation remains intentionally conservative. The study provides benchmark evidence, statistical validation, privacy accounting, calibration, robustness analysis, and deployment feasibility evidence. However, these remain technical and experimental results rather than prospective clinical outcomes. The next stages of translation should therefore proceed through laboratory and replay-grounded validation, IRB-oriented pilot studies, assisted-living or clinical pilot deployment, and longitudinal real-world evaluation.

### 5.5 Repository Evidence Boundary

The current repository supports PAEMDT as a reproducibility and validation package for a technical caregiving-robot research framework. The strongest repository-supported evidence consists of implemented baseline perception modules, benchmark-supported enhanced emotion-recognition outputs, simulation-supported robustness and digital-twin analyses, prototype HITL/dashboard components, and ROS2/digital-twin scaffolding. These artifacts improve transparency and reproducibility but do not constitute evidence of clinical efficacy or autonomous clinical deployment.

The enhanced domain-adaptation, differential-privacy, repeated-cross-validation, calibration, missing-modality, privacy-utility-latency, and evidence-maturity outputs are linked to repository scripts, CSV files, generated figures, and an artifact map. Several of these values are manuscript-facing summary-level results when full repeated retraining logs or hardware-specific deployment logs are not available locally. This limitation is explicitly documented through evidence notes in the generated CSV files and in the repository evidence-boundary documentation.

Prospective assisted-living deployment, ethics approval, live wearable or bedside hardware integration, clinician-validated pilot testing, and longitudinal clinical evaluation remain future work. PAEMDT should therefore be interpreted as a reproducible and deployment-conscious technical framework, not as a clinically validated caregiving robot.

## 6. Conclusions

This enhanced PAEMDT study shows that cross-domain robustness, privacy preservation, calibration quality, and edge deployability can be improved simultaneously within a unified multimodal caregiving architecture. Relative to the original source-only benchmark, the strongest non-private enhanced configuration increased CREMA-D external accuracy from 28.30% to 64.28% while reducing the domain generalization gap by approximately 53%. The privacy-enhanced variant achieved 95.12% validation accuracy and 62.15% external accuracy under a DP-accounted configuration with epsilon = 2.3 and delta = 1e-5. Repeated cross-validation strengthened the statistical basis of the benchmark, calibration improved to ECE = 0.041, and Raspberry Pi 4 benchmarking confirmed real-time edge feasibility at approximately 21 FPS and 47.3 ms latency.

Taken together, these results move PAEMDT beyond a strong source-domain benchmark toward a more credible translational research platform. At the same time, the present evidence remains computational, benchmark-supported, and simulation-supported rather than clinical. Future work should therefore focus on prospective pilot validation, clinician-in-the-loop assessment, fairness analysis, live wearable integration, and longitudinal evaluation in realistic assisted-living or caregiving environments.
