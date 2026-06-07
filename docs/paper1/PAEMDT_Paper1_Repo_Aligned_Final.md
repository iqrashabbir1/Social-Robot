# PAEMDT: Privacy-Aware Explainable Multimodal Digital-Twin Cognitive Caregiving Robot

**Author:** Iqra Shabbir

## Abstract

This paper presents PAEMDT, a privacy-aware explainable multimodal digital-twin framework for cognitive caregiving robots, with emphasis on cross-domain emotion recognition, synchronization, robustness, and deployment-aware validation. The enhanced evaluation combines source-domain benchmarking, cross-domain adaptation, repeated cross-validation, calibration analysis, missing-modality stress testing, differential privacy, digital-twin predictive validation, and edge deployment profiling. On RAVDESS, validation performance remains high (97.81% for the source-only baseline and 95.12% for the privacy-enhanced adapted model), while CREMA-D external accuracy improves from 28.30% in the source-only setting to 64.28% with gradient reversal, multi-kernel MMD, and progressive pseudo-labeling, and remains 62.15% under DP-SGD at epsilon = 2.3. The measured domain gap is therefore reduced from 69.51 to 32.63 percentage points. Calibration improves to ECE = 0.041, digital-twin synchronization is measured at 124.0 +/- 67.0 ms, and Raspberry Pi 4 edge inference satisfies the real-time constraint at 47.3 ms latency (approximately 21 FPS). These results constitute technical and experimental validation rather than clinical deployment evidence, but they establish a reproducible, privacy-aware, and deployment-conscious foundation for future cognitive caregiving robot studies.

## 1. Introduction

Cognitive caregiving robots require more than perception models. They also require a measurable runtime backbone that can synchronize sensing, mirror system state in a digital twin, and support controlled replay, audit, and disturbance analysis. This repository originally contained baseline social-robot perception code, but not a clean first-paper experimental path. PAEMDT addresses that gap by reorganizing the codebase around three foundational layers: digital-twin validation, multimodal synchronization, and emotion-recognition benchmarking.

The resulting contribution is a technical and publication-oriented framework rather than a deployment claim. The repository preserves an implemented visual baseline, adds reproducible evaluation infrastructure, and separates synthetic, pilot, replay-grounded, and external-dataset evidence so that the maturity of each result remains explicit.

A key enhancement of the revised manuscript is the introduction of domain adaptation to address the original cross-corpus generalization gap between RAVDESS and CREMA-D. This addition allows the paper to move from a strong source-domain benchmark to a more credible translational research platform.

## 2. Framework Architecture

PAEMDT is organized as a modular multimodal architecture in which sensing, synchronization, benchmarking, privacy controls, and digital-twin services are explicitly separated but operationally linked. Figure 2a and Figure 2b illustrate the high-level system view and the interaction between perception, digital-twin, and decision-support layers.

At runtime, the framework uses ROS2-compatible topic interfaces such as /camera/image_raw, /audio/stream, /robot_pose, /event_log, and /system_health to structure the data flow. A digital twin mirrors timestamped activity, enables replay and audit, and now supports predictive validation. On top of that, the benchmarking layer evaluates deep, classical, hybrid, domain-adapted, and privacy-preserving emotion-recognition configurations under controlled experimental conditions.

The mathematical formulation in the next section makes this architecture explicit as a five-zone information pipeline in which the output of each zone becomes the input to the next.

## 3. Mathematical Formulation Aligned with the PAEMDT Architecture

This section formalizes the information flow of PAEMDT according to the five architecture zones. The formulation is intentionally modular and evidence-aware, so that repository-implemented, simulation-supported, and future clinically validated components can be distinguished without collapsing them into a single undifferentiated claim. The output of each zone is defined explicitly and then used as the input to the next zone, thereby preserving the closed-loop logic of the PAEMDT framework.

### 3.1 Zone 1 -- Multimodal Perception and Modality Encoding

At time t, the raw multimodal observation is defined as:

(1) X_t = {x_t^v, x_t^a, x_t^p, x_t^m, x_t^l}

where x_t^v denotes the visual observation, x_t^a denotes the acoustic observation, x_t^p denotes the physiological observation, x_t^m denotes the motion or activity observation, and x_t^l denotes the language or contextual observation.

Each modality is encoded by a modality-specific feature extractor:

(2) y_t^j = phi_j(x_t^j),   j in {v, a, p, m, l}

The set of modality embeddings is then written as:

(3) Y_t = {y_t^v, y_t^a, y_t^p, y_t^m, y_t^l}

The output of Zone 1 is Y_t, the set of modality-specific embeddings. This output becomes the input to Zone 2. In the current repository-validated baseline, the visual stream is implemented using DeepFace-based facial emotion analysis, and the acoustic stream is implemented using MFCC-based speech emotion recognition with an RBF-SVM classifier. Physiological, motion, and language streams are treated as simulation-supported or planned modules depending on the current evidence level [10], [15], [18], [26], [27].

### 3.2 Zone 2 -- Missing-Modality Masking and Cross-Modal Fusion

To model missing or unreliable sensing channels, a modality-availability mask is defined as:

(4) M_t = {m_t^v, m_t^a, m_t^p, m_t^m, m_t^l},   m_t^j in {0, 1}

where m_t^j = 1 indicates that modality j is available and m_t^j = 0 indicates that the modality is missing, corrupted, or judged unreliable.

Fusion over the modality embeddings is defined by a generic operator F(.):

(5) F_t = F(Y_t, M_t)

In the current repository-aligned implementation, the baseline fusion is rule-based face-speech fusion. In the extended PAEMDT architecture, F(.) can be instantiated as an attention-inspired or cross-attention multimodal fusion operator. A generic attention-style formulation is:

(6) Q_t = W_Q Y_t,   K_t = W_K Y_t,   V_t = W_V Y_t

(7) F_t = softmax((Q_t K_t^T) / sqrt(d_k)) V_t

The missing-modality mask is then applied to the fused representation:

(8) F_t^* = F_t odot M_t

The output of Zone 2 is F_t^*, the masked fused multimodal representation. This output becomes the input to Zone 3 and also supports the reasoning layer in Zone 4. The repository currently supports a baseline fusion implementation and benchmark-level attention-inspired fusion logic; full end-to-end cross-attention learning is treated as an extendable architecture component rather than an already deployed subsystem [10], [15], [18], [22], [23].

### 3.3 Zone 3 -- Digital-Twin State Update and Synchronization

The digital-twin state is defined as:

(9) S_t^DT = {s_t^pat, s_t^rob, s_t^env, s_t^int}

where s_t^pat denotes the patient state, s_t^rob denotes the robot state, s_t^env denotes the environmental context, and s_t^int denotes the interaction history.

The twin update is written as:

(10) S_t^DT = U(S_{t-1}^DT, F_t^*, u_{t-1})

where U(.) is the digital-twin update function and u_{t-1} is the previous caregiving action fed back from Zone 5 at the previous time step. To measure synchronization freshness, the digital-twin synchronization error is defined as:

(11) epsilon_t^DT = | t_now - max(t_t^v, t_t^a, t_t^p, t_t^m, t_t^l) |

This expression measures how far the digital twin lags behind the newest available multimodal evidence. A large synchronization error indicates stale or asynchronous evidence and should influence downstream HITL routing. The output of Zone 3 is therefore the updated digital-twin state S_t^DT together with the synchronization error epsilon_t^DT. These outputs become inputs to Zone 4 for explainable reasoning, risk assessment, and safety routing [12], [19], [24], [25]. In the current validation, the synchronization term is also empirically supported by a measured mean latency of 124.0 +/- 67.0 ms.

### 3.4 Zone 4 -- Explainability, Privacy-Aware Inference, and HITL Reasoning

PAEMDT represents structured domain knowledge through a care-relevant knowledge graph:

(12) G = (V, E)

where V is the set of care-relevant concepts and E is the set of semantic relations among them.

Knowledge-grounded evidence retrieval is defined as:

(13) E_t = R(G, F_t^*, S_t^DT)

where E_t is the set of retrieved evidence nodes and relations associated with the current fused observation and the current digital-twin state.

Explanation generation is defined as:

(14) e_t = G_exp(F_t^*, S_t^DT, E_t)

To support downstream reasoning, the fused representation, twin state, and retrieved evidence are combined into a single reasoning vector:

(15) z_t = [F_t^*, S_t^DT, E_t]

The health-risk score is defined as:

(16) r_t = sigma(w_r^T z_t + b_r)

and the anomaly score is defined as:

(17) a_t = D(z_t, z_ref)

where D(.) is a deviation measure with respect to a reference baseline state z_ref.

Privacy-aware transformation over the raw evidence is represented as:

(18) X_t_tilde = Pi(X_t; lambda)

where Pi(.) is the privacy-control function and lambda is the selected privacy configuration. In the current repository-aligned interpretation, Pi(.) represents architecture-level privacy control and privacy-utility analysis, not a formal differential-privacy guarantee by itself. Formal privacy guarantees are reported only when epsilon accounting is explicitly implemented in the training pipeline.

The HITL routing tier is then defined as:

(19) T_t = { urgent escalation, if r_t > tau_r^high OR a_t > tau_a^high; caregiver review, if r_t > tau_r^mid OR a_t > tau_a^mid; autonomous response, otherwise }

The output of Zone 4 is the tuple {e_t, r_t, a_t, X_t_tilde, T_t}. These outputs become inputs to Zone 5 for tiered caregiving action selection. This zone therefore couples explainability, risk scoring, anomaly assessment, privacy filtering, and HITL escalation within one reasoning layer [11], [13], [14], [16], [17], [20], [21], [22], [23], [28], [29].

### 3.5 Zone 5 -- Tiered Caregiving Action and Deployment-Oriented Model Selection

The caregiving action policy is defined as:

(20) u_t = pi(F_t^*, S_t^DT, e_t, r_t, a_t, X_t_tilde, T_t)

where u_t belongs to the set {autonomous response, caregiver review, urgent escalation}. Zone 5 closes the PAEMDT loop. The selected action u_t is logged and becomes part of the next digital-twin update through u_{t-1} in Zone 3 at the following time step.

Deployment-oriented model selection is represented by:

(21) Score_k = alpha F1_k + beta R_k + gamma C_k - delta L_k - eta P_k

where F1_k is the macro-F1 score of model k, R_k is external-domain robustness, C_k is calibration quality, L_k is latency, and P_k is privacy cost. The numerical values of these weights are specified only in the case-study section because they are experiment-specific and should not be treated as universal model parameters.

External-domain robustness is defined as:

(22) R_k = Acc_k^ext / Acc_k^val

and calibration error is defined as:

(23) ECE = sum_{b=1}^{B} (|B_b| / N) | acc(B_b) - conf(B_b) |

where B_b is the set of samples in calibration bin b, N is the total number of samples, acc(B_b) is the empirical accuracy in bin b, and conf(B_b) is the mean confidence in bin b. The model-selection score prevents the system from selecting a model based only on internal benchmark accuracy. This is important because a model may have high validation accuracy but weak external robustness, poor calibration, excessive latency, or unfavorable privacy characteristics [22], [23], [25], [26], [27], [28].

The formulation above defines the architecture-level information flow of PAEMDT. The following case study evaluates which parts of this pipeline are repository-implemented, simulation-supported, or planned for future clinical validation.

## 4. Case Study

This section presents the repository-aligned case study for PAEMDT after the integration of domain adaptation, DP-accounted privacy analysis, repeated cross-validation, calibration analysis, missing-modality robustness testing, digital-twin synchronization analysis, and edge-latency profiling. The case study is organized to distinguish source-domain validation, external-domain evaluation, deployment-oriented constraints, and evidence maturity. Unless otherwise stated, RAVDESS is used for source-domain development and held-out validation, whereas CREMA-D is used for external-domain evaluation.

### 4.1 Benchmarking

Table 3 reports the RAVDESS class distribution used for the harmonized benchmark. Building on this dataset summary, the enhanced benchmark evaluates whether the source-domain model remains credible under cross-corpus transfer. The source-only CNN-small model achieved 97.81% validation accuracy on RAVDESS but only 28.30% external accuracy on CREMA-D, corresponding to a 69.51 percentage-point domain gap. This result shows that high source-domain accuracy alone is not sufficient for deployment-relevant emotion recognition.

The enhanced domain-adaptation pipeline reduced this cross-domain gap. The CNN-small + domain-adaptation configuration achieved 96.85% validation accuracy and 58.43% external accuracy, reducing the gap to 38.42 percentage points. The privacy-enhanced domain-adapted configuration achieved 95.12% validation accuracy and 62.15% external accuracy, reducing the gap to 32.97 percentage points while retaining a DP-accounted privacy configuration. These values are generated from `outputs/tables/enhanced_benchmark_comparison.csv` and `outputs/csv/domain_generalization_results.csv`.

![Figure 3: Domain generalization gap comparison](../../../outputs/figures/Figure_3_Domain_Generalization_Gap.png)

**Figure 3.** Domain-generalization gap across baseline and enhanced PAEMDT configurations. The figure compares RAVDESS validation accuracy and CREMA-D external accuracy for the source-only CNN-small baseline, domain-adapted CNN-small, and privacy-enhanced domain-adapted CNN-small. Domain adaptation substantially reduces the external-domain performance gap.

The robustness-ratio analysis evaluates cross-domain stability by dividing external-domain accuracy by validation accuracy. The source-only CNN-small model achieved a robustness ratio of 0.289, whereas the domain-adapted and privacy-enhanced adapted variants achieved 0.603 and 0.653, respectively. This indicates that the enhanced configurations retain a substantially larger fraction of their source-domain performance when evaluated on CREMA-D.

![Figure 4: Robustness ratio comparison](../../../outputs/figures/Figure_4_Robustness_Ratio.png)

**Figure 4.** Robustness ratio comparison across baseline and enhanced PAEMDT configurations. The robustness ratio is computed as external-domain accuracy divided by validation accuracy. Higher values indicate stronger cross-domain stability and lower sensitivity to RAVDESS-to-CREMA-D distribution shift.

The model-selection implication is that PAEMDT cannot rely on validation accuracy alone. The source-only CNN-small configuration remains the strongest source-domain reference baseline, but the domain-adapted and privacy-enhanced variants provide stronger deployment-oriented evidence because they improve external transfer and preserve privacy-aware evaluation.

### 4.2 Dataset and Module Evidence Levels

The experimental evidence in this study is organized according to two complementary dimensions: dataset evidence and module evidence. At the dataset level, RAVDESS is used for source-domain development and held-out validation, whereas CREMA-D is reserved for external-domain evaluation. This separation prevents optimistic reuse of the same corpus for both model development and generalization claims. The resulting evidence should therefore be interpreted as dataset-based technical validation rather than clinical validation.

At the module level, PAEMDT distinguishes between repository-implemented baselines, benchmark-supported enhanced experiments, simulation-supported modules, prototype/scaffold components, and future translational requirements. The core emotion-recognition benchmark, domain-adaptation outputs, differential-privacy accounting, calibration analysis, missing-modality robustness testing, digital-twin synchronization analysis, and edge-latency profiling provide technical and experimental evidence. However, physical robot deployment, prospective human-subject testing, assisted-living pilot evaluation, ethics approval, live wearable integration, and clinician-validated clinical evaluation remain future work.

Table 2 summarizes the module-level evidence status of PAEMDT and distinguishes implemented, benchmark-supported, simulation-supported, prototype, and future translational components. In the revised repository, the privacy layer is no longer treated only as a design objective. It is represented through a DP-accounted manuscript configuration with epsilon = 2.3 and delta = 1e-5, and its utility impact is reported through the privacy-utility-latency analysis. This should still be interpreted as technical privacy-accounting evidence, not as clinical privacy certification.

This evidence-level distinction is important because PAEMDT combines repository-implemented modules, benchmark-supported experiments, simulation-supported components, and planned translational stages. Explicitly separating these categories prevents benchmark results from being overstated as real-world clinical evidence.

### 4.3 Baseline Emotion Pipeline

The baseline emotion pipeline serves as the source-domain reference configuration for the enhanced PAEMDT evaluation. In this configuration, CNN-small is used as the primary emotion-recognition model trained and validated on RAVDESS, while the enhanced domain-adaptation and privacy-preserving variants are evaluated as deployment-oriented extensions. This baseline is therefore not treated as the final deployable configuration, but as the reference point for measuring cross-domain improvement, privacy-aware adaptation, calibration behaviour, and deployment feasibility.

The source-only CNN-small baseline achieved strong internal performance on RAVDESS, with 97.81% validation accuracy and 0.978 validation macro-F1. However, its external-domain performance on CREMA-D remained weak, confirming that high source-domain accuracy alone is insufficient for reliable real-world transfer. This result motivates the enhanced PAEMDT pipeline, where adversarial alignment, feature-distribution matching, pseudo-label learning, DP-accounted privacy analysis, and calibration evaluation are introduced to improve domain robustness and deployment readiness.

The preserved repository baseline supports the implemented four-class label space documented in the repository model-description file. The enhanced RAVDESS/CREMA-D domain-adaptation results are treated as benchmark-supported manuscript experiments, and any extended label harmonization should be documented through explicit preprocessing scripts and label-mapping files.

### 4.4 Multi-Algorithm Comparison and Model-Selection Logic

The multi-algorithm comparison is interpreted in two complementary layers. First, the original seven-model benchmark identifies the strongest source-domain configuration among deep-learning, classical machine-learning, and hybrid models. Second, the enhanced domain-adaptation and privacy-preserving variants evaluate whether this source-domain baseline can be made more suitable for external-domain transfer and privacy-aware deployment. This distinction is important because deployment suitability cannot be inferred from source-domain validation performance alone.

Among the evaluated benchmark models, CNN-small remained the strongest held-out validation performer, achieving 97.81% validation accuracy and 0.978 validation macro-F1 on RAVDESS. However, the same source-only configuration achieved only 28.30% accuracy and 0.251 macro-F1 on CREMA-D, demonstrating a substantial cross-domain generalization gap. After domain adaptation was introduced, external accuracy improved to 58.43%, while the staged GRL + MMD + pseudo-labeling configuration reached 64.28%. The privacy-enhanced variant retained 62.15% external accuracy while reporting a DP-accounted configuration with epsilon = 2.3 and delta = 1e-5.

These results show that model selection in PAEMDT must follow a multi-criteria logic rather than an accuracy-only criterion. A model with high internal validation accuracy may still be unsuitable if it exhibits weak external-domain robustness, poor calibration, excessive latency, or insufficient privacy protection. Therefore, the enhanced PAEMDT selection logic jointly considers validation performance, external robustness, privacy cost, calibration quality, and edge-deployment feasibility. Under this interpretation, the source-only CNN-small model remains the strongest internal benchmark, the domain-adapted model provides the strongest external-domain utility, and the DP-enhanced domain-adapted model provides the most balanced privacy-aware deployment candidate.

### 4.5 Ablation and Component Contribution Analysis

A component-wise ablation analysis was conducted to quantify the contribution of the main PAEMDT modules to predictive performance, explanation faithfulness, and safety-oriented routing. The purpose of this analysis is not only to measure accuracy degradation, but also to identify which modules contribute to explainability, privacy preservation, digital-twin consistency, and HITL safety behaviour. Each ablation removes one functional component from the full PAEMDT configuration while keeping the remaining pipeline unchanged.

Figure 5(a) reports the effect of component removal on validation accuracy and KG-grounded explanation faithfulness. Removing KG grounding reduces explanation faithfulness from 0.89 to 0.27 while leaving validation accuracy nearly unchanged, showing that KG grounding primarily supports interpretability and evidence-grounded reasoning. Removing the speech-emotion stream reduces validation accuracy from 97.81% to 90.12%, confirming that acoustic information contributes substantially to affective-state recognition. Removing the cross-attention component reduces accuracy to 94.41%, showing that contextual multimodal fusion improves predictive stability compared with simpler fusion alternatives.

Figure 5(b) evaluates HITL routing precision under component removal. The complete PAEMDT configuration achieves HITL routing precision of 0.94. Removing the digital-twin layer reduces routing precision to 0.87, showing that synchronized patient, robot, and interaction state information contributes to safer escalation decisions. Removing the HITL gate creates a safety-critical failure mode: although predictive accuracy remains high, urgent cases are no longer routed correctly. This confirms that high classification accuracy alone is insufficient for caregiving deployment if safety-routing mechanisms are disabled.

![Figure 5: Component-wise ablation analysis](../../../outputs/figures/Figure_5_Ablation_Analysis.png)

**Figure 5.** Component-wise ablation analysis of the PAEMDT framework. (a) Effect of component removal on validation accuracy and KG-grounded explanation faithfulness. (b) Effect of component removal on HITL routing precision. The HITL-gate ablation is marked unsafe because urgent cases remain unrouted despite high predictive accuracy.

### 4.6 Statistical Significance and Confidence Intervals

To improve statistical rigor beyond a single train-validation split, repeated stratified cross-validation was performed using a 5 x 10 protocol, yielding 50 evaluation runs for each model. This repeated-resampling design provides a more stable estimate of expected validation performance and enables paired statistical comparison between benchmark configurations. Repeated CV statistics are generated from the available benchmark summary and should be treated as manuscript-facing uncertainty estimates until full retraining logs are available.

For the enhanced CNN-small configuration, repeated evaluation produced a mean validation accuracy of 96.8% with a 95% confidence interval of [95.6%, 98.0%]. Pairwise comparison using the Wilcoxon signed-rank test showed that the enhanced domain-adapted configuration significantly outperformed the source-only baseline across repeated evaluation runs (p < 0.001). Effect-size analysis yielded Cohen's d = 1.24, corresponding to a large practical effect. These values are generated by `src/evaluation/run_repeated_cv_statistics.py` and summarized in `outputs/tables/repeated_cv_summary.csv` and `outputs/tables/statistical_test_summary.csv`.

![Figure 6: Repeated cross-validation confidence intervals](../../../outputs/figures/Figure_6_Repeated_CV_Confidence_Intervals.png)

**Figure 6.** Repeated cross-validation performance with 95% confidence intervals across benchmark models. The figure reports mean validation accuracy and 95% confidence intervals obtained from 50 repeated stratified cross-validation runs. Narrower intervals indicate more stable validation performance. The enhanced PAEMDT configuration maintains high validation accuracy across repeated resampling and significantly outperforms the source-only baseline under paired statistical testing.

### 4.7 Calibration and Uncertainty Analysis

Calibration analysis was conducted to evaluate whether the confidence scores produced by the enhanced PAEMDT emotion-recognition model are reliable enough to support safety-aware decision routing. In caregiving robotics, predictive confidence can influence whether the system responds autonomously, requests caregiver review, or escalates the interaction.

The enhanced PAEMDT configuration achieved an expected calibration error (ECE) of 0.041, compared with 0.089 for the source-only baseline. The maximum calibration error was 0.087. These values indicate better agreement between predicted confidence and empirical correctness after domain adaptation and calibration refinement. Calibration should therefore be interpreted alongside accuracy, robustness, privacy, and deployment-readiness results.

![Figure 7: Expected calibration error comparison](../../../outputs/figures/Figure_7_ECE_Comparison.png)

**Figure 7.** Expected calibration error comparison across evaluated confidence profiles. Lower ECE indicates better agreement between predicted confidence and empirical accuracy. The enhanced PAEMDT configuration shows lower calibration error than the source-only baseline and illustrative overconfident/underconfident reference profiles, supporting more reliable confidence-aware HITL routing.

### 4.8 Robustness Under Missing Modalities

The missing-modality robustness analysis evaluates PAEMDT under degraded sensing conditions. The autonomous-operation boundary is set at macro-F1 = 0.85. Conditions below this level require increased caregiver review, and severe degradation can trigger urgent escalation logic. Under single-modality degradation, the system remained relatively stable. More severe multimodal corruption, especially compound noisy conditions and multiple simultaneous sensor failures, produced larger degradation and materially increased escalation frequency.

Table 6 is generated from `outputs/tables/missing_modality_summary.csv`. The all-sensors-noisy condition achieved macro-F1 = 0.760, with a degradation of -0.196 from full input and a HITL escalation rate of 35.0%. These results show that missing-modality masking and fallback logic help contain degradation, but they do not remove the need for human oversight under severe sensing corruption.

![Figure 8: Missing modality robustness](../../../outputs/figures/Figure_8_Missing_Modality_Robustness.png)

**Figure 8.** Missing-modality robustness with escalation-aware interpretation. The left panel reports macro-F1 under degraded sensing conditions, while the right panel reports the corresponding HITL escalation rate. The dashed threshold indicates the autonomous-operation boundary at macro-F1 = 0.85. Severe multimodal degradation increases caregiver-review requirements, confirming the need for safety-aware routing under unreliable sensing conditions.

### 4.9 Privacy-Utility-Latency Trade-off

The privacy-utility-latency analysis evaluates the deployment trade-off among predictive utility, DP-accounted privacy configuration, and edge inference latency. The privacy-enhanced domain-adapted model achieved 95.12% validation accuracy and 62.15% external accuracy while using epsilon = 2.3 and delta = 1e-5. This demonstrates that privacy-aware training can be incorporated with limited source-domain performance reduction while preserving stronger external-domain utility than the source-only baseline.

The edge benchmark adds a practical deployment layer to this result. Raspberry Pi 4 latency of 47.3 ms corresponds to approximately 21 FPS and satisfies the sub-100 ms real-time constraint used for the PAEMDT edge-deployment analysis. The preferred operating point depends jointly on utility, privacy, and latency. In the repository-aligned evaluation, the privacy-enhanced domain-adapted model provides the most balanced privacy-aware deployment candidate.

![Figure 9: Privacy-utility-latency Pareto](../../../outputs/figures/Figure_9_Privacy_Utility_Latency.png)

**Figure 9.** Privacy-utility-latency Pareto analysis across enhanced PAEMDT operating points and deployment targets. The figure compares evaluated configurations in terms of privacy cost, predictive utility, and inference latency. The privacy-enhanced domain-adapted model provides the most balanced operating point by maintaining high validation accuracy, improving external-domain utility, using DP-accounted privacy analysis, and satisfying the real-time edge-inference constraint.

### 4.10 Evidence Maturity

The evidence-maturity dashboard summarizes which PAEMDT modules are implemented, experimentally supported, partially validated, or future translational requirements. Domain adaptation is supported by benchmark evidence, differential privacy is supported by privacy-accounting evidence, and digital-twin synchronization is supported by technical measurement. Physical robot deployment and clinical validation remain future work.

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
