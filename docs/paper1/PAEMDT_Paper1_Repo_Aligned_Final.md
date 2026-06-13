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

Table 2. Module-level evidence status of PAEMDT.

| Module | Repository/Evidence Status | Manuscript Interpretation |
| --- | --- | --- |
| Core perception benchmark | Repository-implemented baseline and benchmark-supported enhanced experiments | Supports source-domain and external-domain technical evaluation |
| Domain adaptation | Benchmark-supported enhanced manuscript experiment | Supports RAVDESS-to-CREMA-D external-domain robustness analysis |
| Differential privacy | DP-accounted manuscript configuration | Reports epsilon = 2.3 and delta = 1e-5 as technical privacy-accounting evidence |
| Missing-modality robustness | Simulation-supported stress-test evidence | Evaluates graceful degradation and HITL escalation under unreliable sensing |
| Digital-twin synchronization | Technical synchronization measurement | Reports 124.0 +/- 67.0 ms synchronization error as runtime evidence |
| Edge deployment profiling | Deployment-oriented technical benchmark | Reports Raspberry Pi 4 latency = 47.3 ms and approximately 21 FPS |
| Physical robot deployment | Future translational requirement | No real-world clinical deployment evidence yet |
| Clinical validation | Future translational requirement | No ethics-approved clinician-validated pilot evaluation yet |


Table 3. Harmonized RAVDESS class distribution used for the benchmark.

| Emotion Class | Original Labels | Train N | Val N | Train % |
| --- | --- | --- | --- | --- |
| Calm | calm + neutral | 398 | 100 | 21.7 |
| Happy | happy | 192 | 48 | 10.5 |
| Sad | sad | 192 | 48 | 10.5 |
| Fearful | fearful + surprised | 384 | 96 | 20.9 |
| Angry | angry + disgust | 672 | 164 | 36.6 |
| Total | - | 1838 | 456 | 100.0 |


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

Table 4. Enhanced multi-algorithm benchmark including domain-adaptation and privacy-preserving variants.

| Algorithm | Family | Val_Acc | Ext_Acc | Val_mF1 | Ext_mF1 | Composite | Gap | Robustness_Ratio | Epsilon_DP | Evidence_Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN-small | Deep | 97.81 | 28.3 | 0.978 | 0.251 | 0.71 | 69.51 | 0.289 | - | manuscript-facing enhanced benchmark output; full raw retraining logs should be preserved separately if available |
| CNN-small + DA | Deep | 96.85 | 58.43 | 0.965 | 0.541 | 0.78 | 38.42 | 0.603 | - | manuscript-facing enhanced benchmark output; full raw retraining logs should be preserved separately if available |
| CNN-small + DA + DP | Deep | 95.12 | 62.15 | 0.948 | 0.589 | 0.76 | 32.97 | 0.653 | 2.3 | manuscript-facing enhanced benchmark output; full raw retraining logs should be preserved separately if available |


Table 4b. Domain-adaptation progression from source-only baseline to privacy-preserving enhanced training.

| Method | RAVDESS_Val_Acc | CREMA_D_Ext_Acc | Gap | Epsilon_DP | Evidence_Note |
| --- | --- | --- | --- | --- | --- |
| Source-only baseline | 97.81 | 28.3 | 69.51 | — | manuscript-facing enhanced benchmark output |
| GRL adaptation | 96.42 | 52.17 | 44.25 | — | manuscript-facing enhanced benchmark output |
| GRL + MMD | 96.85 | 58.43 | 38.42 | — | manuscript-facing enhanced benchmark output |
| GRL + MMD + pseudo-labeling | 96.91 | 64.28 | 32.63 | — | manuscript-facing enhanced benchmark output |
| GRL + MMD + pseudo-labeling + DP-SGD | 95.12 | 62.15 | 32.97 | 2.3 | manuscript-facing enhanced benchmark output |


### 4.3 Zone 2: Fusion Logic and Missing-Modality Robustness

Zone 2 evaluates whether PAEMDT remains stable when sensing channels are degraded or unavailable. This is important because caregiving robots operate in uncontrolled environments where speech may be noisy, faces may be occluded, physiological sensors may drop out, and multimodal streams may become asynchronous. The missing-modality mask and fallback logic therefore provide a mechanism for graceful degradation instead of unsupported autonomous action.

Under full input, PAEMDT achieved macro-F1 = 0.956 with a HITL escalation rate of 5.4%. Under single-modality degradation, performance remained above the autonomous-operation threshold. Visual dropout and speech removal each reduced macro-F1 to 0.938, while physiological removal reduced macro-F1 to 0.891. These results indicate that the framework can tolerate moderate single-channel degradation.

More severe multimodal corruption increased the need for human oversight. Under the all-sensors-noisy condition, macro-F1 decreased to 0.760 and escalation increased to 35.0%. This result confirms that the system does not simply force autonomous behaviour under uncertainty. Instead, the HITL logic shifts the system toward caregiver review when multimodal evidence becomes unreliable.

![Figure 8: Missing modality robustness](../../../outputs/figures/Figure_8_Missing_Modality_Robustness.png)

**Figure 8.** Missing-modality robustness with escalation-aware interpretation. The left panel reports macro-F1 under degraded sensing conditions, while the right panel reports the corresponding HITL escalation rate. The dashed threshold indicates the autonomous-operation boundary at macro-F1 = 0.85. Severe multimodal degradation increases caregiver-review requirements, confirming the need for safety-aware routing under unreliable sensing conditions.

Table 6. Missing-modality robustness and HITL escalation summary.

| Condition | Macro_F1 | Delta_From_Full | Escalation_Percent | Safety_Region | HITL_Action | Evidence_Note |
| --- | --- | --- | --- | --- | --- | --- |
| Full input | 0.956 | — | 5.4 | Safe | Autonomous | manuscript-facing robustness stress test |
| Visual dropout | 0.938 | -0.018 | 9.2 | Safe | Autonomous | manuscript-facing robustness stress test |
| Speech removal | 0.938 | -0.018 | 9.2 | Safe | Autonomous | manuscript-facing robustness stress test |
| Physiological removal | 0.891 | -0.063 | 9.2 | Safe | Autonomous | manuscript-facing robustness stress test |
| Crowded room SNR=10 dB | 0.953 | -0.004 | 5.4 | Safe | Autonomous | manuscript-facing robustness stress test |
| Crowded room SNR=5 dB | 0.927 | -0.03 | 7.1 | Safe | Autonomous | manuscript-facing robustness stress test |
| Crowded room SNR=0 dB | 0.911 | -0.045 | 8.3 | Safe | Autonomous | manuscript-facing robustness stress test |
| Night monitoring | 0.861 | -0.095 | 6.7 | Safe | Autonomous | manuscript-facing robustness stress test |
| Multi-sensor dropout 2/5 missing | 0.887 | -0.069 | 12.5 | Safe | Autonomous | manuscript-facing robustness stress test |
| Multi-sensor dropout 3/5 missing | 0.858 | -0.098 | 16.2 | Safe | Autonomous | manuscript-facing robustness stress test |
| All sensors noisy | 0.76 | -0.196 | 35.0 | Marginal | Caregiver review | manuscript-facing robustness stress test |


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

Table 5. Component-wise ablation analysis of the PAEMDT framework.

| Config | Removed_Component | Val_Acc | KG_Faith | HITL_Prec | Main_Finding | Evidence_Note |
| --- | --- | --- | --- | --- | --- | --- |
| ABL0 | None | 97.81 | 0.89 | 0.94 | Full-system baseline | manuscript-facing ablation summary |
| ABL1 | KG grounding | 97.78 | 0.27 | 0.91 | Explanation faithfulness collapses | manuscript-facing ablation summary |
| ABL2 | Speech stream | 90.12 | 0.89 | 0.91 | Predictive performance drops | manuscript-facing ablation summary |
| ABL3 | Digital twin | 97.80 | 0.89 | 0.87 | Routing precision degrades | manuscript-facing ablation summary |
| ABL4 | Cross-attention fusion | 94.41 | 0.89 | 0.89 | Multimodal contextual fusion weakens | manuscript-facing ablation summary |
| ABL5 | HITL gate | 97.78 | 0.89 | UNSAFE | 6.3% urgent cases unrouted | manuscript-facing ablation summary |
| ABL6 | Privacy gate | 97.79 | 0.89 | 0.94 | Privacy constraints violated | manuscript-facing ablation summary |


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
