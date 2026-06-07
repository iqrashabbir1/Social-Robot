# Manuscript Patches for PAEMDT Paper 1

These patches provide Word-ready replacement text for the PAEMDT manuscript. The equations should be pasted into Word using MathType or Word Equation Editor, not as plain text.

## Abstract Boundary Sentence

The reported results constitute technical and experimental validation of the PAEMDT software framework and benchmark pipeline; prospective clinical validation, ethics-approved field deployment, and clinician-validated assisted-living trials remain future work.

## End of Introduction Organization Paragraph

Section 2 presents the PAEMDT architecture. Section 3 provides the mathematical formulation aligned with the five-zone framework. Section 4 presents the case-study evaluation. Section 5 discusses implications, limitations, and translational requirements. Section 6 concludes the paper.

## Section 3 Mathematical Formulation

This section formalizes the PAEMDT information flow according to the five-zone architecture. The formulation is modular and evidence-aware so that repository-implemented, benchmark-supported, simulation-supported, prototype, and future clinical-validation components remain clearly distinguished.

### Zone 1: Multimodal Perception

Raw multimodal observation:

$$
X_t = \\{x_t^v, x_t^a, x_t^p, x_t^m, x_t^l\\}
$$

Modality encoder:

$$
y_t^i = \\phi_i(x_t^i)
$$

Embedding set:

$$
Y_t = \\{y_t^v, y_t^a, y_t^p, y_t^m, y_t^l\\}
$$

### Zone 2: Missing-Modality Masking and Fusion

Modality mask:

$$
M_t = \\{m_t^v, m_t^a, m_t^p, m_t^m, m_t^l\\}
$$

Fusion:

$$
F_t = \\mathcal{F}(Y_t, M_t)
$$

Attention-style fusion:

$$
Q_t = W_QY_t, \\quad K_t = W_KY_t, \\quad V_t = W_VY_t
$$

$$
F_t = \\operatorname{softmax}\\left(\\frac{Q_tK_t^T}{\\sqrt{d_k}}\\right)V_t
$$

Masked fused representation:

$$
F_t^* = F_t \\odot M_t
$$

### Zone 3: Digital-Twin Synchronization

Digital-twin state:

$$
S_t^{DT} = \\{s_t^{pat}, s_t^{rob}, s_t^{env}, s_t^{int}\\}
$$

Digital-twin update:

$$
S_t^{DT} = U_{DT}(S_{t-1}^{DT}, F_t^*, u_{t-1})
$$

Synchronization error:

$$
\\epsilon_t^{DT} = \\left|t_{now} - \\max(t_t^v, t_t^a, t_t^p, t_t^m, t_t^l)\\right|
$$

### Zone 4: Explainability, Privacy-Aware Inference, and HITL Reasoning

Knowledge graph:

$$
G = (V, E)
$$

Evidence retrieval:

$$
E_t = R(G, F_t^*, S_t^{DT})
$$

Explanation:

$$
e_t = G_{exp}(F_t^*, S_t^{DT}, E_t)
$$

Reasoning vector:

$$
z_t = [F_t^*, S_t^{DT}, \\epsilon_t^{DT}, E_t]
$$

Risk score:

$$
hr_t = \\sigma(w_r^Tz_t + b_r)
$$

Anomaly score:

$$
a_t = D(z_t, z_{ref})
$$

Privacy transform:

$$
\\widetilde{X}_t = \\Pi(X_t; \\lambda)
$$

HITL tier:

$$
T_t =
\\begin{cases}
\\text{urgent escalation}, & hr_t > \\tau_r^{high} \\ \\text{or}\\ a_t > \\tau_a^{high} \\\\
\\text{caregiver review}, & hr_t > \\tau_r^{mid} \\ \\text{or}\\ a_t > \\tau_a^{mid} \\ \\text{or}\\ \\epsilon_t^{DT} > \\tau_{sync} \\\\
\\text{autonomous response}, & \\text{otherwise}
\\end{cases}
$$

### Zone 5: Tiered Caregiving Action and Model Selection

Action:

$$
u_t = \\pi(F_t^*, S_t^{DT}, e_t, hr_t, a_t, \\widetilde{X}_t, T_t)
$$

Composite score:

$$
Score_k = \\alpha F1_k + \\beta R_k + \\gamma C_k - \\delta L_k - \\eta P_k
$$

Robustness:

$$
R_k = \\frac{Acc_k^{ext}}{Acc_k^{val}}
$$

Expected calibration error:

$$
ECE = \\sum_m \\frac{|B_m|}{n}\\left|acc(B_m) - conf(B_m)\\right|
$$

Training objective:

$$
\\mathcal{L}_{total} = \\mathcal{L}_{cls}^s + \\lambda_{adv}\\mathcal{L}_{GRL} + \\lambda_{mmd}MMD^2(Z_s, Z_t) + \\lambda_{pl}\\mathcal{L}_{PL}
$$

DP-SGD:

$$
\\widetilde{g} = \\frac{1}{B}\\left(\\sum_i \\operatorname{clip}(g_i, C) + \\mathcal{N}(0, \\sigma_{DP}^2C^2I)\\right)
$$

## Section 4.2 Dataset and Module Evidence Levels

The experimental evidence in this study is organized according to dataset evidence and module evidence. RAVDESS is used for internal development and held-out validation, whereas CREMA-D is reserved for external-domain evaluation. This separation prevents optimistic reuse of the same dataset for model development and generalization claims. The reported results should therefore be interpreted as dataset-based technical and experimental validation rather than clinical validation.

At the module level, PAEMDT distinguishes implemented components, benchmark-supported enhanced experiments, simulation-supported components, prototype modules, and future translational requirements. Core benchmarking, domain adaptation, differential privacy, calibration analysis, missing-modality testing, digital-twin replay, and edge profiling provide technical evidence. Physical robot deployment, prospective human-subject testing, and clinical validation remain future work.

## Discussion Subsection: Repository Evidence Boundary

The repository evidence boundary is intentionally conservative. The preserved four-class baseline is the implemented repository baseline, whereas the enhanced RAVDESS/CREMA-D experiments are manuscript-facing benchmark-supported domain-adaptation outputs. Simulation-supported physiology, medication, privacy, HITL, and digital-twin modules should not be interpreted as real assisted-living deployment evidence. The current contribution is a reproducible technical and experimental framework that prepares PAEMDT for future clinical-validation studies.

## Conclusion Limitation Sentence

Although PAEMDT now provides traceable benchmark, privacy, robustness, calibration, and deployment-oriented artifacts, prospective clinical validation, ethics approval, live wearable integration, and clinician-validated assisted-living pilots remain future work.
