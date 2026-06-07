// PAEMDT Presentation - Academic/IEEE Style
const PptxGenJS = require("/tmp/pptx_work/node_modules/pptxgenjs");

const pres = new PptxGenJS();
pres.layout = "LAYOUT_16x9";
pres.title = "PAEMDT Framework";
pres.author = "M. Asim Amin";

// ── Palette ──────────────────────────────────────────────────────────────
const C = {
  navy:    "1B3A6B",
  steel:   "2E6DA4",
  accent:  "E8A020",
  light:   "EBF2FA",
  white:   "FFFFFF",
  dark:    "1A1A2E",
  muted:   "64748B",
  text:    "1E293B",
  grid:    "CBD5E1",
  green:   "166534",
  red:     "991B1B",
  orange:  "9A3412",
};

function hdr(slide, title) {
  slide.addShape("rect", { x: 0, y: 0, w: 10, h: 0.7, fill: { color: C.navy } });
  slide.addText(title, {
    x: 0.35, y: 0, w: 9, h: 0.7,
    fontSize: 18, bold: true, color: C.white, valign: "middle", margin: 0, fontFace: "Calibri"
  });
  slide.addShape("rect", { x: 0, y: 0.7, w: 10, h: 0.04, fill: { color: C.accent } });
  slide.background = { color: C.light };
}

function note(slide, txt) { slide.addNotes(txt); }

function card(slide, x, y, w, h, fill) {
  slide.addShape("rect", {
    x, y, w, h, fill: { color: fill },
    shadow: { type: "outer", color: "000000", blur: 5, offset: 2, angle: 135, opacity: 0.10 }
  });
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 1 – Title
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.dark };
  s.addShape("rect", { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.accent } });
  s.addShape("rect", { x: 0, y: 0.08, w: 3.2, h: 5.545, fill: { color: C.navy } });
  s.addText("PAEMDT", { x: 0.15, y: 1.0, w: 2.9, h: 0.8, fontSize: 28, bold: true, color: C.accent, fontFace: "Calibri", align: "center" });
  s.addText("Framework", { x: 0.15, y: 1.75, w: 2.9, h: 0.5, fontSize: 14, color: "CADCFC", fontFace: "Calibri", align: "center" });
  s.addShape("oval", { x: 0.4, y: 3.8, w: 0.55, h: 0.55, fill: { color: C.accent, transparency: 40 } });
  s.addShape("oval", { x: 1.1, y: 4.2, w: 0.35, h: 0.35, fill: { color: "CADCFC", transparency: 50 } });
  s.addShape("oval", { x: 1.8, y: 3.6, w: 0.25, h: 0.25, fill: { color: C.white, transparency: 60 } });

  s.addText("PAEMDT: A Privacy-Aware Explainable\nMultimodal Digital-Twin Framework for\nCognitive Caregiving Robots", {
    x: 3.4, y: 0.9, w: 6.3, h: 2.0, fontSize: 20, bold: true, color: C.white, fontFace: "Calibri", align: "left", valign: "top"
  });
  s.addShape("rect", { x: 3.4, y: 2.95, w: 6.2, h: 0.03, fill: { color: C.accent } });
  s.addText("M. Asim Amin", { x: 3.4, y: 3.1, w: 6.2, h: 0.4, fontSize: 13, color: "CADCFC", fontFace: "Calibri" });
  s.addText("Supervisor Review · June 2026", { x: 3.4, y: 3.5, w: 6.2, h: 0.35, fontSize: 12, color: C.muted, fontFace: "Calibri" });
  s.addText("Research Framework Presentation  |  Not a Clinical Product", {
    x: 3.4, y: 4.9, w: 6.2, h: 0.4, fontSize: 10, color: C.muted, fontFace: "Calibri", italic: true
  });
  note(s, "Welcome. This presentation introduces PAEMDT — a research framework integrating multimodal perception, digital twin simulation, knowledge-graph grounded explainability, HITL safety routing, and privacy-aware deployment for cognitive caregiving robots. This is NOT a clinical product; it is a reproducible research and staged validation framework.");
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 2 – Research Motivation
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  hdr(s, "Research Motivation: The Ageing Crisis and Caregiving Gap");

  const stats = [
    { val: "2.1B", lbl: "Adults aged 60+\nby 2050 (WHO)", col: C.navy },
    { val: "~40M", lbl: "Dementia cases\nglobally (2023)", col: C.steel },
    { val: "~$1.3T", lbl: "Global dementia\ncost (USD, 2023)", col: C.orange },
  ];
  stats.forEach((st, i) => {
    const x = 0.4 + i * 3.2;
    card(s, x, 0.9, 2.9, 1.5, st.col);
    s.addText(st.val, { x, y: 0.95, w: 2.9, h: 0.75, fontSize: 36, bold: true, color: C.white, align: "center", fontFace: "Calibri" });
    s.addText(st.lbl, { x, y: 1.65, w: 2.9, h: 0.65, fontSize: 11, color: "CADCFC", align: "center", fontFace: "Calibri" });
  });

  s.addText([
    { text: "Workforce gap: ", options: { bold: true } },
    { text: "Caregiver shortfall projected to reach 13 million globally by 2030.", options: {} },
    { text: "\n", options: {} },
    { text: "Cognitive decline hallmarks: ", options: { bold: true } },
    { text: "Emotional dysregulation, speech impairment, gait anomalies — all observable via sensor data.", options: {} },
    { text: "\n", options: {} },
    { text: "Opportunity: ", options: { bold: true } },
    { text: "Socially Assistive Robots (SARs) can complement human caregivers for monitoring, engagement, and early-warning intervention.", options: {} },
    { text: "\n", options: {} },
    { text: "Challenge: ", options: { bold: true } },
    { text: "Current SARs lack the multimodal perception, trustworthy explainability, and clinical safety mechanisms needed for real-world deployment.", options: {} },
  ], { x: 0.4, y: 2.55, w: 9.2, h: 2.7, fontSize: 13, color: C.text, fontFace: "Calibri", valign: "top" });

  note(s, "Set the scene: the world is ageing rapidly. Dementia and cognitive decline create enormous caregiving demand that human labour alone cannot meet. SARs are a promising complement — but current systems are not ready for deployment because they lack robustness, explainability, and clinical-grade safety mechanisms.");
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 3 – Core Problem in Current SAR Systems
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  hdr(s, "Core Problem: What Current SAR Systems Lack");

  const problems = [
    { head: "Unimodal Perception", body: "Most systems process audio OR video — not both. Cross-modal fusion that mirrors human holistic interpretation is absent." },
    { head: "Black-Box Decisions", body: "Deep-learning classifiers produce opaque outputs. Clinicians cannot audit or trust decisions without interpretable explanations." },
    { head: "No Safety Escalation", body: "Systems either act autonomously or defer all decisions to humans. There is no principled HITL routing tied to model uncertainty." },
    { head: "Privacy Disregard", body: "Sensitive biometric data (face, voice, gait) is rarely anonymised before processing. Regulatory compliance (GDPR, HIPAA) is not addressed." },
    { head: "No Digital Twin", body: "Decisions are made on real-time streams without a simulation layer for safe scenario testing or counterfactual reasoning." },
    { head: "Weak External Validation", body: "Accuracy on training datasets is reported; cross-domain generalisation and calibration under distribution shift are almost never measured." },
  ];

  problems.forEach((p, i) => {
    const col = i % 2 === 0 ? 0.35 : 5.15;
    const y = 0.9 + Math.floor(i / 2) * 1.5;
    card(s, col, y, 4.5, 1.35, C.white);
    s.addShape("rect", { x: col, y, w: 0.07, h: 1.35, fill: { color: C.red } });
    s.addText(p.head, { x: col + 0.15, y: y + 0.08, w: 4.2, h: 0.35, fontSize: 13, bold: true, color: C.navy, fontFace: "Calibri" });
    s.addText(p.body, { x: col + 0.15, y: y + 0.42, w: 4.2, h: 0.85, fontSize: 11, color: C.text, fontFace: "Calibri", valign: "top" });
  });

  note(s, "Detail each failure mode. Emphasise that these are systemic, co-occurring gaps — not isolated bugs. A system that fixes one (e.g., adds an explanation layer) but ignores the others (privacy, calibration, domain shift) is still not deployment-ready.");
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 4 – Research Gap
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  hdr(s, "Research Gap: Fragmented Architectures and Missing Validation");

  card(s, 0.35, 0.9, 4.4, 4.4, C.white);
  s.addShape("rect", { x: 0.35, y: 0.9, w: 4.4, h: 0.38, fill: { color: C.navy } });
  s.addText("Literature Survey Findings", { x: 0.42, y: 0.9, w: 4.25, h: 0.38, fontSize: 13, bold: true, color: C.white, fontFace: "Calibri", valign: "middle" });

  const gaps = [
    "Multimodal fusion studied in isolation — no integration with KGs or LLMs.",
    "Explainability limited to saliency maps; no domain-ontology grounding.",
    "Privacy addressed separately (federated, DP) — never co-designed with perception.",
    "HITL routing ad hoc or absent — no uncertainty-threshold-based escalation.",
    "Validation confined to internal benchmarks; cross-dataset generalisation rarely reported.",
    "No framework unifies all five concerns in a single coherent architecture.",
  ];
  gaps.forEach((g, i) => {
    s.addText([
      { text: `${i + 1}. `, options: { bold: true, color: C.accent } },
      { text: g, options: { color: C.text } }
    ], { x: 0.5, y: 1.38 + i * 0.62, w: 4.1, h: 0.55, fontSize: 11.5, fontFace: "Calibri", valign: "top" });
  });

  card(s, 5.1, 0.9, 4.4, 4.4, C.white);
  s.addShape("rect", { x: 5.1, y: 0.9, w: 4.4, h: 0.38, fill: { color: C.steel } });
  s.addText("PAEMDT Positioning", { x: 5.18, y: 0.9, w: 4.25, h: 0.38, fontSize: 13, bold: true, color: C.white, fontFace: "Calibri", valign: "middle" });

  const positions = [
    { head: "Unified architecture", body: "Five zones co-designed from the ground up." },
    { head: "KG + LLM explanations", body: "Causally grounded, human-readable rationales." },
    { head: "Uncertainty-aware HITL", body: "Safety escalation tied to confidence thresholds." },
    { head: "Privacy by design", body: "k-anonymity and DP embedded in Zone 1." },
    { head: "Staged validation", body: "Internal → cross-domain → clinical roadmap." },
  ];
  positions.forEach((p, i) => {
    s.addShape("oval", { x: 5.18, y: 1.38 + i * 0.75, w: 0.22, h: 0.22, fill: { color: C.accent } });
    s.addText(p.head + ": " + p.body, {
      x: 5.46, y: 1.32 + i * 0.75, w: 3.9, h: 0.55,
      fontSize: 11.5, color: C.text, fontFace: "Calibri", valign: "middle"
    });
  });

  note(s, "Walk through the literature gaps column first, then show how PAEMDT's positioning directly addresses each gap. The key differentiator is co-design: all five concerns addressed together rather than bolted on separately.");
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 5 – Proposed Solution Overview
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  hdr(s, "Proposed Solution: PAEMDT Overview");

  card(s, 0.35, 0.85, 9.3, 1.0, C.navy);
  s.addText("PAEMDT is a reproducible, privacy-aware, explainable multimodal digital-twin research framework for validating cognitive caregiving robot behaviour prior to clinical deployment.", {
    x: 0.5, y: 0.88, w: 9.0, h: 0.9, fontSize: 12.5, color: C.white, fontFace: "Calibri", valign: "middle", italic: true
  });

  const pillars = [
    { num: "01", head: "Multimodal Fusion", body: "Audio + video fused via cross-attention transformer." },
    { num: "02", head: "Digital Twin", body: "Simulated patient state for safe counterfactual testing." },
    { num: "03", head: "KG + LLM XAI", body: "Ontology-grounded, clinician-auditable explanations." },
    { num: "04", head: "HITL Routing", body: "Uncertainty-threshold escalation to human oversight." },
    { num: "05", head: "Privacy Layer", body: "k-anonymity & differential privacy at input." },
  ];

  pillars.forEach((p, i) => {
    const x = 0.3 + i * 1.9;
    card(s, x, 2.0, 1.75, 3.3, C.white);
    s.addShape("rect", { x, y: 2.0, w: 1.75, h: 0.45, fill: { color: C.navy } });
    s.addText(p.num, { x, y: 2.0, w: 1.75, h: 0.45, fontSize: 18, bold: true, color: C.accent, align: "center", fontFace: "Calibri", valign: "middle" });
    s.addText(p.head, { x: x + 0.07, y: 2.5, w: 1.6, h: 0.4, fontSize: 11, bold: true, color: C.navy, fontFace: "Calibri", align: "center" });
    s.addText(p.body, { x: x + 0.07, y: 2.95, w: 1.6, h: 1.25, fontSize: 10.5, color: C.text, fontFace: "Calibri", valign: "top" });
  });

  s.addText("Main contribution: integration — not any single component — is the novelty.", {
    x: 0.35, y: 5.35, w: 9.3, h: 0.25, fontSize: 12, bold: true, color: C.accent, fontFace: "Calibri", align: "center"
  });

  note(s, "Emphasise that PAEMDT is a framework, not a deployed robot. Each pillar exists in prior literature separately; the contribution is their co-design into one coherent, staged-validation architecture.");
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 6 – Five-Zone Architecture
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  hdr(s, "High-Level Framework Architecture: Five Zones");

  const zones = [
    { z: "Zone 1", head: "Privacy-Aware Input", body: "Raw sensor streams → k-anonymity masking → differential privacy noise → anonymised feature tensors.", col: "1B4F72" },
    { z: "Zone 2", head: "Multimodal Perception", body: "Audio CNN + Video ViT encoders → cross-attention fusion → joint emotion/state representation.", col: "1A6B3A" },
    { z: "Zone 3", head: "Digital Twin Engine", body: "Simulated patient & robot state; counterfactual scenario generation; safe protocol testing.", col: "6E2C00" },
    { z: "Zone 4", head: "KG + LLM Explanation", body: "Classifier output → KG ontology lookup → LLM rationale generation → auditable clinical text.", col: "4A235A" },
    { z: "Zone 5", head: "HITL Safety & Action", body: "Composite score + uncertainty → routing: autonomous action | clinician alert | escalation.", col: "7B241C" },
  ];

  zones.forEach((z, i) => {
    const y = 0.85 + i * 0.9;
    card(s, 0.35, y, 1.35, 0.75, z.col);
    s.addText(z.z, { x: 0.35, y, w: 1.35, h: 0.75, fontSize: 11, bold: true, color: C.white, align: "center", valign: "middle", fontFace: "Calibri" });
    card(s, 1.85, y, 7.8, 0.75, C.white);
    s.addShape("rect", { x: 1.85, y, w: 0.06, h: 0.75, fill: { color: z.col } });
    s.addText(z.head + ": ", { x: 2.0, y: y + 0.05, w: 7.5, h: 0.28, fontSize: 12, bold: true, color: z.col, fontFace: "Calibri" });
    s.addText(z.body, { x: 2.0, y: y + 0.36, w: 7.5, h: 0.32, fontSize: 11, color: C.text, fontFace: "Calibri" });
    if (i < 4) {
      s.addShape("rect", { x: 0.815, y: y + 0.75, w: 0.04, h: 0.12, fill: { color: C.accent } });
    }
  });

  s.addText("Digital Twin (Zone 3) provides feedback loops to all other zones", {
    x: 2.0, y: 5.25, w: 7.6, h: 0.25, fontSize: 10, color: C.muted, italic: true, fontFace: "Calibri"
  });

  note(s, "Walk through each zone sequentially: data enters Zone 1 already anonymised; Zone 2 fuses modalities; Zone 3 maintains a simulation mirror; Zone 4 generates explanations; Zone 5 decides whether the robot acts, alerts, or escalates. Emphasise the feedback loop from Zone 3 back to Zones 2 and 5.");
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 7 – Internal Logic
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  hdr(s, "Internal Logic: Fusion · Digital Twin · KG+LLM · HITL · Privacy");

  const items = [
    { head: "Cross-Attention Fusion", body: "Audio encoder Q × Video encoder KV → attention map captures cross-modal dependencies. Learned weights indicate which modality is more informative per sample.", col: C.steel },
    { head: "Digital Twin Engine", body: "Mirrors real patient state Sₜ with simulated state Ŝₜ. Runs counterfactual branches (e.g. 'what if emotion score drops 20%?') without risking the real patient.", col: "166534" },
    { head: "KG-Grounded LLM Explanation", body: "Classifier label → KG traversal (dementia-care ontology) → retrieved causal chain → LLM prompt → human-readable rationale. Each word traceable to ontology nodes.", col: "4A235A" },
    { head: "HITL Routing Logic", body: "If confidence >= θ_high → autonomous action. If θ_low <= confidence < θ_high → alert clinician. If confidence < θ_low → full escalation + suspend action.", col: "6E2C00" },
    { head: "Privacy Layer Mechanics", body: "k-anonymity: features generalised so each sample is indistinguishable from >= k-1 others. DP noise (epsilon-bounded Laplace) added to model gradients. Federated-learning API ready.", col: "7B241C" },
  ];

  items.forEach((it, i) => {
    const y = 0.85 + i * 0.93;
    s.addShape("rect", { x: 0.35, y, w: 0.07, h: 0.78, fill: { color: it.col } });
    s.addText(it.head, { x: 0.52, y: y + 0.02, w: 9.1, h: 0.28, fontSize: 12.5, bold: true, color: it.col, fontFace: "Calibri" });
    s.addText(it.body, { x: 0.52, y: y + 0.34, w: 9.1, h: 0.45, fontSize: 11, color: C.text, fontFace: "Calibri" });
  });

  note(s, "This is the mechanistic slide. Cross-attention: both modalities enter; a weighted representation exits. DT: real sensor data enters; simulation divergence triggers recalibration. KG+LLM: a class label enters; a sentence exits. HITL: a confidence score enters; a routing decision exits. Privacy: raw biometrics enter; anonymised tensors exit.");
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 8 – Mathematical Formulation
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  hdr(s, "Mathematical Formulation Summary");

  const eqs = [
    { name: "Cross-Attention\nFusion", eq: "Attn(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V", desc: "Q from audio encoder; K,V from video encoder. Output is a fused representation weighted by cross-modal relevance." },
    { name: "Robustness\nRatio (RR)", eq: "RR = Acc_internal / Acc_external", desc: "Measures performance degradation under domain shift. RR >> 1 signals overfit to training domain. Ideal RR ≈ 1." },
    { name: "Composite\nScore (CCS)", eq: "CCS = α·E + β·B + γ·C  (α+β+γ=1)", desc: "Weighted sum of Emotion E, Behaviour B, Cognitive C scores. Thresholds on CCS trigger HITL routing." },
    { name: "Expected\nCalibration\nError (ECE)", eq: "ECE = Σ (|Bₘ|/n) · |acc(Bₘ) − conf(Bₘ)|", desc: "Measures alignment between model confidence and actual accuracy across confidence bins. Low ECE = well-calibrated." },
    { name: "HITL\nRouting Rule", eq: "p >= θH → auto  |  θL <= p < θH → alert  |  p < θL → escalate", desc: "p = posterior confidence of top class. θH and θL are tunable thresholds set by clinical protocol." },
  ];

  eqs.forEach((eq, i) => {
    const y = 0.85 + i * 0.93;
    card(s, 0.35, y, 9.3, 0.85, C.white);
    s.addShape("rect", { x: 0.35, y, w: 2.0, h: 0.85, fill: { color: C.navy } });
    s.addText(eq.name, { x: 0.38, y, w: 1.95, h: 0.85, fontSize: 10.5, bold: true, color: C.white, fontFace: "Calibri", align: "center", valign: "middle" });
    s.addText(eq.eq, { x: 2.45, y: y + 0.05, w: 4.0, h: 0.38, fontSize: 11.5, bold: true, color: C.accent, fontFace: "Consolas" });
    s.addText(eq.desc, { x: 2.45, y: y + 0.46, w: 7.0, h: 0.35, fontSize: 10.5, color: C.text, fontFace: "Calibri" });
  });

  note(s, "You don't need to derive these live. Cross-attention: cross-modal weighting. RR: quantifies domain-shift risk (all classifiers show RR > 1.4). CCS: integrates heterogeneous signals into one actionable score. ECE: calibration metric (lower is better). HITL rule: formalises the escalation policy.");
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 9 – Experimental Setup
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  hdr(s, "Experimental Setup: Datasets and Classifiers");

  card(s, 0.35, 0.85, 4.4, 4.5, C.white);
  s.addShape("rect", { x: 0.35, y: 0.85, w: 4.4, h: 0.38, fill: { color: C.navy } });
  s.addText("Datasets", { x: 0.42, y: 0.85, w: 4.25, h: 0.38, fontSize: 13, bold: true, color: C.white, fontFace: "Calibri", valign: "middle" });

  s.addTable([
    [{ text: "Dataset", options: { bold: true, color: C.white, fill: { color: C.steel } } }, { text: "Speakers", options: { bold: true, color: C.white, fill: { color: C.steel } } }, { text: "Samples", options: { bold: true, color: C.white, fill: { color: C.steel } } }, { text: "Role", options: { bold: true, color: C.white, fill: { color: C.steel } } }],
    ["RAVDESS", "24", "7,356", "Internal train/test"],
    ["CREMA-D", "91", "7,442", "External domain shift"],
  ], {
    x: 0.45, y: 1.32, w: 4.1, h: 0.75,
    border: { pt: 0.5, color: C.grid },
    fontFace: "Calibri", fontSize: 10,
    rowH: 0.24,
  });

  s.addText([{ text: "Protocol: ", options: { bold: true } }, { text: "80/20 stratified split on RAVDESS. CREMA-D held out for domain-shift testing only — zero training overlap.", options: {} }], {
    x: 0.45, y: 2.62, w: 4.1, h: 0.62, fontSize: 11, color: C.text, fontFace: "Calibri", valign: "top"
  });
  s.addText([{ text: "Features: ", options: { bold: true } }, { text: "MFCCs (40 coeff.), mel-spectrograms, pitch, ZCR, chroma. z-score normalisation applied.", options: {} }], {
    x: 0.45, y: 3.3, w: 4.1, h: 0.62, fontSize: 11, color: C.text, fontFace: "Calibri", valign: "top"
  });
  s.addText([{ text: "Privacy: ", options: { bold: true } }, { text: "Speaker identity suppressed via k-anonymity (k=5) before feature extraction.", options: {} }], {
    x: 0.45, y: 3.98, w: 4.1, h: 0.55, fontSize: 11, color: C.text, fontFace: "Calibri", valign: "top"
  });

  card(s, 5.1, 0.85, 4.4, 4.5, C.white);
  s.addShape("rect", { x: 5.1, y: 0.85, w: 4.4, h: 0.38, fill: { color: C.steel } });
  s.addText("Seven Classifiers", { x: 5.17, y: 0.85, w: 4.25, h: 0.38, fontSize: 13, bold: true, color: C.white, fontFace: "Calibri", valign: "middle" });

  const clfs = ["SVM (RBF kernel)", "Random Forest (500 trees)", "Gradient Boosting (XGBoost)", "MLP Neural Network", "k-Nearest Neighbours", "Naïve Bayes", "Logistic Regression"];
  clfs.forEach((c, i) => {
    s.addShape("oval", { x: 5.18, y: 1.38 + i * 0.52, w: 0.2, h: 0.2, fill: { color: C.accent } });
    s.addText(c, { x: 5.46, y: 1.33 + i * 0.52, w: 3.9, h: 0.35, fontSize: 12, color: C.text, fontFace: "Calibri", valign: "middle" });
  });
  s.addText("Metrics: Accuracy, F1-macro, ECE, Robustness Ratio, HITL routing rate", {
    x: 5.18, y: 5.05, w: 4.2, h: 0.25, fontSize: 10.5, color: C.muted, italic: true, fontFace: "Calibri"
  });

  note(s, "Key design choice: CREMA-D is held out entirely for external testing. The model never sees CREMA-D during training. Seven classifiers range from simple baselines (NB, LR) through ensemble methods (RF, GB) to a neural model (MLP), plus SVM as the classical strong baseline.");
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 10 – Main Results
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  hdr(s, "Main Results: Internal Accuracy vs External Domain Degradation");

  const clfs    = ["SVM", "RF", "XGBoost", "MLP", "kNN", "NB", "LR"];
  const internal = [87.2, 84.6, 85.9, 82.3, 71.4, 63.8, 76.1];
  const external = [54.1, 58.3, 56.7, 52.9, 44.2, 41.6, 49.3];
  const rr = internal.map((v, i) => parseFloat((v / external[i]).toFixed(2)));

  s.addChart(pres.charts.BAR, [
    { name: "Internal (RAVDESS)", labels: clfs, values: internal },
    { name: "External (CREMA-D)", labels: clfs, values: external },
  ], {
    x: 0.35, y: 0.85, w: 6.2, h: 4.1,
    barDir: "col",
    chartColors: [C.navy, C.accent],
    chartArea: { fill: { color: C.white }, roundedCorners: false },
    catAxisLabelColor: C.text,
    valAxisLabelColor: C.muted,
    valGridLine: { color: C.grid, size: 0.5 },
    catGridLine: { style: "none" },
    showValue: true,
    dataLabelFontSize: 9,
    dataLabelColor: C.text,
    showLegend: true,
    legendPos: "b",
    legendFontSize: 11,
    valAxisMinVal: 0,
    valAxisMaxVal: 100,
    showTitle: true,
    title: "Accuracy (%) — Internal vs External",
    titleFontSize: 12,
    titleColor: C.navy,
  });

  // RR table
  card(s, 6.8, 0.85, 2.85, 4.1, C.white);
  s.addShape("rect", { x: 6.8, y: 0.85, w: 2.85, h: 0.38, fill: { color: C.navy } });
  s.addText("Robustness Ratio", { x: 6.87, y: 0.85, w: 2.7, h: 0.38, fontSize: 11, bold: true, color: C.white, fontFace: "Calibri", valign: "middle", align: "center" });

  const rrRows = [["Classifier", "RR"], ...clfs.map((c, i) => [c, String(rr[i])])];
  rrRows.forEach((row, i) => {
    const y = 1.3 + i * 0.42;
    const bg = i === 0 ? C.steel : i % 2 === 0 ? "F1F5F9" : C.white;
    s.addShape("rect", { x: 6.82, y, w: 2.8, h: 0.38, fill: { color: bg } });
    s.addText(row[0], { x: 6.88, y, w: 1.55, h: 0.38, fontSize: 11, bold: i === 0, color: i === 0 ? C.white : C.text, fontFace: "Calibri", valign: "middle" });
    const rrVal = parseFloat(row[1]);
    const rrCol = i === 0 ? C.white : rrVal > 1.55 ? C.red : C.green;
    s.addText(row[1], { x: 8.22, y, w: 1.35, h: 0.38, fontSize: 11, bold: i === 0, color: rrCol, fontFace: "Calibri", valign: "middle", align: "center" });
  });

  s.addText("RR > 1.5 = significant domain shift (red). Accuracy alone is misleading.", {
    x: 0.35, y: 5.12, w: 9.3, h: 0.25, fontSize: 11, color: C.muted, italic: true, fontFace: "Calibri", align: "center"
  });

  note(s, "Headline finding: SVM achieves 87% internally but only 54% on CREMA-D (RR = 1.61) — a 33-point drop. All RR values exceed 1.4, confirming strong domain shift across all classifiers. Internal accuracy is an optimistic estimate of real-world performance. PAEMDT's staged validation roadmap is motivated precisely by this result.");
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 11 – Ablation, Calibration, Robustness, Privacy
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  hdr(s, "Ablation · Calibration · Robustness · Privacy–Utility–Latency Analysis");

  const panels = [
    {
      title: "Ablation Study",
      col: C.navy,
      lines: [
        "Remove privacy layer: +1.8% acc, but identity leakage risk rises",
        "Remove HITL routing: autonomous errors reach unsafe states",
        "Remove KG explanation: clinician trust score drops (qualitative)",
        "Remove digital twin: no safe scenario testing possible",
        "All five zones are load-bearing",
      ]
    },
    {
      title: "Calibration (ECE)",
      col: C.steel,
      lines: [
        "SVM ECE: 0.09  (best calibrated)",
        "RF ECE: 0.13",
        "XGBoost ECE: 0.12",
        "MLP ECE: 0.18  (overconfident)",
        "Low ECE critical for reliable HITL triggering",
      ]
    },
    {
      title: "Robustness (Gaussian Noise)",
      col: "166534",
      lines: [
        "sigma=0.01: Delta-Acc < 1% (all classifiers)",
        "sigma=0.05: Delta-Acc 2-5% (SVM most stable)",
        "sigma=0.10: Delta-Acc 6-14% (MLP most fragile)",
        "RF, XGBoost: moderate degradation",
        "Ensembles preferred; MLP needs input sanitisation",
      ]
    },
    {
      title: "Privacy–Utility–Latency",
      col: "6E2C00",
      lines: [
        "k-anonymity (k=5): -1.2% accuracy",
        "DP noise (eps=1.0): -2.1% accuracy",
        "Combined stack: -3.0% accuracy",
        "Inference latency overhead: +18 ms/sample",
        "Privacy cost is modest and clinically justifiable",
      ]
    },
  ];

  panels.forEach((p, i) => {
    const x = i % 2 === 0 ? 0.35 : 5.15;
    const y = i < 2 ? 0.85 : 3.2;
    card(s, x, y, 4.5, 2.1, C.white);
    s.addShape("rect", { x, y, w: 4.5, h: 0.33, fill: { color: p.col } });
    s.addText(p.title, { x: x + 0.1, y, w: 4.3, h: 0.33, fontSize: 12, bold: true, color: C.white, fontFace: "Calibri", valign: "middle" });
    p.lines.forEach((ln, j) => {
      const last = j === p.lines.length - 1;
      s.addText(ln, {
        x: x + 0.1, y: y + 0.38 + j * 0.33, w: 4.25, h: 0.3,
        fontSize: 10.5, color: last ? p.col : C.text, bold: last, fontFace: "Calibri"
      });
    });
  });

  note(s, "Four complementary analyses: Ablation confirms all zones contribute. Calibration shows SVM is most trustworthy for HITL thresholding. Robustness reveals MLP is fragile under sensor noise; ensembles preferred. Privacy–Utility–Latency shows 3% accuracy drop and 18ms latency — acceptable for a caregiving monitoring system.");
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 12 – Evidence Maturity & Validation Roadmap
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  hdr(s, "Evidence Maturity and Validation Roadmap");

  const stages = [
    {
      stage: "Stage 1\n(Current)",
      head: "Internal Benchmark",
      status: "COMPLETE",
      scol: "166534",
      items: ["RAVDESS 80/20 split", "7 classifiers evaluated", "Accuracy, ECE, RR metrics", "Ablation study", "Privacy-latency trade-off"],
    },
    {
      stage: "Stage 2\n(Partial)",
      head: "Cross-Domain Validation",
      status: "IN PROGRESS",
      scol: C.orange,
      items: ["CREMA-D external test (done)", "Additional corpora needed", "Demographic diversity check", "Real sensor stream pilot", "Multi-site data agreement"],
    },
    {
      stage: "Stage 3\n(Future)",
      head: "Clinical Deployment",
      status: "PLANNED",
      scol: C.muted,
      items: ["IRB / ethics approval", "Real patient cohort (n >= 30)", "Longitudinal monitoring study", "Clinical outcome linkage", "Regulatory pathway (MDR/FDA)"],
    },
  ];

  stages.forEach((st, i) => {
    const x = 0.35 + i * 3.2;
    card(s, x, 0.85, 2.95, 4.5, C.white);
    s.addShape("rect", { x, y: 0.85, w: 2.95, h: 0.55, fill: { color: C.navy } });
    s.addText(st.stage, { x, y: 0.85, w: 2.95, h: 0.55, fontSize: 12, bold: true, color: C.white, fontFace: "Calibri", align: "center", valign: "middle" });
    s.addText(st.head, { x: x + 0.1, y: 1.48, w: 2.75, h: 0.32, fontSize: 11.5, bold: true, color: C.navy, fontFace: "Calibri" });
    s.addShape("rect", { x: x + 0.1, y: 1.84, w: 2.0, h: 0.26, fill: { color: st.scol } });
    s.addText(st.status, { x: x + 0.1, y: 1.84, w: 2.0, h: 0.26, fontSize: 9.5, bold: true, color: C.white, fontFace: "Calibri", align: "center", valign: "middle" });
    st.items.forEach((it, j) => {
      s.addShape("rect", { x: x + 0.12, y: 2.28 + j * 0.5, w: 0.18, h: 0.18, fill: { color: st.scol } });
      s.addText(it, { x: x + 0.38, y: 2.18 + j * 0.5, w: 2.5, h: 0.45, fontSize: 11, color: C.text, fontFace: "Calibri", valign: "middle" });
    });
  });

  s.addText("PAEMDT is at Stage 1 (complete) and Stage 2 (partial). Stage 3 requires ethics approval, clinical partnerships, and regulatory engagement.", {
    x: 0.35, y: 5.12, w: 9.3, h: 0.35, fontSize: 11, color: C.muted, italic: true, fontFace: "Calibri", align: "center"
  });

  note(s, "Be explicit about where PAEMDT stands: Stage 1 complete; Stage 2 partially done (CREMA-D done, broader datasets and real sensors still needed). Stage 3 does not yet exist. This transparency is a strength — it shows the research is honest about its evidence maturity and has a clear path forward.");
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 13 – Limitations
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  hdr(s, "Limitations: Honest Assessment of Current Constraints");

  const lims = [
    { head: "Single train-test split", body: "80/20 RAVDESS split, not k-fold cross-validated. Reported accuracy may be split-dependent.", sev: "Medium", scol: C.orange },
    { head: "Acted-emotion corpora only", body: "RAVDESS and CREMA-D use professional actors, not real patients with cognitive impairment. Ecological validity unestablished.", sev: "High", scol: C.red },
    { head: "No clinical deployment", body: "Framework has never been tested with real patients in a care setting. Clinical safety properties are unverified in practice.", sev: "High", scol: C.red },
    { head: "Simulated digital twin", body: "DT engine uses synthetic patient models, not real longitudinal patient data. Fidelity to real patient trajectories unknown.", sev: "Medium", scol: C.orange },
    { head: "Privacy guarantees are theoretical", body: "k-anonymity and DP analysed analytically; no adversarial re-identification attack conducted empirically.", sev: "Low-Medium", scol: C.steel },
    { head: "Audio-only experiments", body: "Current experiments use audio features only. Full audio-video fusion pipeline is architecturally specified but not yet experimentally validated.", sev: "High", scol: C.red },
  ];

  lims.forEach((l, i) => {
    const col = i % 2 === 0 ? 0.35 : 5.15;
    const y = 0.88 + Math.floor(i / 2) * 1.47;
    card(s, col, y, 4.5, 1.32, C.white);
    s.addShape("rect", { x: col, y, w: 0.07, h: 1.32, fill: { color: l.scol } });
    s.addText(l.head, { x: col + 0.17, y: y + 0.06, w: 4.1, h: 0.3, fontSize: 12, bold: true, color: C.navy, fontFace: "Calibri" });
    s.addShape("rect", { x: col + 0.17, y: y + 0.38, w: 1.0, h: 0.22, fill: { color: l.scol } });
    s.addText("Risk: " + l.sev, { x: col + 0.17, y: y + 0.38, w: 1.0, h: 0.22, fontSize: 9, bold: true, color: C.white, fontFace: "Calibri", align: "center", valign: "middle" });
    s.addText(l.body, { x: col + 0.17, y: y + 0.64, w: 4.15, h: 0.62, fontSize: 10.5, color: C.text, fontFace: "Calibri", valign: "top" });
  });

  note(s, "Present these limitations openly — they are a sign of research rigour. Most important: full multimodal video fusion experiments are still pending; current results are audio-only. A supervisor will appreciate this candour. For each limitation, a mitigation path exists — mention those briefly if asked.");
}

// ─────────────────────────────────────────────────────────────────────────
// SLIDE 14 – Conclusion & Future Work
// ─────────────────────────────────────────────────────────────────────────
{
  const s = pres.addSlide();
  s.background = { color: C.dark };
  s.addShape("rect", { x: 0, y: 0, w: 10, h: 0.08, fill: { color: C.accent } });

  s.addText("Conclusion & Future Work", {
    x: 0.35, y: 0.15, w: 9.3, h: 0.6, fontSize: 22, bold: true, color: C.white, fontFace: "Calibri"
  });

  card(s, 0.35, 0.85, 4.3, 4.0, "1B2850");
  s.addText("Core Contributions", { x: 0.45, y: 0.9, w: 4.1, h: 0.33, fontSize: 13, bold: true, color: C.accent, fontFace: "Calibri" });
  const contribs = [
    "Five-zone privacy-first architecture: co-designed perception, DT, XAI, HITL, privacy",
    "Domain shift quantified: SVM -33pp, all classifiers RR > 1.4",
    "Calibration (ECE) linked to HITL safety routing thresholds",
    "Privacy-utility-latency trade-off: -3% acc, +18ms for full stack",
    "Staged validation roadmap from benchmark to clinical deployment",
  ];
  contribs.forEach((c, i) => {
    s.addShape("rect", { x: 0.47, y: 1.41 + i * 0.67, w: 0.18, h: 0.18, fill: { color: C.accent } });
    s.addText(c, { x: 0.73, y: 1.3 + i * 0.67, w: 3.85, h: 0.62, fontSize: 11, color: "CADCFC", fontFace: "Calibri", valign: "middle" });
  });

  card(s, 4.95, 0.85, 4.7, 4.0, "1B2850");
  s.addText("Future Work", { x: 5.05, y: 0.9, w: 4.5, h: 0.33, fontSize: 13, bold: true, color: C.accent, fontFace: "Calibri" });
  const future = [
    "Full audio-video cross-attention fusion experiments",
    "k-fold cross-validation to reduce split variance",
    "Real patient pilot study (n >= 10)",
    "Adversarial privacy auditing (re-identification tests)",
    "LLM explanation quality evaluation with clinical raters",
    "Federated multi-site training with hospital partners",
  ];
  future.forEach((f, i) => {
    s.addShape("rect", { x: 5.07, y: 1.41 + i * 0.57, w: 0.25, h: 0.08, fill: { color: C.steel } });
    s.addText(f, { x: 5.38, y: 1.3 + i * 0.57, w: 4.2, h: 0.52, fontSize: 11, color: "CADCFC", fontFace: "Calibri", valign: "middle" });
  });

  card(s, 0.35, 5.0, 9.3, 0.5, C.navy);
  s.addText("PAEMDT demonstrates that trustworthy SAR deployment requires more than accuracy — it demands robustness, calibration, explainability, HITL safety, and privacy, validated in stages.", {
    x: 0.5, y: 5.02, w: 9.0, h: 0.44, fontSize: 11.5, color: C.white, italic: true, fontFace: "Calibri", valign: "middle", align: "center"
  });

  note(s, "Summarise: PAEMDT contributes a unified framework — co-design is the novelty. The domain-shift result is the most important empirical finding. Future work closes three gaps: (1) full multimodal experiments, (2) real patient data, (3) clinical deployment infrastructure. Thank the supervisor and invite questions.");
}
// ─────────────────────────────────────────────────────────────────────────
pres.writeFile({ fileName: '/tmp/PAEMDT_Presentation.pptx' })
  .then(() => {
    console.log('Saved to /tmp');
    require('fs').copyFileSync('/tmp/PAEMDT_Presentation.pptx', '/sessions/sleepy-nifty-faraday/mnt/PAEMDT/PAEMDT_Presentation.pptx');
    console.log('Copied to PAEMDT folder');
  })
  .catch(e => console.error('ERROR:', e));
