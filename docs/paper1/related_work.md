# Related Work

## Social and Caregiving Robots
Social and caregiving robots have been studied for companionship, monitoring, prompting, and human-robot interaction support. However, many reported systems emphasize either interaction design or application framing while under-specifying the runtime sensing and synchronization backbone needed for reproducible experiments. This creates a gap between prototype demonstrations and publishable, measurable robotics systems.

## Digital Twins in Robotics
Digital twins have become relevant in robotics for simulation alignment, system monitoring, replay, and fault analysis. In assistive robotics, this idea is especially important because safety-critical behaviors depend on timely and interpretable system state. Yet many robotics digital-twin studies focus on simulation fidelity or industrial monitoring rather than multimodal caregiving sensing and closed-loop human-interaction timing.

## Multimodal Emotion Recognition in HRI
Emotion recognition in human-robot interaction increasingly uses multimodal fusion instead of vision-only pipelines. Classical ML remains relevant as a benchmark when features are carefully engineered, while deep and transformer-style fusion methods provide more expressive integration. Still, reproducible multimodal HRI benchmarks often suffer from fragmented data sources, limited synchronization reporting, or incomplete documentation of missing-modality behavior.

## Research Gap
The key gap is fragmentation. Existing systems often study digital twins, synchronized multimodal sensing, or emotion recognition separately. Paper 1 instead treats them as a single measurable backbone for future cognitive caregiving robotics. The goal is not yet full caregiving intelligence, but a research-grade framework that makes later modules experimentally supportable.
