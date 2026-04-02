# Results

## CS1: ROS2 plus Digital Twin Validation
The CS1 pipeline produced executable outputs in `outputs/csv/cs1/` and `outputs/figures/cs1/`. In the current simulation-backed run:
- M1 mean latency: `19.1193 ms`
- M2 mean latency: `28.5686 ms`
- M3 mean latency: `32.5039 ms`
- M4 mean latency: `40.4978 ms`
- M4 message drop rate: `0.0531`
- M4 task success rate: `0.9459`

These values show that the digital-twin scaffold can represent nominal and disturbed runtime modes while exposing measurable degradation under injected faults.

## CS2: Multimodal Synchronization
The CS2 synchronization study produced aligned session metadata, window indices, modality-availability traces, and synchronization-quality summaries. In the current placeholder run:
- aligned nominal mean alignment error: `52.6608 ms`
- aligned nominal full-modality window rate: `0.5072`
- missing-modality stress mean alignment error: `63.1930 ms`
- missing-modality stress full-modality window rate: `0.4159`

The results are intended as pipeline-readiness indicators rather than claims about a deployed sensing stack.

## CS3: Emotion Benchmark
The real preserved baseline remains the strongest implemented evidence in the repository:
- B0 accuracy: `0.8350`
- B0 macro F1: `0.8155`

The multimodal benchmark families are currently synthetic placeholder experiments:
- B1 classical SVM: accuracy `1.0000`, macro F1 `1.0000`
- B2 deep late fusion: accuracy `0.9417`, macro F1 `0.7929`
- B3 transformer-style fusion: accuracy `0.9583`, macro F1 `0.9647`

These values are reproducible outputs from the current placeholder benchmark and should be interpreted as software-benchmark sanity checks, not as real-world multimodal HRI claims.
