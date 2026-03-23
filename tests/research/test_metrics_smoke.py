from src.evaluation.metrics import classification_summary, expected_calibration_error, latency_summary


def test_classification_summary_smoke() -> None:
    summary = classification_summary(
        y_true=["happy", "sad", "happy", "neutral"],
        y_pred=["happy", "sad", "neutral", "neutral"],
    )
    assert round(summary.accuracy, 2) == 0.75
    assert summary.macro_f1 >= 0.0


def test_expected_calibration_error_smoke() -> None:
    ece = expected_calibration_error(
        confidences=[0.9, 0.8, 0.2, 0.6],
        correctness=[1, 1, 0, 1],
        bins=4,
    )
    assert ece >= 0.0


def test_latency_summary_smoke() -> None:
    summary = latency_summary([10.0, 20.0, 25.0, 30.0])
    assert summary["mean_ms"] == 21.25
    assert summary["max_ms"] == 30.0
