import numpy as np
from src.evaluation.metrics import calculate_metrics, summarize, grouped_summary


def test_perfect_signed_density():
    a = np.zeros((24, 64, 64), dtype=np.float32)
    a[2:5, 8:14, 8:14] = -0.5
    metrics = calculate_metrics(a, a)
    assert metrics["body_iou"] == 1
    assert metrics["signed_body_iou"] == 1
    assert metrics["true_body_sign_accuracy"] == 1
    assert metrics["density_rmse"] == 0


def test_wrong_sign_preserves_support_but_not_polarity():
    a = np.zeros((4, 4, 4))
    a[1:3, 1:3, 1:3] = -0.5
    m = calculate_metrics(a, -a)
    assert m["body_iou"] == 1
    assert m["signed_body_iou"] == 0


def test_grouping():
    row = {"geometry": "compact", "noise_percent": 5, "empty_prediction": False, "body_iou": 0.75}
    assert grouped_summary([row, row])["compact"]["noise_5"]["sample_count"] == 2
