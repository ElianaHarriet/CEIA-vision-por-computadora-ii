"""Tests for the mask flattener (class-aware multiclass strategy)."""
import numpy as np

from evaluation.mask_flattener import MaskFlattener


def _prediction(masks, confs, classes):
    return {"masks": masks, "confidences": confs, "classes": classes}


def test_empty_masks_return_none():
    flattener = MaskFlattener(strategy="class_aware")
    assert flattener._flatten_single(_prediction([], [], [])) is None


def test_class_aware_paints_semantic_values():
    masks = [np.zeros((5, 5))]
    masks[0][1:3, 1:3] = 1.0
    confs = [0.9]
    classes = [1]
    out = MaskFlattener(strategy="class_aware")._flatten_single(
        _prediction(masks, confs, classes))
    assert set(np.unique(out)) == {0, 2}


def test_class_aware_overlap_high_confidence_wins():
    masks = [np.zeros((5, 5)), np.zeros((5, 5))]
    masks[0][0:2, :] = 1.0
    masks[1][1:3, :] = 1.0
    confs = [0.6, 0.9]
    classes = [0, 2]
    out = MaskFlattener(strategy="class_aware")._flatten_single(
        _prediction(masks, confs, classes))
    # Row 0: class 0 -> 1. Row 1: overlap, class 2 (0.9) wins -> 3.
    # Row 2: class 2 -> 3. Rows 3-4: background -> 0.
    expected = np.unique([
        [1, 1, 1, 1, 1],
        [3, 3, 3, 3, 3],
        [3, 3, 3, 3, 3],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ])
    assert set(np.unique(out)) == set(expected)


def test_custom_class_map():
    masks = [np.zeros((3, 3))]
    masks[0][:, :] = 1.0
    confs = [0.5]
    classes = [2]
    class_map = {0: 10, 1: 11, 2: 12}
    out = MaskFlattener(strategy="class_aware", class_map=class_map)._flatten_single(
        _prediction(masks, confs, classes))
    assert set(np.unique(out)) == {12}