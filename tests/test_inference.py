import torch
import numpy as np
import os
import sys

from inference import InferenceEngine

def test_filter_preds():
    engine = InferenceEngine.__new__(InferenceEngine)

    preds = [
        {
            "boxes": torch.tensor([
                [0.,0.,10.,10.],
                [20.,20.,40.,40.]
            ]),
            "scores": torch.tensor([0.9, 0.2]),
            "labels": torch.tensor([1,2])
        }
    ]

    filtered = engine.filter_preds(
        preds,
        conf_thresh=0.5
    )

    assert len(filtered[0]["boxes"]) == 1
    assert filtered[0]["scores"][0] == 0.9
    assert filtered[0]["labels"][0] == 1

def test_preprocess_shape():
    engine = InferenceEngine.__new__(InferenceEngine)
    
    engine.device = "cuda" if torch.cuda.is_available else "cpu"

    image = torch.randn(3,224,224)

    out = engine._preprocess(image)

    assert out.shape == image.shape
