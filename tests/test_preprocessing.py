import numpy as np
import os
import sys 
from PIL import Image 
import torch


from coco_dataset import base_transform, CocoDataset, scale_boxes, horizontal_flip_boxes


def test_preprocessing_output_shape():
    fake_image = np.random.randint(0,255,(100,150,3),dtype=np.uint8)
    pil_image = Image.fromarray(fake_image)
    transformed = base_transform(pil_image)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (3, 224, 224)
    assert transformed.dtype == torch.float32

def test_preprocessing_value_range():
    fake_image = np.random.randint(0,255,(100, 100, 3),dtype=np.uint8)
    pil_image = Image.fromarray(fake_image)
    transformed = base_transform(pil_image)
    assert transformed.min() >= 0.0
    assert transformed.max() <= 1.0


# Test output structure without augmentations 
def test_dataset_output_structure():
    root_dir = "/sc/projects/sci-lippert/chair/MJ/datasets/coco"

    dataset = CocoDataset(
        img_dir=os.path.join(root_dir, "val2017"),
        ann_file=os.path.join(
            root_dir,
            "annotations/instances_val2017.json"
        ),
        max_samples=1,
        train=False
    )

    image, target = dataset[0]

    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert "boxes" in target
    assert "labels" in target
    assert target["boxes"].dtype == torch.float32
    assert target["labels"].dtype == torch.long

# Test bounding box scaling
def test_scale_boxes():
    boxes = torch.tensor([
        [10., 20., 30., 40.]
    ])

    scaled = scale_boxes(
        boxes.clone(),
        orig_w=100,
        orig_h=100
    )

    expected = torch.tensor([
        [22.4, 44.8, 67.2, 89.6]
    ])

    assert torch.allclose(
        scaled,
        expected
    )

# Testing horizontal flip 
def test_horizontal_flip_boxes():
    boxes = torch.tensor([
        [10., 20., 50., 60.]
    ])

    flipped = horizontal_flip_boxes(
        boxes,
        width=224
    )

    expected = torch.tensor([
        [174., 20., 214., 60.]
    ])

    assert torch.equal(
        flipped,
        expected
    )




