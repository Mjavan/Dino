
import os
import torch
from coco_dataset import CocoDataset, create_dataloaders, collate_fn

# Test Dataloader split
def test_dataloader_split_sizes():

    root_dir = "/sc/projects/sci-lippert/chair/MJ/datasets/coco"
    IMG_DIR = os.path.join(root_dir, "val2017")
    ANN_FILE = os.path.join(root_dir, "annotations/instances_val2017.json")

    dataset_size = 100

    dataset = CocoDataset(
        img_dir=IMG_DIR,
        ann_file=ANN_FILE,
        max_samples=dataset_size,
        train = False
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        img_dir=IMG_DIR,
        ann_file=ANN_FILE,
        dataset=dataset,
        max_samples=dataset_size,
        val_ratio=0.1,
        test_ratio=0.1
    )

    assert len(train_loader.dataset) == 80
    assert len(val_loader.dataset) == 10
    assert len(test_loader.dataset) == 10

# REPRODUCIBITY TEST
def test_split_reproducibility():
    
    root_dir = "/sc/projects/sci-lippert/chair/MJ/datasets/coco"
    IMG_DIR = os.path.join(root_dir, "val2017")
    ANN_FILE = os.path.join(root_dir, "annotations/instances_val2017.json")

    dataset = CocoDataset(img_dir=IMG_DIR,
        ann_file=ANN_FILE,
        max_samples=100,
        train = False)

    train1, _, _ = create_dataloaders(
        img_dir=IMG_DIR,
        ann_file=ANN_FILE, 
        dataset=dataset,
        max_samples=100,
        split_seed=42
    )

    train2, _, _ = create_dataloaders(
        img_dir=IMG_DIR,
        ann_file=ANN_FILE, 
        dataset=dataset,
        max_samples=100,
        split_seed=42
    )

    assert (
        train1.dataset.indices
        ==
        train2.dataset.indices
    )

# BATCH STRUCTURE TEST
def test_batch_structure():
    
    root_dir = "/sc/projects/sci-lippert/chair/MJ/datasets/coco"
    IMG_DIR = os.path.join(root_dir, "val2017")
    ANN_FILE = os.path.join(root_dir, "annotations/instances_val2017.json")

    dataset = CocoDataset(img_dir=IMG_DIR,
        ann_file=ANN_FILE,
        max_samples=100,
        train = False)

    train_loader, _, _ = create_dataloaders(
        img_dir=IMG_DIR,
        ann_file=ANN_FILE,
        dataset=dataset,
        batch_size=4
    )

    images, targets = next(iter(train_loader))

    assert len(images) == 4
    assert len(targets) == 4

    assert isinstance(images[0], torch.Tensor)

    assert "boxes" in targets[0]
    assert "labels" in targets[0]

# COLATTE_FN TEST
def test_collate_fn():
    batch = [
        (
            torch.zeros((3, 224, 224)),
            {"labels": torch.tensor([1])}
        ),
        (
            torch.ones((3, 224, 224)),
            {"labels": torch.tensor([2])}
        )
    ]

    images, targets = collate_fn(batch)

    assert len(images) == 2
    assert len(targets) == 2

    assert isinstance(images, list)
    assert isinstance(targets, list)

# Train/Validation Augmentation Tests
def test_train_val_modes():

    root_dir = "/sc/projects/sci-lippert/chair/MJ/datasets/coco"
    IMG_DIR = os.path.join(root_dir, "val2017")
    ANN_FILE = os.path.join(root_dir, "annotations/instances_val2017.json")

    dataset = CocoDataset(img_dir=IMG_DIR,
        ann_file=ANN_FILE,
        max_samples=100,
        train = False)

    train_loader, val_loader, test_loader = create_dataloaders(IMG_DIR,
                       ANN_FILE, 
                       dataset,
                       max_samples=2000,
                       batch_size=4, 
                       split_seed=42,
                       val_ratio=0.1, 
                       test_ratio=0.1,
                       num_workers=4,
                       )

    assert train_loader.dataset.dataset.train is True
    assert val_loader.dataset.dataset.train is False
    assert test_loader.dataset.dataset.train is False


