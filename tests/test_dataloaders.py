
import os
from coco_dataset import CocoDataset, create_dataloaders

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


