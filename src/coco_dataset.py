import torch
from pycocotools.coco import COCO
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader
import numpy as np, random, torch
from torch.utils.data import Subset

# ===== CONFIG =====
VALID_CLASSES = [1, 3, 17]  # person, car, cat

# Map COCO → your labels (start from 1, 0 = background)
CLASS_MAP = {c: i + 1 for i, c in enumerate(VALID_CLASSES)}

# ===== TRANSFORM =====
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ===== COLLATE FN =====
def collate_fn(batch):
    images, targets = [], []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return images, targets

# ===== SCALE BBOUNDING BOXES ========
def scale_boxes(boxes, orig_w, orig_h,
                target_w=224,
                target_h=224):

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    return boxes

# ===== TEST HORIZONTAL FLIP =====
def horizontal_flip_boxes(boxes, width=224):
    flipped = boxes.clone()

    flipped[:, [0, 2]] = width - boxes[:, [2, 0]]

    return flipped



class CocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, max_samples=None, train=True):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        ids = sorted(list(self.coco.imgs.keys()))
        # if we want to use a part of dataset for training 
        if max_samples is not None: 
            ids = ids[:max_samples]
        self.ids = ids 
        self.train = train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(
            f"Failed to load image: {img_path}"
            ) from e


        orig_w, orig_h = image.size

        boxes = []
        labels = []

        # ===== FILTER + REMAP =====
        for ann in anns:
            cat = ann["category_id"]

            if cat not in VALID_CLASSES:
                continue

            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(CLASS_MAP[cat])

        # ===== HANDLE EMPTY =====
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]  # background

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # ===== RESIZE IMAGE =====
        image = base_transform(image)

        # ===== SCALE BOXES =====
        boxes = scale_boxes(boxes, orig_w, orig_h,
                target_w=224,
                target_h=224)

        # ===== AUGMENTATION FOR TRAINING =====
        if self.train:
            if random.random() < 0.5:
                image = transforms.functional.hflip(image)
                boxes = horizontal_flip_boxes(boxes)

        return image, {"boxes": boxes, "labels": labels}


def create_dataloaders(img_dir,
                       ann_file, 
                       dataset,
                       max_samples=2000,
                       batch_size=4, 
                       split_seed=42,
                       val_ratio=0.1, 
                       test_ratio=0.1,
                       num_workers=4,
                       ):

    assert 0 < val_ratio < 1, "val_ratio must be in range (0, 1)"
    assert 0 < val_ratio + test_ratio < 1, "val_ratio + test_ratio must be < 1" 
    
    # for reproducibility 
    g = torch.Generator().manual_seed(split_seed)
   
    total_size = len(dataset)
    if total_size == 0:
        raise ValueError("Dataset is empty")
    val_size = int(val_ratio * total_size)
    test_size = int(test_ratio * total_size)
    train_size = total_size - val_size - test_size

    train_idx, val_idx, test_idx = random_split(
        range(total_size),
        [train_size, val_size, test_size],
        generator=g
    )

    # check if splits reamin fixed accross runs 
    # print("Train indices (first 10):", train_ds.indices[:10])
    # print("Val indices (first 10):", val_ds.indices[:10])
    # print("Test indices (first 10):", test_ds.indices[:10])

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # 🔥 create THREE datasets with different behavior
    train_ds = Subset(
        CocoDataset(img_dir, ann_file, max_samples, train=True),
        train_idx.indices
    )

    val_ds = Subset(
        CocoDataset(img_dir, ann_file, max_samples, train=False),
        val_idx.indices
    )

    test_ds = Subset(
        CocoDataset(img_dir, ann_file, max_samples, train=False),
        test_idx.indices
    )

    # 🔥 create data loaders 
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn
        )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn
        )

    return train_loader, val_loader, test_loader


# ===== TEST =====
if __name__ == "__main__":
    root_dir = "/sc/projects/sci-lippert/chair/MJ/datasets/coco"

    dataset = CocoDataset(
        img_dir=os.path.join(root_dir, "val2017"),
        ann_file=os.path.join(root_dir, "annotations/instances_val2017.json")
    )

    print("Dataset size:", len(dataset))

    image, target = dataset[0]

    #print("Image shape:", image.shape)
    #print("Target:", target)

    train_loader, val_loader, test_loader = create_dataloaders(dataset)

    # print(f"len trainloder:{len(train_loader)}")
    # print(f"len valloader:{len(val_loader)}")
    # print(f"len testloader:{len(test_loader)}")
    
    # print("###################################")
    # print(f"len dataset splits:")
    # print(f"len trainset:{len(train_loader.dataset)}")
    # print(f"len valset:{len(val_loader.dataset)}")
    # print(f"len testset:{len(test_loader.dataset)}")