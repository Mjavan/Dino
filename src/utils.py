# utils.py
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def patch_to_xy(index, grid_size=14):
    row = index // grid_size
    col = index % grid_size
    return row, col


def patch_to_bbox(row, col, patch_size=16):
    x1 = col * patch_size
    y1 = row * patch_size
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    return [x1, y1, x2, y2]

def decode_predictions(cls, box, threshold=0.5):
    probs = cls.softmax(dim=-1)
    scores, labels = probs.max(dim=-1)

    results = []

    for i in range(cls.shape[1]):
        if scores[0, i] > threshold:
            row, col = patch_to_xy(i)
            base_box = patch_to_bbox(row, col)

            results.append({
                "label": labels[0, i].item(),
                "score": scores[0, i].item(),
                "box": base_box
            })

    return results

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def assign_targets(gt_boxes, gt_labels):
    grid_size = 14
    targets_cls = torch.zeros(196, dtype=torch.long)
    targets_box = torch.zeros(196, 4)

    for i in range(196):
        row = i // grid_size
        col = i % grid_size

        patch_box = [
            col * 16,
            row * 16,
            col * 16 + 16,
            row * 16 + 16
        ]

        best_iou = 0
        best_idx = -1

        for j, gt in enumerate(gt_boxes):
            iou = compute_iou(patch_box, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou > 0.3:
            targets_cls[i] = gt_labels[best_idx]
            targets_box[i] = torch.tensor(gt_boxes[best_idx])

    return targets_cls, targets_box


## function to viusalise image annd target 
CLASS_NAMES = {
    0: "background",
    1: "person",
    2: "car",
    3: "cat"
}

# This funtion is used for checking images and targets before starting training 
def visualize_data(image, target):
    boxes = target["boxes"]
    labels = target["labels"]   # 🔥 ADD THIS

    print(f'image:{image.shape}, boxes:{boxes.shape}')

    img = image.permute(1, 2, 0).cpu().numpy()
    h, w = img.shape[:2]

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box, label in zip(boxes, labels):   # 🔥 FIX TYPO
        xmin, ymin, xmax, ymax = box

        # Convert normalized → pixel coords
        xmin *= w
        xmax *= w
        ymin *= h
        ymax *= h

        # 🔥 Get class name
        class_name = CLASS_NAMES[int(label)]

        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)

        # 🔥 DRAW TEXT (this was missing)
        y_text = max(ymin - 5, 0)
        ax.text(
            xmin,
            y_text,
            class_name,
            fontsize=10,
            color='white',
            bbox=dict(facecolor='red', alpha=0.7, pad=2)
        )

    plt.axis("off")
    plt.show()

# This function is used for debugging predictions during training and validation 
def visualise(image, pred=None, target=None, score_thresh=0.5, save_path=None):
    """
    image: Tensor [C, H, W]
    pred: dict with boxes, scores, labels
    target: dict with boxes, labels
    """
    # convert image to numpy
    img = image.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    ax = plt.gca()

    # =========================
    # 🔴 PREDICTIONS (RED)
    # =========================
    if pred is not None:
        boxes = pred.get("boxes", [])
        scores = pred.get("scores", [])

        for box, score in zip(boxes, scores):
            if score < score_thresh:
                continue

            x1, y1, x2, y2 = box.tolist()

            # skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # it draws predicted bbox with red colour 
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    color="red",
                    linewidth=2
                )
            )

            ax.text(
                x1, y1,
                f"{score:.2f}",
                color="red",
                fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5)
            )

    # =========================
    # 🟢 GROUND TRUTH (GREEN)
    # =========================
    if target is not None:
        boxes = target.get("boxes", [])

        for box in boxes:
            x1, y1, x2, y2 = box.tolist()

            # skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # drawing ground truth bounding boxes  
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    color="green",
                    linewidth=2
                )
            )
    plt.axis("off")

    # =========================
    # 💾 SAVE
    # =========================
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"📸 Saved visualization: {save_path}")
    plt.close()


def visualize_inference(image, 
                        pred=None, 
                        target=None, 
                        score_thresh=0.5, 
                        save_path=None):
                        
    img = image.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    ax = plt.gca()

    # Predictions (red)
    if pred is not None:
        for box, score in zip(pred["boxes"], pred["scores"]):
            if score < score_thresh:
                continue

            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, color="red", linewidth=2
            ))
            ax.text(x1, y1, f"{score:.2f}", color="red")

    # Ground truth (green)
    if target is not None:
        for box in target["boxes"]:
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, color="green", linewidth=2
            ))

    plt.axis("off")

    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    else:
        plt.savefig("debug.png")

    plt.close()


def plot_losses(loss_dict, save_path=None, show=True):
    """
    Plots training and validation loss curves.

    Args:
        loss_dict (dict): {'train': [...], 'val': [...]}
        save_path (str, optional): path to save the figure (e.g. 'plots/loss.png')
        show (bool): whether to display the plot
    """

    train_losses = loss_dict.get("train_epochs", [])
    val_losses = loss_dict.get("val_epochs", [])

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_losses, label="Train Loss", marker='o')
    if val_losses:
        plt.plot(epochs, val_losses, label="Validation Loss", marker='s')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)

    # Save if path provided
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📁 Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()



