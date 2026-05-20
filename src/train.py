import torch
from torch.utils.data import DataLoader, random_split
import os
import argparse 
import json
import wandb

from model import DinoFasterRCNN
from coco_dataset import *
from utils import *

## Fixing Random Seeds 
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    # for reproducibility (slower but deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# ===== TRAINER CLASS =====
class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, 
        checkpoint_dir="checkpoints", vis_dir=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device

        self.best_val_loss = float("inf")
        # Here we use first batch of val loader for debugging model 
        self.fixed_images, self.fixed_targets = next(iter(self.val_loader))
        # Making directories to save checkpoints and visualisations 
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        self.save_dir = checkpoint_dir
        self.vis_dir = vis_dir

    def _move_to_device(self, images, targets):
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return images, targets

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for images, targets in self.train_loader:
            images, targets = self._move_to_device(images, targets)

            loss_dict = self.model(images, targets)
            loss = sum(loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # detailed logging
            #print(
            #    f"[Train] total: {loss.item():.4f} | "
            #    f"cls: {loss_dict['loss_classifier'].item():.4f} | "
            #    f"box: {loss_dict['loss_box_reg'].item():.4f} | "
            #    f"obj: {loss_dict['loss_objectness'].item():.4f} | "
            #    f"rpn: {loss_dict['loss_rpn_box_reg'].item():.4f}"
            #)

        return total_loss / len(self.train_loader)

    def validate(self):
        # ⚠️ keep train mode for Faster-RCNN losses
        # Faster R-CNN requires train mode to compute losses
        self.model.train()
        total_loss = 0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images, targets = self._move_to_device(images, targets)

                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values())

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss
        }

        # ✅ always overwrite last checkpoint (for resume)
        torch.save(checkpoint, os.path.join(self.save_dir, "last.pth"))
        print(f"💾 Saved LAST checkpoint (epoch {epoch})")

        # ✅ save best model separately
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(self.save_dir, "best.pth"))
            print("✅ Saved BEST model")

    def train(self, epochs, plot_dir=None, score_thresh=0.3):
        loss_dict = {'train_epochs':[], 'val_epochs':[]}

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate()

            loss_dict['train_epochs'].append(train_loss)
            loss_dict['val_epochs'].append(val_loss)

            print("\n====================================")
            print(f"[Epoch {epoch+1}/{epochs}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            print("====================================\n")

            if (epoch + 1) % 5 == 0:
                print("📸 Visualizing training predictions...")
                self.visualize_training(epoch, num_images=3, score_thresh=score_thresh)

            self.save_checkpoint(epoch, val_loss)

        plot_losses(loss_dict, 
        save_path=os.path.join(plot_dir,"loss_curve.png"), 
        show=False)
        

    @torch.no_grad()
    def visualize_training(self, epoch, num_images=3, score_thresh=0.3):
        self.model.eval()

        images = [img.to(self.device) for img in self.fixed_images]
        outputs = self.model(images)

        outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]

        for i in range(num_images):
            visualise(
                self.fixed_images[i],
                pred=outputs[i],
                target=self.fixed_targets[i],
                score_thresh=score_thresh,
                save_path=os.path.join(
                    self.vis_dir,
                    f"vis_epoch_{epoch}_img_{i}.png"
                )
            )

parser = argparse.ArgumentParser(description="Run Faster-RCNN for training!") 

parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (e.g. exp_001)")

parser.add_argument("--epochs", type=int, default=7, help="Number of epochs for training")

parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training")

parser.add_argument("--samples", type=int, default=2000, help="Number of samples for training and inference")

parser.add_argument("--train_seed", type=int, default=42, help="Seed for reproducibility of training")

parser.add_argument("--split_seed", type=int, default=42, help="Seed for reproducibility of spliting data")

parser.add_argument("--freeze_backbone", type=bool, default=False, help="True, to freeze backbobe, false otherwise.")

parser.add_argument("--score_thresh", type=float, default=0.3, help="Score to filter bboexs")

args = parser.parse_args()

# ===== MAIN =====
def main(args):
    
    # Fix random seed 
    TRAIN_SEED = args.train_seed
    SPLIT_SEED = args.split_seed

    # FIRST THING
    set_seed(TRAIN_SEED)

    # GENERATE EXPERIMENT NUMBER 
    def get_next_exp_name(base_dir="runs"):
        os.makedirs(base_dir, exist_ok=True)

        existing = [d for d in os.listdir(base_dir) if d.startswith("exp_")]
        if not existing:
            return "exp_001"

        nums = [int(d.split("_")[1]) for d in existing]
        return f"exp_{max(nums)+1:03d}"

    # CREATE EXPERIMENT DIRECTORY
    if args.exp_name is None:
        exp_name = get_next_exp_name()
    else:
        exp_name = args.exp_name

    exp_dir = os.path.join("runs", exp_name)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    plot_dir = os.path.join(exp_dir, "plots")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    # SAVE CONFIG
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"Running experiment: {exp_name}")
    print(f"Saving to: {exp_dir}")

    # CONFIG
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 4
    epochs = args.epochs
    lr = args.lr

    # MODEL
    model = DinoFasterRCNN(num_classes=num_classes).to(device)

    # Freezing backbobe or not
    if args.freeze_backbone:
        # freeze backbone first 
        print('Freeze backbone model')
        for p in model.model.backbone.parameters():
            p.requires_grad = False

    # DATA
    root_dir = "/sc/projects/sci-lippert/chair/MJ/datasets/coco"
    IMG_DIR = os.path.join(root_dir, "val2017")
    ANN_FILE = os.path.join(root_dir, "annotations/instances_val2017.json")

    base_dataset = CocoDataset(
        img_dir=IMG_DIR,
        ann_file=ANN_FILE,
        max_samples=args.samples,
        train = False
    )

    # dataloaders 
    train_loader, val_loader, test_loader = create_dataloaders(IMG_DIR,
     ANN_FILE,
     base_dataset, 
     args.samples, 
     split_seed=SPLIT_SEED)
    

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # TRAIN
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        checkpoint_dir= checkpoint_dir,
        vis_dir=os.path.join(exp_dir, "visualizations")
    )

    trainer.train(epochs, plot_dir, args.score_thresh)


if __name__ == "__main__":
    main(args)