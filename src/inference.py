import torch 
import numpy as np
import pandas as pd
import argparse
import json
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import warnings
warnings.filterwarnings("ignore")

from model import DinoFasterRCNN
from coco_dataset import *
from utils import visualize_inference

parser = argparse.ArgumentParser(description="Run Faster-RCNN for inference!")
parser.add_argument("--exp_name", type=str, required=True, help="Experiment name (e.g. exp_001)")
parser.add_argument("--checkpoint", type=str, default="best", choices=("best","last"), help="Which checkpoint to use")
args = parser.parse_args()

# class for doing inference 
class InferenceEngine:
    def __init__(self, exp_name, checkpoint, device="cuda", conf_thresh=0.5):
        self.device = device
        self.conf_thresh = conf_thresh
        self.exp_name = exp_name
        self.checkpoint = checkpoint
        
        self._load_config()
        print(f"training_config:{self.config}")
        self._load_checkpoint()
        self._load_test_loader()
    
    def _preprocess(self, image):
        # assuming image is already tensor [C,H,W]
        return image.to(self.device)

    def _load_config(self):
        config_path = os.path.join("runs", self.exp_name, "config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)

    # make model and load checkpoint
    def _load_checkpoint(self):
        # make model first
        self.model = DinoFasterRCNN(num_classes=4)
        # load checkpoints
        checkpoint_path = os.path.join("runs", self.exp_name, "checkpoints",f"{self.checkpoint}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    # load test loader 
    def _load_test_loader(self):
        # DATA
        SPLIT_SEED = self.config["split_seed"]
        root_dir = "/sc/projects/sci-lippert/chair/MJ/datasets/coco"
        img_dir = os.path.join(root_dir, "val2017")
        ann_file = os.path.join(root_dir, "annotations/instances_val2017.json")
        dataset = CocoDataset(
        img_dir= img_dir,
        ann_file= ann_file,
        max_samples=self.config["samples"]
        )
        _, _, self.test_loader = create_dataloaders(img_dir = img_dir,
         ann_file = ann_file, 
         dataset = dataset, 
         max_samples = self.config["samples"],
         split_seed=self.config["split_seed"])

    @torch.no_grad()
    def predict(self):
        all_preds = []
        all_targets = []

        for images, targets in self.test_loader:
            images = [self._preprocess(img) for img in images]
            outputs = self.model(images)

            # move everything to cpu
            outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

            all_preds.extend(outputs)   # ← NO filtering here
            all_targets.extend(targets)

        return all_preds, all_targets  

    
    def filter_preds(self, preds, conf_thresh=0.5):
        filtered = []
        for out in preds:
            keep = out["scores"] > conf_thresh
            filtered.append({
                "boxes": out["boxes"][keep],
                "scores": out["scores"][keep],
                "labels": out["labels"][keep],
            })

        return filtered

    def evaluation(self, preds, targets):

        assert len(preds) == len(targets), "the length of preds and targets should be the same"
        metric = MeanAveragePrecision()
        metric.update(preds, targets)
        results = metric.compute()
        
        print("++++++++++++ Printing Results +++++++++++++++")
        print("mAP:", results["map"].item())
        print("mAP@50:", results["map_50"].item())
        print("mAP@75:", results["map_75"].item())
        print("Recall:", results["mar_100"].item())

        if "map_per_class" in results:
            print("mAP per class:", results["map_per_class"].item())
        return results 

    @torch.no_grad()
    def visualize_predictions(self, num_images=5, score_thresh=0.5):
        count = 0

        for images, targets in self.test_loader:
            images_gpu = [img.to(self.device) for img in images]

            outputs = self.model(images_gpu)

            # move to CPU
            outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]

            inf_dir = os.path.join("results", self.exp_name)
            os.makedirs(inf_dir, exist_ok=True)

            for i in range(len(images)):

                save_path = os.path.join(
                inf_dir,
                f"img_{count}.png")

                visualize_inference(
                    images[i],
                    pred=outputs[i],
                    target=targets[i],
                    score_thresh=score_thresh,
                    save_path=save_path)
                    
                count += 1

                print(f"Image {i}")
                #print("Boxes:", outputs[i]["boxes"])
                print("Num boxes:", len(outputs[i]["boxes"]))
                print("Labels:", outputs[i]["labels"])
                print("-----")

                if count >= num_images:
                    return


if __name__=="__main__":
    engine = InferenceEngine(exp_name= args.exp_name, checkpoint=args.checkpoint)
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print(f"Inference for {args.exp_name} using {args.checkpoint} model!")

    preds, targets = engine.predict()

    pred0 = preds[0]
    target0 = targets[0]

    print(pred0["boxes"].shape)
    print(pred0["scores"].shape)
    print(pred0["labels"].shape)

    engine.evaluation(preds, targets)
    
    engine.visualize_predictions(num_images=5, score_thresh=0.1)


    







    

