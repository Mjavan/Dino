import torch 
import numpy as np
import pandas as pd
from torchmetrics.detection.mean_ap import MeanAveragePrecision

import warnings
warnings.filterwarnings("ignore")

from model2 import DinoFasterRCNN
from coco_dataset import *
from utils import visualize_inference

# class for doing inference 
class InferenceEngine:
    def __init__(self, device="cuda", conf_thresh=0.5):
        self.device = device
        self.conf_thresh = conf_thresh

        self._load_checkpoint()
        self._load_test_loader()

    def _preprocess(self, image):
        # assuming image is already tensor [C,H,W]
        return image.to(self.device)

    # make model and load checkpoint
    def _load_checkpoint(self):
        # make model first
        self.model = DinoFasterRCNN(num_classes=4)
        # load checkpoints
        checkpoint = torch.load("checkpoints/best.pth", map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    # load test loader 
    def _load_test_loader(self):
        # DATA
        SPLIT_SEED = 42 
        root_dir = "/sc/projects/sci-lippert/chair/MJ/datasets/coco"
        dataset = CocoDataset(
        img_dir=os.path.join(root_dir, "val2017"),
        ann_file=os.path.join(root_dir, "annotations/instances_val2017.json"),
        max_samples=2000
        )
        _, _, self.test_loader = create_dataloaders(dataset, split_seed=SPLIT_SEED)

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

            for i in range(len(images)):
                visualize_inference(
                    images[i],
                    pred=outputs[i],
                    target=targets[i],
                    score_thresh=score_thresh,
                    save_path=f"visualizations/img_{count}.png"
                )
                count += 1

                print(f"Image {i}")
                #print("Boxes:", outputs[i]["boxes"])
                print("Num boxes:", len(outputs[i]["boxes"]))
                print("Labels:", outputs[i]["labels"])
                print("-----")

                if count >= num_images:
                    return


if __name__=="__main__":
    engine = InferenceEngine()
    preds, targets = engine.predict()

    pred0 = preds[0]
    target0 = targets[0]

    print(pred0["boxes"].shape)
    print(pred0["scores"].shape)
    print(pred0["labels"].shape)

    engine.evaluation(preds, targets)
    
    engine.visualize_predictions(num_images=5, score_thresh=0.1)


    







    

