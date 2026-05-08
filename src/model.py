import torch
import torch.nn as nn
import timm

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# Faster R-CNN with DINO backbone
class DinoFasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        backbone = DinoBackbone()
        backbone.out_channels = 768  # required

        anchor_generator = AnchorGenerator(
            sizes=((16, 32, 64, 128),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            min_size=224,
            max_size=224
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)

### Dino model 
class DinoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch16_224.dino",
            pretrained=True
        )

    def forward(self, x):
        # input: x :(1,3,224,224)
        # output: feats: (B, C, H, W)=(1, 768, 14, 14)
        # (1, 197, 768) = (B, N+1, C) => N=196 patches + 1 CLS token
        # C => embedding dimension
        # N= 14x14 = 196, patch resolution= 14
        # the CLS token is a learned vector that attends to all patches  
        feats = self.model.forward_features(x)   # (B, N+1, C)
        #print(f'features:{feats.shape}')
        feats = feats[:, 1:, :]                  # remove CLS
        #print(f'features after removing CLS token:{feats.shape}')

        B, N, C = feats.shape
        H = W = int(N ** 0.5)
        
        # C here is the dimension of embeddings for each patch
        # feats: (B, C, H, W)=(1, 768, 14, 14)
        feats = feats.reshape(B, H, W, C).permute(0, 3, 1, 2)
        #print(f'feat:{feats.shape}')
        return {"0": feats}   # important: dict format
    
if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DinoBackbone().to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"output of Dinov3:")
    print(output["0"].shape)  # should be [1, C, H, W]

    model2 = DinoFasterRCNN(4)
    print(model2)

    