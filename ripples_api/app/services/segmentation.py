import torch
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
import os
from torch import nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

# The SegFormer weights which are knowledge distillation from SAM
weight_dir = "/Users/eatappleagain/trial/RipplesDetection/ripples_api/app/resource/state_dict/segformer_data_size_300.pth"

# Definition of my SegFormer
class MySegFormer_0409(nn.Module):
    def __init__(self, num_classes, backbone="b0", id2label=None):
        super().__init__()
        self.num_classes = num_classes
        self.id2label = id2label or {i: str(i) for i in range(num_classes)}
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/mit-{backbone}",
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id={v: k for k, v in self.id2label.items()},
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        y = self.segformer(x)
        # Interpolate to original size
        y = nn.functional.interpolate(
            y.logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
            antialias=True
        )
        return {"out": y}

# Declare the model
model = MySegFormer_0409(num_classes=2, backbone="b0")
# Load fine-tuned weights 
model.load_state_dict(torch.load(weight_dir, map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

def inference(frame_path: str) -> np.ndarray:
    image = Image.open(frame_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)["out"]
    mask = output.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
    return mask
