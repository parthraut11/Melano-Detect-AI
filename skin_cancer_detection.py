import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

NORM_MEAN = [0.7630392, 0.5456477, 0.57004845]
NORM_STD = [0.1409286, 0.15261266, 0.16997074]
IMAGE_SIZE = 224


with CLASS_NAMES_PATH.open("r", encoding="utf-8") as handle:
    CLASS_NAMES = json.load(handle)


def build_model():
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


MODEL = build_model()

PREPROCESS = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]
)


def prepare_image(image: Image.Image) -> torch.Tensor:
    tensor = PREPROCESS(image.convert("RGB"))
    return tensor.unsqueeze(0)


def predict(image: Image.Image) -> np.ndarray:
    tensor = prepare_image(image)
    with torch.no_grad():
        logits = MODEL(tensor)
        probabilities = F.softmax(logits, dim=1)
    return probabilities.squeeze(0).cpu().numpy()
