import torch
from PIL import Image
import numpy as np

from model.parameters import transform
from model.train import FruitClassifier
from .schemas import PixelData

model = FruitClassifier(num_classes=1)
model.load_state_dict(torch.load("./models/model-0.875-v2.pth", map_location=torch.device('cpu')))
model.eval()

def predict(data: PixelData):
    arr = np.array(data.pixels, dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(image)
        probability = torch.sigmoid(logits).item()
        fruit_state = "rotten" if probability > 0.5 else "fresh"
        probability = (probability if fruit_state == "rotten" else 1 - probability) * 100
        return probability, fruit_state
