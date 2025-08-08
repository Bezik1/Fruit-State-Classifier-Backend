import torch
from PIL import Image
import model.parameters as parameters
from .train import FruitClassifier

model = FruitClassifier(num_classes=1)
model.load_state_dict(torch.load("./models/model-0.875-v2.pth", map_location=torch.device('cpu')))
model.eval()

img_path = "./evaluation_data/test_2-fresh.jpg"

image = Image.open(img_path).convert('RGB')
image = parameters.transform(image).unsqueeze(0)

with torch.no_grad():
    logits = model(image)
    prob = torch.sigmoid(logits).item()

print(f"Probability for rotten fruit: {prob:.4f}")

fruit_state = "rotten" if prob > 0.5 else "fresh"
print(f"Fruit state: {fruit_state}")
