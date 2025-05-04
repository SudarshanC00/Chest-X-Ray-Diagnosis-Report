# inference.py
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from model import BayesianEfficientNet   # adjust import

# 1. Rebuild the model and load weights
BASE_DIR = Path(__file__).resolve().parent

# Path to checkpoint
ckpt_path = BASE_DIR / 'chexpert_bayesian_effnet.pth'

# Load
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BayesianEfficientNet().to(device)
model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
model.eval()

# 2. Preprocessing pipeline
imagenet_stats = dict(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(**imagenet_stats),
])

# 3. MC-Dropout prediction function
def predict_with_uncertainty(img: Image.Image, T: int = 20):
    x = transform(img).unsqueeze(0).to(device)
    model.dropout.train()   # keep dropout on
    preds = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(x)
            preds.append(torch.sigmoid(logits))
    preds = torch.stack(preds)           # [T, 1, C]
    mean = preds.mean(dim=0).cpu().numpy()[0]
    var  = preds.var(dim=0).cpu().numpy()[0]
    return mean, var

# 4. Convenience wrapper
label_cols = [
    "No Finding","Enlarged Cardiomegastinum","Cardiomegaly","Lung Opacity",
    "Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis",
    "Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"
]

def infer_image(pil_img):
    mean, var = predict_with_uncertainty(pil_img, T=20)
    return {cls: {"conf": float(mean[i]), "uncertainty": float(var[i])}
            for i, cls in enumerate(label_cols)}