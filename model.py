import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class BayesianEfficientNet(nn.Module):
    def __init__(self, model_name="efficientnet-b0", num_classes=14, p_drop=0.2):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained(model_name)
        in_f = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        self.dropout = nn.Dropout(p=p_drop)
        self.head    = nn.Linear(in_f, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = self.dropout(x)      # keeps dropout active at inference if .train()
        return self.head(x)