import torch
import torch.nn as nn
from torchvision import models
import sys

# Define class (same as app.py)
class ENB4WithEmbeddings(nn.Module):
    def __init__(self, num_classes=2):
        super(ENB4WithEmbeddings, self).__init__()
        self.base_model = models.efficientnet_b4(weights=None)
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes, bias=True)
        )

def test_load():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_lines = []
    def log(msg):
        print(msg)
        output_lines.append(str(msg) + "\n")
        
    try:
        model_enb4 = ENB4WithEmbeddings(num_classes=2)
        state_dict = torch.load('efficientnetb4_model.pth', map_location=device)
        
        log(f"Loaded keys sample: {list(state_dict.keys())[:5]}")
        log(f"Model keys sample: {list(model_enb4.state_dict().keys())[:5]}")
        
        # Logic from app.py
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('features.') or k.startswith('classifier.'):
                new_state_dict[f'base_model.{k}'] = v
            else:
                new_state_dict[k] = v
        
        log(f"Mapped keys sample: {list(new_state_dict.keys())[:5]}")
        
        # Try strict=False
        missing, unexpected = model_enb4.load_state_dict(new_state_dict, strict=False)
        log(f"MISSING_COUNT: {len(missing)}")
        if missing: log(f"MISSING_SAMPLE: {missing[:5]}")
        log(f"UNEXPECTED_COUNT: {len(unexpected)}")
        if unexpected: log(f"UNEXPECTED_SAMPLE: {unexpected[:5]}")
            
    except Exception as e:
        log(f"Load failed with exception: {e}")
    finally:
        with open('debug_output.txt', 'w') as f:
            f.writelines(output_lines)


if __name__ == "__main__":
    test_load()
