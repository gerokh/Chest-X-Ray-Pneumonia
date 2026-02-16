import torch
import torchvision.transforms as transforms
import sys

try:
    # We must load with weights_only=False because it's a pickle of a class
    tr = torch.load('test_transforms.pth', map_location='cpu', weights_only=False)
    print("Type:", type(tr))
    if hasattr(tr, 'transforms'):
        print("Transforms list:")
        for t in tr.transforms:
            print(f" - {t}")
    else:
        print("Transform object:", tr)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
