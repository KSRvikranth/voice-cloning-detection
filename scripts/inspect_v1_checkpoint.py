
from pathlib import Path
import torch

model_path = Path("models/voice_cnn_multisource.pth")

if not model_path.exists():
    print("V1 model file not found!")
else:
    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    print("V1 checkpoint loaded.")
    print("\nModel layers and tensor shapes:")

    for name, tensor in state_dict.items():
        print(name, tuple(tensor.shape))