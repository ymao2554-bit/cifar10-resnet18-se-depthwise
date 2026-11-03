"""Single-image inference demo for the selected model."""
import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from model import make_model

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--model", choices=["baseline","depthwise","se","depthwise_se"], default="baseline")
    p.add_argument("--checkpoint", type=Path)
    a=p.parse_args()
    checkpoint=a.checkpoint or Path(__file__).parent / "results" / f"{a.model}.pt"
    model=make_model(a.model)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    tf=transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),
                           transforms.Normalize((.4914,.4822,.4465),(.2470,.2435,.2616))])
    with torch.inference_mode():
        probabilities=torch.softmax(model(tf(Image.open(a.image).convert("RGB")).unsqueeze(0)),1)[0]
    values,indices=probabilities.topk(3)
    for rank,(v,i) in enumerate(zip(values.tolist(), indices.tolist()),1):
        print(f"{rank}. {CLASSES[i]}: {v:.4f}")

if __name__ == "__main__": main()
