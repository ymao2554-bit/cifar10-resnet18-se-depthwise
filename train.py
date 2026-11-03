"""Train one controlled CIFAR-10 experiment; all reported figures derive from saved JSON."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import make_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loaders(root, batch, workers, seed):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = datasets.CIFAR10(root, train=True, download=False, transform=train_tf)
    val_set = datasets.CIFAR10(root, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(root, train=False, download=False, transform=eval_tf)
    indices = np.random.default_rng(seed).permutation(50000)
    kw = dict(batch_size=batch, num_workers=workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=workers > 0)
    return (DataLoader(Subset(train_set, indices[5000:]), shuffle=True, **kw),
            DataLoader(Subset(val_set, indices[:5000]), shuffle=False, **kw),
            DataLoader(test_set, shuffle=False, **kw))


@torch.inference_mode()
def evaluate(model, loader, device, get_predictions=False):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        p = logits.argmax(1)
        correct += (p == y).sum().item()
        total += y.numel()
        if get_predictions:
            preds.extend(p.cpu().tolist())
            labels.extend(y.cpu().tolist())
    out = {"loss": loss_sum / total, "accuracy": correct / total, "correct": correct, "total": total}
    if get_predictions:
        out.update(predictions=preds, labels=labels)
    return out


def macs(model):
    total = [0]
    hooks = []
    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            total[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.Linear):
            total[0] += out.numel() * m.in_features
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.inference_mode():
        model(torch.zeros(1, 3, 32, 32))
    for h in hooks:
        h.remove()
    return total[0]


@torch.inference_mode()
def cpu_latency(model, repeats=100):
    model = model.cpu().eval()
    x = torch.randn(1, 3, 32, 32)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    for _ in range(20):
        model(x)
    start = time.perf_counter()
    for _ in range(repeats):
        model(x)
    elapsed = (time.perf_counter() - start) * 1000 / repeats
    torch.set_num_threads(old)
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["baseline", "depthwise", "se", "depthwise_se"], required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=32)
    args = ap.parse_args()
    seed_all(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test = loaders(args.data, args.batch, args.workers, args.seed)
    model = make_model(args.model, args.width)
    params = sum(p.numel() for p in model.parameters())
    model_macs = macs(model)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best = [], -1
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = count = correct = 0
        for x, y in train:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.item() * y.numel(); count += y.numel()
            correct += (logits.argmax(1) == y).sum().item()
        validation = evaluate(model, val, device)
        row = {"epoch": epoch + 1, "train_loss": loss_sum/count, "train_accuracy": correct/count,
               "val_loss": validation["loss"], "val_accuracy": validation["accuracy"]}
        history.append(row)
        if validation["accuracy"] > best:
            best = validation["accuracy"]
            torch.save(model.state_dict(), out / f"{args.model}.pt")
        scheduler.step()
        print(json.dumps({"model":args.model, **row}), flush=True)
    train_seconds = time.perf_counter() - start
    model.load_state_dict(torch.load(out / f"{args.model}.pt", map_location=device, weights_only=True))
    testing = evaluate(model, test, device, True)
    latency = cpu_latency(model)
    result = {"name":args.model, "seed":args.seed, "epochs":args.epochs, "batch":args.batch,
              "width":args.width, "device":str(device), "parameters":params, "macs":model_macs,
              "best_val_accuracy":best, "test":testing, "cpu_latency_ms":latency,
              "train_seconds":train_seconds, "history":history}
    (out / f"{args.model}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FINAL", json.dumps({k:v for k,v in result.items() if k not in ("history","test")}, ensure_ascii=False),
          "TEST", testing["accuracy"], flush=True)

if __name__ == "__main__":
    main()
