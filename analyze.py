"""Recompute visualizations from experiment JSON files and saved checkpoints."""
import json
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import make_model

ROOT = Path(__file__).parent
NAMES = ["baseline", "depthwise", "se", "depthwise_se"]
DISPLAY = {"baseline":"Baseline", "depthwise":"DW Conv", "se":"SE", "depthwise_se":"DW + SE"}
LABELS = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
COLORS = ["#2463a6", "#e1843c", "#328f79", "#8156a5"]
plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11, "axes.spines.top":False,
                     "axes.spines.right":False, "figure.facecolor":"white", "axes.labelcolor":"#203040"})


@torch.inference_mode()
def corrupted_accuracy(model, loader, corruption, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device)
        mean = torch.tensor((0.4914,0.4822,0.4465), device=device)[None,:,None,None]
        std = torch.tensor((0.2470,0.2435,0.2616), device=device)[None,:,None,None]
        pixels = x * std + mean
        if corruption == "dark":
            pixels = pixels * 0.65
        elif corruption == "noise":
            g = torch.Generator(device=device).manual_seed(total + 42)
            pixels = pixels + torch.randn(x.shape, generator=g, device=device) * 0.05
        x = (pixels.clamp(0,1) - mean) / std
        p = model(x).argmax(1).cpu()
        correct += (p == y).sum().item(); total += len(y)
    return correct / total


def main():
    results = ROOT / "results"; figures = ROOT / "figures"; figures.mkdir(exist_ok=True)
    data = [json.loads((results / f"{n}.json").read_text(encoding="utf-8")) for n in NAMES]
    with (results / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["model","test_accuracy","correct","total","parameters","macs","cpu_latency_ms","train_seconds"])
        for d in data:
            writer.writerow([d["name"],d["test"]["accuracy"],d["test"]["correct"],d["test"]["total"],
                             d["parameters"],d["macs"],d["cpu_latency_ms"],d["train_seconds"]])

    split_source = datasets.CIFAR10(ROOT / "data", train=True, download=False)
    test_source = datasets.CIFAR10(ROOT / "data", train=False, download=False)
    indices = np.random.default_rng(data[0]["seed"]).permutation(50000)
    y = np.asarray(split_source.targets)
    counts = [np.bincount(y[indices[5000:]], minlength=10),
              np.bincount(y[indices[:5000]], minlength=10),
              np.bincount(np.asarray(test_source.targets), minlength=10)]
    fig, ax = plt.subplots(figsize=(8.4,3.8), layout="constrained")
    x=np.arange(10); bottom=np.zeros(10)
    for count, label, color in zip(counts,["Train","Validation","Test"],["#2463a6","#e5a55e","#328f79"]):
        ax.bar(x,count,bottom=bottom,label=label,color=color,width=.72)
        bottom+=count
    ax.set(xticks=x,xticklabels=LABELS,ylabel="Images",title="Class distribution by split",ylim=(0,6500))
    ax.tick_params(axis="x",labelrotation=30);ax.legend(frameon=False,ncol=3)
    fig.savefig(figures / "data_distribution.png",dpi=220);plt.close(fig)

    fig, ax=plt.subplots(figsize=(9,2.7),layout="constrained")
    ax.set_xlim(0,10);ax.set_ylim(0,2);ax.axis("off")
    stages=[("Input","3×32×32"),("Stem","3×3 conv"),("Stage 1","2 × 32"),("Stage 2","2 × 64"),
            ("Stage 3","2 × 128"),("Stage 4","2 × 256"),("GAP + FC","10 classes")]
    xs=np.linspace(.75,9.25,len(stages))
    for i,((name,detail),cx) in enumerate(zip(stages,xs)):
        color="#1f4e79" if i in (0,6) else "#e4eef6"
        fg="white" if i in (0,6) else "#1f3a53"
        ax.text(cx,1.0,f"{name}\n{detail}",ha="center",va="center",fontsize=9.5,color=fg,
                bbox=dict(boxstyle="round,pad=.65",facecolor=color,edgecolor="#87a7c1",linewidth=1))
        if i:
            ax.annotate("",xy=(cx-.5,1),xytext=(xs[i-1]+.5,1),arrowprops=dict(arrowstyle="->",color="#6c849b",lw=1.7))
    fig.savefig(figures / "architecture.png",dpi=220);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(8.4,3.5),layout="constrained")
    for ax,key,title,unit in [(axes[0],"parameters","Parameters","million"),(axes[1],"macs","MACs per image","million")]:
        values=[d[key]/1e6 for d in data]
        ax.barh(range(4),values,color=COLORS,height=.62)
        ax.set(yticks=range(4),yticklabels=[DISPLAY[n] for n in NAMES],xlabel=unit,title=title)
        ax.invert_yaxis();ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
        for i,v in enumerate(values):ax.text(v+max(values)*.015,i,f"{v:.2f}",va="center",fontsize=9)
        ax.set_xlim(0,max(values)*1.23)
    fig.savefig(figures / "compression.png",dpi=220);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(8.4,3.5),layout="constrained")
    pairs=[("Baseline","SE",data[0],data[2]),("DW Conv","DW + SE",data[1],data[3])]
    for ax,key,mult,title in [(axes[0],"test",100,"Test accuracy (%)"),(axes[1],"cpu_latency_ms",1,"CPU latency (ms / image)")]:
        for i,(left,right,a,b) in enumerate(pairs):
            va=a[key]["accuracy"]*mult if key=="test" else a[key]
            vb=b[key]["accuracy"]*mult if key=="test" else b[key]
            ax.plot([0,1],[va,vb],color=COLORS[i],lw=2,marker="o",markersize=8,label=left+" → "+right)
            ax.text(1.04,vb,f"{vb:.2f}",va="center",fontsize=9,color=COLORS[i])
        ax.set(xticks=[0,1],xticklabels=["Without SE","With SE"],title=title,xlim=(-.15,1.25))
        ax.grid(axis="y",alpha=.15)
    axes[0].legend(frameon=False,loc="lower center",fontsize=8)
    fig.savefig(figures / "se_effect.png",dpi=220);plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.4,3.4),layout="constrained")
    for d,color in zip(data,COLORS):
        ax.plot([h["epoch"] for h in d["history"]],[h["val_loss"] for h in d["history"]],
                label=DISPLAY[d["name"]],color=color,lw=2)
    ax.set(xlabel="Epoch",ylabel="Validation loss",title="Validation cross-entropy during training")
    ax.grid(alpha=.16);ax.legend(frameon=False,ncol=2)
    fig.savefig(figures / "loss_curves.png",dpi=220);plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    for d, color in zip(data, COLORS):
        ax.plot([h["epoch"] for h in d["history"]], [100*h["val_accuracy"] for h in d["history"]],
                label=DISPLAY[d["name"]], lw=2.4, color=color)
    ax.set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Validation accuracy across training")
    ax.grid(alpha=.16); ax.legend(frameon=False, ncol=2)
    fig.savefig(figures / "learning_curves.png", dpi=220); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    for d, color in zip(data, COLORS):
        x, y = d["macs"] / 1e6, d["test"]["accuracy"] * 100
        ax.scatter(x, y, s=1100, color=color, alpha=.18, edgecolors="none")
        ax.scatter(x, y, s=90, color=color)
        ax.annotate(DISPLAY[d["name"]], (x,y), xytext=(7,7), textcoords="offset points", fontsize=10)
    ax.set(xlabel="MACs per image (million)", ylabel="Test accuracy (%)", title="Accuracy and compute trade-off")
    ax.grid(alpha=.16)
    fig.savefig(figures / "tradeoff.png", dpi=220); plt.close(fig)

    best = max(data, key=lambda d:d["test"]["accuracy"])
    cm = confusion_matrix(best["test"]["labels"], best["test"]["predictions"], labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(7, 6.2), layout="constrained")
    im = ax.imshow(cm, cmap="Blues")
    for i in range(10):
        for j in range(10):
            if i == j or cm[i,j] >= 35:
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        fontsize=7, color="white" if cm[i,j] > 400 else "#243a52")
    ax.set(xticks=range(10), yticks=range(10), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted", ylabel="True", title=f"Confusion matrix: {best['name']}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im, ax=ax, shrink=.78, label="Images")
    fig.savefig(figures / "confusion.png", dpi=220); plt.close(fig)

    raw_test = datasets.CIFAR10(ROOT / "data", train=False, download=False)
    labels = np.asarray(best["test"]["labels"])
    preds = np.asarray(best["test"]["predictions"])
    picks = list(np.flatnonzero(labels == preds)[:5]) + list(np.flatnonzero(labels != preds)[:5])
    fig, axes = plt.subplots(2, 5, figsize=(9, 3.7), layout="constrained")
    for ax, idx in zip(axes.flat, picks):
        ax.imshow(raw_test.data[idx], interpolation="nearest")
        ax.set_title(f"true {LABELS[labels[idx]]}\npred {LABELS[preds[idx]]}",
                     color="#2463a6" if labels[idx] == preds[idx] else "#c34b3d", fontsize=9)
        ax.axis("off")
    fig.savefig(figures / "predictions.png", dpi=220); plt.close(fig)

    mean, std = (0.4914,0.4822,0.4465), (0.2470,0.2435,0.2616)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    test = datasets.CIFAR10(ROOT / "data", train=False, download=False, transform=tf)
    loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robust = {}
    for d in data:
        model = make_model(d["name"], d["width"])
        model.load_state_dict(torch.load(results / f"{d['name']}.pt", map_location="cpu", weights_only=True))
        model.to(device)
        robust[d["name"]] = {c:corrupted_accuracy(model, loader, c, device) for c in ("dark","noise")}
    (results / "robustness.json").write_text(json.dumps(robust, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    x=np.arange(4); w=.26
    for j, condition in enumerate(("clean","dark","noise")):
        vals=[100*(data[i]["test"]["accuracy"] if condition=="clean" else robust[NAMES[i]][condition]) for i in range(4)]
        ax.bar(x+(j-1)*w, vals, w, label=condition, color=["#2463a6","#e5a55e","#7b68a6"][j])
    ax.set(xticks=x, xticklabels=[DISPLAY[n] for n in NAMES], ylabel="Accuracy (%)", title="Performance under input changes", ylim=(0,100))
    ax.tick_params(axis="x", labelrotation=15); ax.legend(frameon=False, ncol=3)
    fig.savefig(figures / "robustness.png", dpi=220); plt.close(fig)
    print(json.dumps(robust, indent=2))

if __name__ == "__main__": main()
