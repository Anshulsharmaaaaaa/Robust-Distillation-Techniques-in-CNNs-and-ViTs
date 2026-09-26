import argparse
import json
from pathlib import Path

import torch
import yaml

from src.data import build_test_loader
from src.models import build_model
from src.utils import load_checkpoint, count_parameters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", required=True, choices=["tiny_cnn", "medium_cnn", "tiny_vit", "medium_vit"])
    p.add_argument("--data_dir", default="data")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--noise_levels", default="0,0.05,0.10,0.20")
    p.add_argument("--output", default="results/evaluation.json")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(args.model).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()

    results = {"model": args.model, "parameters": count_parameters(model), "checkpoint": args.checkpoint, "device": str(device), "evaluations": {}}
    for level in [float(x) for x in args.noise_levels.split(",") if x.strip()]:
        loader = build_test_loader(args.data_dir, args.batch_size, args.workers, noise_std=level)
        correct, total = 0, 0
        with torch.no_grad():
            for images, targets in loader:
                images, targets = images.to(device), targets.to(device)
                pred = model(images).argmax(1)
                correct += int((pred == targets).sum())
                total += targets.numel()
        results["evaluations"][str(level)] = 100.0 * correct / total
        print(f"Noise std={level:.2f}: {results['evaluations'][str(level)]:.2f}%")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
