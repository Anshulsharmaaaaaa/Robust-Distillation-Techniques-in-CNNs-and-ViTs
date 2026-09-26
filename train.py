import argparse
import os
from pathlib import Path

import torch
import yaml

from src.data import build_loaders
from src.models import build_model
from src.distillation.engine import fit, count_parameters
from src.utils import load_checkpoint, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline or distilled CNN/ViT models.")
    p.add_argument("--config", required=True, help="Path to YAML experiment config")
    p.add_argument("--teacher_checkpoint", default=None, help="Optional teacher checkpoint")
    p.add_argument("--device", default=None, help="cpu or cuda (auto by default)")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    seed_everything(int(config.get("seed", 42)))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = int(config["data"].get("batch_size", 128))
    workers = int(config["data"].get("workers", 2))
    train_loader, test_loader = build_loaders(config["data"].get("data_dir", "data"), batch_size, workers)

    student = build_model(config["student"]["model"], config["data"].get("num_classes", 10)).to(device)
    teacher = None
    if args.teacher_checkpoint:
        teacher = build_model(config["teacher"]["model"], config["data"].get("num_classes", 10)).to(device)
        load_checkpoint(args.teacher_checkpoint, teacher, device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

    Path(config["output"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(config["output"]["checkpoint_dir"], config["output"]["checkpoint_name"])
    history_path = os.path.join(config["output"]["result_dir"], config["output"]["history_name"])
    Path(config["output"]["result_dir"]).mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Student: {config['student']['model']} ({count_parameters(student):,} trainable params)")
    if teacher is not None:
        print(f"Teacher: {config['teacher']['model']} ({count_parameters(teacher):,} params)")

    fit(student, teacher, train_loader, test_loader, config, device, checkpoint_path, history_path)
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
