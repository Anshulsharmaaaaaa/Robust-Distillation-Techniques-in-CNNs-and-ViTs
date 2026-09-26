Robust Distillation Techniques in CNNs and ViTs
A reproducible deep-learning project that studies Knowledge Distillation (KD) for image classification across Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), with an emphasis on robustness to corrupted/noisy inputs.
The project is designed for a BTech CSE final-year implementation and a clean GitHub portfolio. It supports:
CNN teacher → CNN student distillation
ViT teacher → ViT student distillation
Cross-architecture CNN → ViT and ViT → CNN distillation
Standard response-based KD
Feature-based distillation
Robust KD with noisy/augmented inputs and teacher-student consistency
Clean accuracy + corruption robustness evaluation
Config-driven experiments
CSV/JSON logging for reproducible comparisons
1. Problem Statement
Deep neural networks can achieve high image-classification accuracy, but large models are expensive to train and deploy. Knowledge distillation transfers information from a larger teacher model to a smaller student model. This project investigates whether a carefully designed robust distillation objective can preserve accuracy while improving student stability under input perturbations.
Research question
> How do standard and robust knowledge-distillation strategies affect clean accuracy and corruption robustness when compressing CNNs and ViTs, including cross-architecture teacher-student pairs?
2. Objectives
Build baseline CNN and ViT classifiers on CIFAR-10.
Train larger teacher models and smaller student models.
Implement baseline supervised training.
Implement response-based KD using softened teacher logits.
Implement feature-based KD using intermediate representations.
Add robust distillation using noisy/augmented views and consistency loss.
Compare CNN→CNN, ViT→ViT, CNN→ViT, and ViT→CNN settings where computational resources allow.
Evaluate clean accuracy and corruption robustness.
Record results in machine-readable files and generate plots.
3. Method Overview
The student minimizes a weighted combination of supervised cross-entropy, logit KD, feature KD, and robust consistency:
`L = alpha * CE + beta * KD + gamma * FeatureKD + delta * RobustConsistency`
For response-based KD, teacher and student predictions are softened with temperature `T`. Robust consistency compares predictions on clean and perturbed/augmented views.
Robust perturbations used in this project
Random crop + horizontal flip through the training pipeline
Gaussian input noise during robust training
Stronger augmented/noisy views for the consistency objective
These perturbations are intended as a practical robustness experiment rather than a formal certified-robustness guarantee.
4. Dataset
CIFAR-10 is used by default because it is small enough for student projects and supports quick experimentation. The dataset contains 10 classes of 32×32 RGB images.
The code automatically downloads CIFAR-10 through `torchvision` on first use.
5. Repository Structure
```text
robust-distillation-cnn-vit/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── Makefile
├── configs/
│   ├── baseline_cnn.yaml
│   ├── kd_cnn.yaml
│   ├── robust_kd_cnn.yaml
│   ├── baseline_vit.yaml
│   ├── kd_vit.yaml
│   └── robust_kd_vit.yaml
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── utils.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   ├── vit.py
│   │   └── factory.py
│   └── distillation/
│       ├── __init__.py
│       ├── losses.py
│       └── engine.py
├── train.py
├── evaluate.py
├── plot_results.py
├── scripts/
│   ├── train_baseline.sh
│   └── run_all_experiments.sh
├── tests/
│   └── test_losses.py
├── reports/
│   └── project_report.md
├── results/
│   └── .gitkeep
├── weights/
│   └── .gitkeep
└── notebooks/
    └── .gitkeep
```
6. Quick Start
Clone
```bash
git clone https://github.com/<YOUR_USERNAME>/robust-distillation-cnn-vit.git
cd robust-distillation-cnn-vit
```
Create environment
Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Linux/macOS:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Run tests
```bash
python -m pytest -q
```
7. Training
Baseline CNN
```bash
python train.py --config configs/baseline_cnn.yaml
```
CNN knowledge distillation
First train the teacher:
```bash
python train.py --config configs/baseline_cnn.yaml
```
Then distill a smaller CNN:
```bash
python train.py --config configs/kd_cnn.yaml --teacher_checkpoint weights/cnn_teacher.pt
```
Robust CNN distillation
```bash
python train.py --config configs/robust_kd_cnn.yaml --teacher_checkpoint weights/cnn_teacher.pt
```
ViT
```bash
python train.py --config configs/baseline_vit.yaml
python train.py --config configs/robust_kd_vit.yaml --teacher_checkpoint weights/vit_teacher.pt
```
> Tip: start with 5–10 epochs to verify the pipeline. For a final report, run longer training and repeat the experiments with fixed seeds.
8. Evaluation
```bash
python evaluate.py --checkpoint weights/student.pt --model tiny_cnn --batch_size 128
```
For corruption robustness, the evaluator reports clean accuracy and accuracy under Gaussian-noise corruption at several noise levels.
9. Result Plotting
Place experiment CSV files in `results/` and run:
```bash
python plot_results.py --input results/experiment_results.csv --output results/accuracy_plot.png
```
10. Suggested Experiment Matrix
ID	Teacher	Student	Method	Main metric
E1	—	Small CNN	Baseline CE	Clean accuracy
E2	Large CNN	Small CNN	Standard KD	Clean accuracy
E3	Large CNN	Small CNN	Robust KD	Clean + noisy accuracy
E4	—	Small ViT	Baseline CE	Clean accuracy
E5	Large ViT	Small ViT	Standard KD	Clean accuracy
E6	Large ViT	Small ViT	Robust KD	Clean + noisy accuracy
E7	Large CNN	Small ViT	Robust KD	Cross-architecture
E8	Large ViT	Small CNN	Robust KD	Cross-architecture
11. What Counts as a Good Result?
Do not define success only as higher clean accuracy. Report at least:
Accuracy on clean CIFAR-10 test images
Accuracy at Gaussian noise σ = 0.05, 0.10, 0.20
Accuracy drop from clean → noisy
Parameter count
Approximate model size
Training time per experiment, if available
A robustly distilled student is interesting when it maintains competitive clean accuracy while reducing the accuracy degradation under perturbation.
12. GitHub Workflow
Recommended commit sequence:
```text
01-init-readme
02-add-data-pipeline
03-add-cnn-models
04-add-vit-models
05-add-kd-losses
06-add-training-engine
07-add-evaluation
08-add-baseline-configs
09-add-robust-kd
10-add-tests
11-add-result-plots
12-finalize-report
```
13. Limitations
This implementation is a research/education project, not a production training system. The Gaussian-noise robustness test is a limited corruption benchmark and should not be interpreted as adversarial robustness, certified robustness, or security assurance.
14. Future Work
Add CIFAR-100 and Tiny-ImageNet.
Add PGD/FGSM evaluation for adversarial robustness.
Compare DeiT, Swin, ConvNeXt, and modern lightweight CNNs.
Add attention-map and relational knowledge distillation.
Add mixed precision and multi-GPU training.
Use automated hyperparameter sweeps.
Add statistical confidence intervals across multiple random seeds.
15. Academic Deliverables
This repository can support:
Project proposal
Synopsis
Literature review
Methodology chapter
Experimental results
Final BTech project report
Viva presentation
GitHub portfolio
16. License
MIT License. See `LICENSE`.
