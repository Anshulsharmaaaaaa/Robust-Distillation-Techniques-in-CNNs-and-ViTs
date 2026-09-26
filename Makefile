install:
	python -m pip install -r requirements.txt

test:
	python -m pytest -q

baseline-cnn:
	python train.py --config configs/baseline_cnn.yaml

robust-cnn:
	python train.py --config configs/robust_kd_cnn.yaml --teacher_checkpoint weights/cnn_teacher.pt

baseline-vit:
	python train.py --config configs/baseline_vit.yaml

robust-vit:
	python train.py --config configs/robust_kd_vit.yaml --teacher_checkpoint weights/vit_teacher.pt
