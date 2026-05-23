.PHONY: install prepare train eval mlflow serve

install:
	pip install -r requirements.txt
	pip install -r app/requirements.txt

prepare:
	python data/prepare.py

train:
	caffeinate -i python train/train.py

eval:
	python eval/eval.py

mlflow:
	mlflow ui --host 127.0.0.1 --port 5000

serve:
	uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
