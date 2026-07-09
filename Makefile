.PHONY: help setup download-instance download-semantic download-all eda-instance eda-semantic train-instance train-semantic compare depth pipeline all

SRC = car_damage_detection/src

help:
	@echo "📦 Setup:"
	@echo "  make setup             - Sync dependencies with uv"
	@echo ""
	@echo "📥 Datasets:"
	@echo "  make download-instance - Download instance segmentation dataset"
	@echo "  make download-semantic - Download semantic segmentation dataset"
	@echo "  make download-all      - Download both datasets"
	@echo ""
	@echo "📊 EDA:"
	@echo "  make eda-instance      - EDA notebook for instance dataset"
	@echo "  make eda-semantic      - EDA notebook for semantic dataset"
	@echo ""
	@echo "🤖 Training:"
	@echo "  make train-instance    - Train instance segmentation (YOLOv8-Seg)"
	@echo "  make train-semantic    - Train semantic segmentation"
	@echo ""
	@echo "📈 Evaluation:"
	@echo "  make compare           - Compare instance vs semantic results"
	@echo ""
	@echo "🔗 Pipeline:"
	@echo "  make depth             - Run depth estimation + damage classifier"
	@echo "  make pipeline          - Run full pipeline (seg + depth → damage)"
	@echo ""
	@echo "⚡ All:"
	@echo "  make all               - Run everything sequentially"

setup:
	@echo "📦 Installing dependencies..."
	uv sync

download-instance:
	@echo "📥 Downloading instance segmentation dataset..."
	uv run python $(SRC)/utils/download_instance_dataset.py
	@echo "✅ Instance dataset downloaded"

download-semantic:
	@echo "📥 Downloading semantic segmentation dataset..."
	uv run python $(SRC)/utils/download_semantic_dataset.py
	@echo "✅ Semantic dataset downloaded"

download-all: download-instance download-semantic
	@echo "✅ All datasets downloaded"

eda-instance:
	@echo "📊 Running EDA for instance dataset..."
	@if [ -f car_damage_detection/notebooks/01_eda_instance.ipynb ]; then \
		uv run jupyter nbconvert --to notebook --execute car_damage_detection/notebooks/01_eda_instance.ipynb --output 01_eda_instance_executed.ipynb; \
	else \
		echo "Notebook not found. Run: uv run jupyter notebook car_damage_detection/notebooks/01_eda_instance.ipynb"; \
	fi

eda-semantic:
	@echo "📊 Running EDA for semantic dataset..."
	@if [ -f car_damage_detection/notebooks/02_eda_semantic.ipynb ]; then \
		uv run jupyter nbconvert --to notebook --execute car_damage_detection/notebooks/02_eda_semantic.ipynb --output 02_eda_semantic_executed.ipynb; \
	else \
		echo "Notebook not found. Run: uv run jupyter notebook car_damage_detection/notebooks/02_eda_semantic.ipynb"; \
	fi

train-instance:
	@echo "🤖 Training instance segmentation..."
	uv run python $(SRC)/segmentation/train_instance.py

train-semantic:
	@echo "🤖 Training semantic segmentation..."
	uv run python $(SRC)/segmentation/train_semantic.py

compare:
	@echo "📈 Comparing instance vs semantic..."
	uv run jupyter nbconvert --to notebook --execute car_damage_detection/notebooks/03_comparison.ipynb --output 03_comparison_executed.ipynb 2>/dev/null || \
	@echo "Run: uv run jupyter notebook car_damage_detection/notebooks/03_comparison.ipynb"

depth:
	@echo "🔗 Running depth estimation + damage classifier..."
	uv run python $(SRC)/depth/depth_estimator.py

pipeline:
	@echo "🔗 Running full pipeline..."
	uv run python $(SRC)/pipeline.py

all: setup download-all eda-instance eda-semantic train-instance train-semantic compare depth pipeline
	@echo "✅ Full pipeline complete"
