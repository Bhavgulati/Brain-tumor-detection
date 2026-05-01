# NeuroScan AI — Brain Tumor & Skin Cancer Detection

AI-powered medical imaging platform built with Flask + PyTorch (ResNet50).

## Features
- Brain MRI classification (Glioma, Meningioma, No Tumor, Pituitary)
- Skin Cancer detection (Benign vs Malignant)
- GradCAM visual explainability heatmaps
- Doctor and Patient dashboards
- PDF report generation
- Scan comparison and statistics

## Tech Stack
Python, Flask, PyTorch, ResNet50, SQLite, ReportLab, Bootstrap 5, Chart.js

## Setup
pip install -r requirements.txt
python app.py

## Architecture
- Model: ResNet50 Transfer Learning from ImageNet
- Classes: Glioma, Meningioma, No Tumor, Pituitary + Benign/Malignant
- Explainability: GradCAM heatmaps on last conv layer
- Accuracy: ~95% Brain Tumor | ~83% Skin Cancer

## Dataset
- Brain Tumor MRI: Kaggle — Masoud Nickparvar (5712 images, 4 classes)
- Skin Cancer: Kaggle — Hasnain Javed (9605 images, 2 classes)
