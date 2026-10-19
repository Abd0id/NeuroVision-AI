# NeuroVision AI 

Intelligent Brain MRI Classification & Grad-CAM Visual Explainability Platform.

## Overview

**NeuroVision AI** provides deep learning-driven brain MRI analysis to assist researchers and clinicians in detecting and categorizing brain tumors from axial, coronal, and sagittal MRI scans.

### Monitored Conditions
- **Glioma** (`glioma`) - Primary intra-axial brain tumor arising from glial support cells.
- **Meningioma** (`meningioma`) - Extra-axial dural-based tumor arising from arachnoid meninges.
- **Pituitary Tumor** (`pituitary`) - Adenoma located in the sellar / suprasellar space.
- **No Tumor Detected** (`notumor`) - Normal, unremarkable brain parenchyma.

---

## Getting Started

### 1. Requirements & Dependencies
Ensure dependencies are installed using `uv`:
```bash
uv sync
```
Or install with pip:
```bash
pip install -e .
```

### 2. Model Serialization
The application automatically scans the `models/` folder for serialized neural network models:
- Accepted formats: `.keras`, `.h5`, `.hdf5`, or TensorFlow `SavedModel` directories.
- Example model path: `models/best_model.keras`
- If multiple models are present, a dropdown in the Streamlit sidebar allows switching between them dynamically.
- A baseline architecture initializer (`create_and_save_baseline_model`) is included so you can test the entire pipeline even before completing model training.

### 3. Launching the Streamlit App
Run the Streamlit application from the project root:
```bash
uv run streamlit run app.py
```
Or directly:
```bash
streamlit run app.py
```

The web dashboard will be available at `http://localhost:8501`.

---

## Application Features

1. **Model Hub & Hot-Swapping**:
   - Automatically detects, inspects, and loads serialized models from `models/`.
   - Displays parameter metadata, file size, and architecture status.

2. **MRI Ingestion**:
   - **Upload Custom Scan**: Drag & drop any DICOM-converted MRI scan (`.png`, `.jpg`, `.jpeg`).
   - **Quick Sample Gallery**: Built-in test samples from `data/processed` and `data/raw` for instant 1-click diagnostic testing.

3. **Radiological Image Controls**:
   - Brightness & Contrast fine-tuning.
   - Dark/Light Inversion (ideal for subtle hypointense / hyperintense lesion visualization).
   - Sobel Edge & Contour detection filter.

4. **Explainable AI (Grad-CAM)**:
   - Deep Convolutional activation heatmaps revealing exactly where the neural network concentrated its attention.
   - Selectable colormaps (`Jet`, `Turbo`, `Viridis`, `Inferno`).
   - Adjustable opacity overlay slider ($\alpha$).
   - Side-by-side comparison mode (Raw MRI, Heatmap, Superimposed).

5. **Diagnostic Findings & Monograph**:
   - Primary predicted class and confidence percentage.
   - Differential diagnosis probability distribution across all 4 classes.
   - Medical monograph detailing pathophysiology, MRI characteristics, symptoms, and clinical next steps.

6. **Export Diagnostic Reports**:
   - Download complete assessment reports in Markdown (`.md`) or structured JSON format.

---

## ⚠️ Medical Disclaimer
*NeuroVision AI is developed for research and educational purposes only. It is not an FDA/CE-cleared diagnostic device and should never replace consultation with a board-certified radiologist or medical oncologist.*
