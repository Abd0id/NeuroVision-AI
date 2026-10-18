"""
NeuroVision AI - Model & Preprocessing Utilities.

Provides model discovery, loading, image preprocessing, inference,
Grad-CAM explainability, and dataset sample discovery for brain MRI scans.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

# Standard classes in alphabetical order matching model_training.ipynb
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_INFO = {
    "glioma": {
        "display_name": "Glioma",
        "category": "Tumor Detected",
        "severity": "High",
        "badge_color": "#ef4444",  # Red
        "icon": "⚠️",
        "description": (
            "Gliomas originate in glial cells (astrocytes, oligodendrocytes, or ependymal cells) "
            "that surround and support nerve cells. They are among the most common primary brain tumors "
            "and can range from low-grade to aggressive high-grade tumors (e.g., glioblastoma)."
        ),
        "typical_symptoms": [
            "Persistent or worsening headaches",
            "Nausea and vomiting due to increased intracranial pressure",
            "Seizures of new onset",
            "Cognitive or personality changes",
            "Focal neurological deficits (weakness, numbness, vision changes)",
        ],
        "mri_features": (
            "Typically hyperintense on T2/FLAIR sequences, heterogeneous contrast enhancement "
            "on T1-post contrast, often with surrounding vasogenic edema and mass effect."
        ),
        "clinical_recommendation": (
            "Urgent neuro-oncology and neurosurgical consultation. High-resolution contrast-enhanced "
            "MRI (T1+C, T2, FLAIR, DWI, MRS) and multidisciplinary tumor board review recommended."
        ),
    },
    "meningioma": {
        "display_name": "Meningioma",
        "category": "Tumor Detected",
        "severity": "Moderate to High",
        "badge_color": "#f59e0b",  # Amber
        "icon": "⚠️",
        "description": (
            "Meningiomas arise from the meninges—the protective membranes covering the brain and spinal cord. "
            "The vast majority (around 85-90%) are benign (WHO Grade I) and slow-growing, though atypical "
            "or malignant variants exist. They exert symptoms primarily by compressing adjacent brain tissue."
        ),
        "typical_symptoms": [
            "Localized headaches",
            "Progressive focal neurological symptoms depending on tumor location",
            "Seizures",
            "Visual disturbances or hearing loss (if near cranial nerves)",
            "Motor weakness or sensory deficits",
        ],
        "mri_features": (
            "Extra-axial, well-circumscribed dural-based mass, typically isointense to hypointense on T1, "
            "isointense to hyperintense on T2, with intense homogeneous post-contrast enhancement and a characteristic 'dural tail'."
        ),
        "clinical_recommendation": (
            "Neurosurgical evaluation to determine surgical resectability vs. watchful waiting "
            "or stereotactic radiosurgery (Gamma Knife / CyberKnife) based on tumor size, location, and symptoms."
        ),
    },
    "notumor": {
        "display_name": "No Tumor Detected",
        "category": "Normal / Unremarkable",
        "severity": "Normal",
        "badge_color": "#10b981",  # Emerald Green
        "icon": "✅",
        "description": (
            "The MRI scan demonstrates normal intracranial parenchyma without evidence of abnormal mass lesion, "
            "neoplastic enhancement, pathological midline shift, or abnormal perilesional edema."
        ),
        "typical_symptoms": [
            "No specific neoplastic symptoms identified",
            "Any presenting headaches or symptoms may be related to non-neoplastic causes (migraine, tension, sinus, cervical)",
        ],
        "mri_features": (
            "Normal brain parenchyma, symmetrical ventricles, normal gray-white matter differentiation, "
            "absence of pathological contrast enhancement or mass effect."
        ),
        "clinical_recommendation": (
            "Routine clinical follow-up as indicated by the patient's presenting symptoms. "
            "Correlate with clinical history and other diagnostic evaluations if non-tumor pathology is suspected."
        ),
    },
    "pituitary": {
        "display_name": "Pituitary Tumor",
        "category": "Tumor Detected",
        "severity": "Moderate",
        "badge_color": "#8b5cf6",  # Purple
        "icon": "⚠️",
        "description": (
            "Pituitary adenomas arise in the sellar region from pituitary gland parenchyma. "
            "Most are benign, but can cause clinical effects either via hormonal hypersecretion "
            "(functional adenomas like prolactinoma, Cushing's, acromegaly) or local mass effect on the optic chiasm."
        ),
        "typical_symptoms": [
            "Bitemporal hemianopsia (tunnel vision) from optic chiasm compression",
            "Endocrine dysregulation (amenorrhea, galactorrhea, fatigue, unexplained weight changes)",
            "Headaches, particularly retro-orbital",
            "Hypopituitarism (decreased energy, hypotension, low libido)",
        ],
        "mri_features": (
            "Sellar and/or suprasellar mass expanding the sella turcica. Microadenomas (<10mm) show delayed "
            "enhancement compared to normal gland. Macroadenomas (>10mm) demonstrate sellar remodeling and suprasellar extension."
        ),
        "clinical_recommendation": (
            "Endocrinology and neurosurgical consultation. Comprehensive serum pituitary hormone panel "
            "(prolactin, IGF-1, ACTH, cortisol, TSH/free T4, LH/FSH) and formal formal visual field perimetry test."
        ),
    },
}


def find_serialized_models(models_dir: str = "models") -> List[Dict[str, Any]]:
    """
    Search the models directory for serialized Keras/TF models (.keras, .h5, SavedModel dirs).
    Returns a list of metadata dictionaries.
    """
    found = []
    base_path = Path(models_dir)
    if not base_path.exists():
        base_path.mkdir(parents=True, exist_ok=True)
        return found

    # Search for files with .keras, .h5, .hdf5 extensions
    for ext in ("*.keras", "*.h5", "*.hdf5"):
        for path in base_path.glob(ext):
            stat = path.stat()
            size_mb = round(stat.st_size / (1024 * 1024), 2)
            found.append({
                "path": str(path),
                "filename": path.name,
                "type": path.suffix.lower(),
                "size_mb": size_mb,
                "modified": stat.st_mtime,
            })

    # Search for SavedModel directories (containing saved_model.pb)
    for path in base_path.iterdir():
        if path.is_dir() and (path / "saved_model.pb").exists():
            stat = path.stat()
            found.append({
                "path": str(path),
                "filename": path.name + " (SavedModel)",
                "type": "saved_model_dir",
                "size_mb": 0.0,
                "modified": stat.st_mtime,
            })

    # Sort most recently modified first
    found.sort(key=lambda x: x["modified"], reverse=True)
    return found


def create_and_save_baseline_model(save_path: str = "models/baseline_model.keras") -> str:
    """
    Constructs and serializes a model with the exact architecture from model_training.ipynb.
    This enables immediate testing of the Streamlit app even before long training sessions finish.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3), name="mri_input"),
        tf.keras.layers.RandomRotation(0.03, name="aug_rot"),
        tf.keras.layers.RandomZoom(0.1, name="aug_zoom"),
        tf.keras.layers.RandomTranslation(0.05, 0.05, name="aug_trans"),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2d_1"),
        tf.keras.layers.MaxPooling2D(name="maxpool_1"),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2d_2"),
        tf.keras.layers.MaxPooling2D(name="maxpool_2"),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv2d_3"),
        tf.keras.layers.MaxPooling2D(name="maxpool_3"),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv2d_4"),
        tf.keras.layers.MaxPooling2D(name="maxpool_4"),
        tf.keras.layers.GlobalAveragePooling2D(name="gap"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_features"),
        tf.keras.layers.Dropout(0.3, name="dropout"),
        tf.keras.layers.Dense(len(CLASSES), activation="softmax", name="predictions"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.save(save_path)
    return save_path


def load_mri_model(model_path: str) -> tf.keras.Model:
    """
    Loads a serialized TensorFlow / Keras model from disk.
    """
    return tf.keras.models.load_model(model_path)


def preprocess_image(image_input: Any, target_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes an image (file-like object, numpy array, or PIL Image) and prepares it for model inference.
    Returns:
        rgb_display: (H, W, 3) uint8 numpy array in RGB format for visualization.
        batch_tensor: (1, 224, 224, 3) float32 numpy array normalized to [0, 1].
    """
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        rgb_original = np.array(pil_img)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            rgb_original = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.shape[2] == 4:
            rgb_original = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            rgb_original = image_input.copy()
    else:
        # File-like object (e.g. Streamlit UploadedFile)
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Unable to decode the uploaded image file.")
        rgb_original = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Resize to target shape (224, 224)
    resized = cv2.resize(rgb_original, target_size, interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1] as performed in make_ds() in model_training.ipynb
    normalized = resized.astype(np.float32) / 255.0
    batch_tensor = np.expand_dims(normalized, axis=0)

    return rgb_original, batch_tensor


def run_inference(model: tf.keras.Model, batch_tensor: np.ndarray) -> Dict[str, Any]:
    """
    Executes model inference and returns formatted prediction results.
    """
    raw_preds = model(batch_tensor, training=False)
    probabilities = raw_preds.numpy()[0] if hasattr(raw_preds, "numpy") else np.array(raw_preds)[0]

    top_idx = int(np.argmax(probabilities))
    top_class = CLASSES[top_idx]
    top_confidence = float(probabilities[top_idx])

    prob_dict = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}

    return {
        "predicted_class": top_class,
        "confidence": top_confidence,
        "probabilities": prob_dict,
        "class_index": top_idx,
        "class_info": CLASS_INFO.get(top_class, {}),
    }


def find_last_conv_layer(model: tf.keras.Model) -> Optional[str]:
    """
    Finds the name of the last Conv2D layer in the given model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # Fallback search inside nested models if any
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer.name
    return None


def generate_gradcam(
    model: tf.keras.Model,
    batch_tensor: np.ndarray,
    target_class_idx: Optional[int] = None,
    original_rgb: Optional[np.ndarray] = None,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a Grad-CAM heatmap highlighting regions the model focused on.
    Returns:
        heatmap_colored: (H, W, 3) colored heatmap.
        overlay: (H, W, 3) heatmap superimposed on original image.
    """
    last_conv_name = find_last_conv_layer(model)
    if not last_conv_name:
        raise ValueError("Could not find a Conv2D layer in the model for Grad-CAM.")

    # Target the last conv layer and the final output layer
    target_conv_layer = model.get_layer(last_conv_name)
    output_layer = model.layers[-1]

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[target_conv_layer.output, output_layer.output],
    )

    tf_tensor = tf.convert_to_tensor(batch_tensor)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf_tensor, training=False)
        if target_class_idx is None:
            target_class_idx = int(tf.argmax(predictions[0]))
        loss = predictions[:, target_class_idx]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        # Fallback if gradient could not be traced
        h, w = batch_tensor.shape[1], batch_tensor.shape[2]
        dummy_map = np.zeros((h, w), dtype=np.uint8)
        heatmap_colored = cv2.applyColorMap(dummy_map, colormap)
        return heatmap_colored, heatmap_colored

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap_np = heatmap.numpy()

    # Determine dimensions for overlay
    if original_rgb is not None:
        target_h, target_w = original_rgb.shape[0], original_rgb.shape[1]
        base_rgb = original_rgb.copy()
    else:
        target_h, target_w = 224, 224
        base_rgb = (batch_tensor[0] * 255).astype(np.uint8)

    # Resize heatmap to match base image
    resized_heatmap = cv2.resize(heatmap_np, (target_w, target_h))
    heatmap_uint8 = np.uint8(255 * resized_heatmap)

    # Apply chosen colormap
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap onto base image
    overlay = cv2.addWeighted(base_rgb, 1.0 - alpha, heatmap_rgb, alpha, 0)

    return heatmap_rgb, overlay


def get_available_sample_images(sample_roots: List[str] = ["data/processed", "data/raw"]) -> Dict[str, List[str]]:
    """
    Finds sample MRI images from the project data directories for quick 1-click testing.
    """
    samples: Dict[str, List[str]] = {cls: [] for cls in CLASSES}
    valid_exts = {".png", ".jpg", ".jpeg"}

    for root_str in sample_roots:
        root_path = Path(root_str)
        if not root_path.exists():
            continue

        for cls in CLASSES:
            cls_dir = root_path / cls
            if cls_dir.exists() and cls_dir.is_dir():
                found_files = [
                    str(f) for f in cls_dir.iterdir()
                    if f.suffix.lower() in valid_exts
                ]
                # Keep up to 20 samples per class
                samples[cls].extend(found_files[:20])

    return samples
