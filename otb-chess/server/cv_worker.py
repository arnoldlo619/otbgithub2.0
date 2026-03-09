#!/usr/bin/env python3.11
"""
cv_worker.py — Chess piece detection worker using ONNX inference.

Loads the deployed ONNX model and provides functions for:
  - Detecting chess pieces in a given image
  - Mapping detections to board squares
  - Returning FEN-compatible piece positions

Model versions:
  v6: chess_pieces_v6.onnx  (previous)
  v7: chess_pieces_v7.onnx  (current — updated 2026-03-09)
"""

import os
import cv2
import numpy as np
from pathlib import Path

# ─── Model Configuration ────────────────────────────────────────────────────

# PIECE_MODEL: set to the current deployed ONNX model filename
PIECE_MODEL = "chess_pieces_v7.onnx"

CV_MODELS_DIR = Path(__file__).parent / "cv-models"
MODEL_PATH    = CV_MODELS_DIR / PIECE_MODEL

# ─── Class Labels ────────────────────────────────────────────────────────────

# Chess piece classes (12 classes: 6 piece types × 2 colors)
PIECE_CLASSES = [
    "white-king",   "white-queen",  "white-rook",
    "white-bishop", "white-knight", "white-pawn",
    "black-king",   "black-queen",  "black-rook",
    "black-bishop", "black-knight", "black-pawn",
]

# FEN character mapping
FEN_MAP = {
    "white-king":   "K",
    "white-queen":  "Q",
    "white-rook":   "R",
    "white-bishop": "B",
    "white-knight": "N",
    "white-pawn":   "P",
    "black-king":   "k",
    "black-queen":  "q",
    "black-rook":   "r",
    "black-bishop": "b",
    "black-knight": "n",
    "black-pawn":   "p",
}

# ─── Inference Parameters ────────────────────────────────────────────────────

INPUT_SIZE   = 640
CONF_THRESH  = 0.35
IOU_THRESH   = 0.45

# ─── Model Loader ────────────────────────────────────────────────────────────

_session = None  # Cached ONNX Runtime session


def load_model():
    """Load the ONNX model into an ONNX Runtime session (cached)."""
    global _session
    if _session is not None:
        return _session

    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {MODEL_PATH}\n"
            f"Run export_v7.py to generate the model."
        )

    print(f"[cv_worker] Loading model: {MODEL_PATH}")
    providers = ["CPUExecutionProvider"]
    _session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
    print(f"[cv_worker] Model loaded: {PIECE_MODEL}")
    return _session


# ─── Pre/Post Processing ─────────────────────────────────────────────────────

def preprocess(image: np.ndarray) -> tuple[np.ndarray, float, tuple]:
    """Resize and normalize image for YOLO inference."""
    h, w = image.shape[:2]
    scale = INPUT_SIZE / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))
    padded  = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    blob = padded.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))  # HWC → CHW
    blob = np.expand_dims(blob, axis=0)   # CHW → NCHW
    return blob, scale, (new_w, new_h)


def postprocess(output: np.ndarray, scale: float, orig_shape: tuple,
                conf_thresh: float = CONF_THRESH,
                iou_thresh:  float = IOU_THRESH) -> list[dict]:
    """
    Parse YOLO output tensor and return filtered detections.

    Returns a list of dicts with keys:
      class_id, class_name, confidence, bbox (x1, y1, x2, y2 in original coords)
    """
    # output shape: (1, num_classes+4, num_anchors)
    preds = output[0]  # (num_classes+4, num_anchors)
    preds = preds.T    # (num_anchors, num_classes+4)

    boxes, scores, class_ids = [], [], []

    for pred in preds:
        cx, cy, bw, bh = pred[:4]
        class_probs = pred[4:]
        class_id    = int(np.argmax(class_probs))
        confidence  = float(class_probs[class_id])

        if confidence < conf_thresh:
            continue

        # Convert center-format to corner-format, undo scale
        x1 = (cx - bw / 2) / scale
        y1 = (cy - bh / 2) / scale
        x2 = (cx + bw / 2) / scale
        y2 = (cy + bh / 2) / scale

        boxes.append([x1, y1, x2, y2])
        scores.append(confidence)
        class_ids.append(class_id)

    if not boxes:
        return []

    # NMS
    boxes_np  = np.array(boxes)
    scores_np = np.array(scores)
    indices   = cv2.dnn.NMSBoxes(
        boxes_np.tolist(), scores_np.tolist(), conf_thresh, iou_thresh
    )

    detections = []
    for i in (indices.flatten() if len(indices) > 0 else []):
        class_id = class_ids[i]
        name     = PIECE_CLASSES[class_id] if class_id < len(PIECE_CLASSES) else f"class_{class_id}"
        detections.append({
            "class_id":   class_id,
            "class_name": name,
            "confidence": float(scores_np[i]),
            "bbox":       [float(v) for v in boxes_np[i]],
        })

    return detections


# ─── Public API ──────────────────────────────────────────────────────────────

def detect_pieces(image: np.ndarray) -> list[dict]:
    """
    Run chess piece detection on an image.

    Args:
        image: BGR numpy array (e.g., from cv2.imread)

    Returns:
        List of detection dicts with class_name, confidence, bbox.
    """
    session = load_model()
    blob, scale, padded_size = preprocess(image)

    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: blob})

    detections = postprocess(outputs[0], scale, image.shape[:2])
    return detections


def detections_to_fen_pieces(detections: list[dict], board_bbox: tuple) -> dict:
    """
    Map detections to board squares and return FEN piece positions.

    Args:
        detections: List from detect_pieces()
        board_bbox: (x1, y1, x2, y2) of the chessboard in the image

    Returns:
        Dict mapping square names (e.g., 'e4') to FEN piece chars (e.g., 'P')
    """
    bx1, by1, bx2, by2 = board_bbox
    board_w = bx2 - bx1
    board_h = by2 - by1
    square_w = board_w / 8
    square_h = board_h / 8

    positions = {}
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        col = int((cx - bx1) / square_w)
        row = int((cy - by1) / square_h)

        if 0 <= col <= 7 and 0 <= row <= 7:
            file_char = chr(ord('a') + col)
            rank_char = str(8 - row)
            square    = f"{file_char}{rank_char}"
            fen_char  = FEN_MAP.get(det["class_name"], "?")
            positions[square] = fen_char

    return positions


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[cv_worker] PIECE_MODEL = {PIECE_MODEL}")
    print(f"[cv_worker] MODEL_PATH  = {MODEL_PATH}")
    print(f"[cv_worker] Model exists: {MODEL_PATH.exists()}")

    # Quick smoke test with a blank image
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        results = detect_pieces(dummy)
        print(f"[cv_worker] Smoke test passed — {len(results)} detections on blank image (expected 0)")
    except Exception as e:
        print(f"[cv_worker] Smoke test error: {e}")
