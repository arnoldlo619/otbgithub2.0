#!/usr/bin/env python3.11
"""
test_full_pipeline.py — Full pipeline integration test for chess piece detection v7.

Tests the complete pipeline:
  1. Model file existence check
  2. ONNX model loading via cv_worker
  3. Inference on a synthetic chessboard image
  4. Detection output validation
  5. FEN position mapping
  6. Performance benchmark (inference latency)

Run with: python3.11 test_full_pipeline.py
"""

import sys
import time
import traceback
import numpy as np
import cv2
from pathlib import Path

# Add server directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cv_worker import (
    PIECE_MODEL,
    MODEL_PATH,
    detect_pieces,
    detections_to_fen_pieces,
    load_model,
    PIECE_CLASSES,
)

# ─── Test Utilities ───────────────────────────────────────────────────────────

PASS = "  [PASS]"
FAIL = "  [FAIL]"
INFO = "  [INFO]"

results = []


def test(name: str, fn):
    """Run a test function and record pass/fail."""
    print(f"\n  Running: {name}")
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, None))
    except AssertionError as e:
        print(f"{FAIL} {name} — AssertionError: {e}")
        results.append((name, False, str(e)))
    except Exception as e:
        print(f"{FAIL} {name} — Exception: {e}")
        traceback.print_exc()
        results.append((name, False, str(e)))


# ─── Synthetic Image Helpers ─────────────────────────────────────────────────

def make_blank_image(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def make_chessboard_image(size=480):
    """Generate a simple 8×8 chessboard pattern."""
    board = np.zeros((size, size, 3), dtype=np.uint8)
    sq = size // 8
    for r in range(8):
        for c in range(8):
            color = (240, 240, 200) if (r + c) % 2 == 0 else (100, 140, 80)
            board[r*sq:(r+1)*sq, c*sq:(c+1)*sq] = color
    return board


# ─── Tests ───────────────────────────────────────────────────────────────────

def test_model_file_exists():
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"{INFO} Model: {MODEL_PATH.name}  ({size_mb:.1f} MB)")
    assert size_mb > 1.0, f"Model file suspiciously small: {size_mb:.2f} MB"


def test_piece_model_constant():
    assert PIECE_MODEL == "chess_pieces_v7.onnx", (
        f"PIECE_MODEL should be 'chess_pieces_v7.onnx', got '{PIECE_MODEL}'"
    )
    print(f"{INFO} PIECE_MODEL = {PIECE_MODEL}")


def test_model_loads():
    session = load_model()
    assert session is not None, "ONNX session is None"
    inputs  = session.get_inputs()
    outputs = session.get_outputs()
    assert len(inputs)  > 0, "No inputs in ONNX model"
    assert len(outputs) > 0, "No outputs in ONNX model"
    print(f"{INFO} Input:  {inputs[0].name}  shape={inputs[0].shape}")
    print(f"{INFO} Output: {outputs[0].name} shape={outputs[0].shape}")


def test_inference_blank_image():
    img = make_blank_image()
    detections = detect_pieces(img)
    assert isinstance(detections, list), "detect_pieces must return a list"
    print(f"{INFO} Blank image → {len(detections)} detections (expected 0)")


def test_inference_chessboard_image():
    img = make_chessboard_image(size=640)
    detections = detect_pieces(img)
    assert isinstance(detections, list), "detect_pieces must return a list"
    print(f"{INFO} Chessboard image → {len(detections)} detections")
    for det in detections[:3]:
        print(f"{INFO}   {det['class_name']}  conf={det['confidence']:.3f}  bbox={[round(v,1) for v in det['bbox']]}")


def test_detection_schema():
    """Verify detection dict has required keys and valid values."""
    img = make_chessboard_image(size=640)
    detections = detect_pieces(img)
    for det in detections:
        assert "class_id"   in det, "Missing 'class_id'"
        assert "class_name" in det, "Missing 'class_name'"
        assert "confidence" in det, "Missing 'confidence'"
        assert "bbox"       in det, "Missing 'bbox'"
        assert 0.0 <= det["confidence"] <= 1.0, f"Confidence out of range: {det['confidence']}"
        assert len(det["bbox"]) == 4, f"bbox must have 4 values, got {len(det['bbox'])}"
        assert det["class_name"] in PIECE_CLASSES, f"Unknown class: {det['class_name']}"
    print(f"{INFO} Schema validated for {len(detections)} detections")


def test_fen_mapping():
    """Test that detections_to_fen_pieces returns valid FEN chars."""
    # Simulate a detection at the center of the board
    mock_detections = [
        {"class_id": 0, "class_name": "white-king",   "confidence": 0.95, "bbox": [220, 220, 260, 260]},
        {"class_id": 6, "class_name": "black-king",   "confidence": 0.93, "bbox": [380, 380, 420, 420]},
        {"class_id": 5, "class_name": "white-pawn",   "confidence": 0.88, "bbox": [100, 100, 140, 140]},
    ]
    board_bbox = (0, 0, 640, 640)
    positions = detections_to_fen_pieces(mock_detections, board_bbox)
    assert isinstance(positions, dict), "detections_to_fen_pieces must return a dict"
    for square, fen_char in positions.items():
        assert len(square) == 2, f"Invalid square name: {square}"
        assert square[0] in "abcdefgh", f"Invalid file: {square[0]}"
        assert square[1] in "12345678", f"Invalid rank: {square[1]}"
        assert fen_char in "KQRBNPkqrbnp?", f"Invalid FEN char: {fen_char}"
    print(f"{INFO} FEN positions: {positions}")


def test_inference_latency():
    """Benchmark inference latency — should be under 5 seconds on CPU."""
    img = make_chessboard_image(size=640)
    N   = 3
    start = time.time()
    for _ in range(N):
        detect_pieces(img)
    elapsed = time.time() - start
    avg_ms  = (elapsed / N) * 1000
    print(f"{INFO} Avg inference latency: {avg_ms:.1f} ms over {N} runs")
    assert avg_ms < 5000, f"Inference too slow: {avg_ms:.0f} ms (limit: 5000 ms)"


def test_class_labels():
    """Verify all 12 chess piece classes are defined."""
    assert len(PIECE_CLASSES) == 12, f"Expected 12 classes, got {len(PIECE_CLASSES)}"
    expected_colors = {"white", "black"}
    for cls in PIECE_CLASSES:
        color = cls.split("-")[0]
        assert color in expected_colors, f"Unexpected color prefix in class: {cls}"
    print(f"{INFO} Classes: {PIECE_CLASSES}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  Full Pipeline Test — chess_pieces_v7.onnx")
    print("=" * 70)

    test("1. Model file exists",          test_model_file_exists)
    test("2. PIECE_MODEL constant",       test_piece_model_constant)
    test("3. Model loads (ONNX session)", test_model_loads)
    test("4. Class labels (12 classes)",  test_class_labels)
    test("5. Inference — blank image",    test_inference_blank_image)
    test("6. Inference — chessboard",     test_inference_chessboard_image)
    test("7. Detection schema",           test_detection_schema)
    test("8. FEN position mapping",       test_fen_mapping)
    test("9. Inference latency",          test_inference_latency)

    # ─── Summary ─────────────────────────────────────────────────────────────
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total  = len(results)

    print("\n" + "=" * 70)
    print(f"  Test Results: {passed}/{total} passed  |  {failed} failed")
    print("=" * 70)

    for name, ok, err in results:
        status = "PASS" if ok else "FAIL"
        line   = f"  [{status}] {name}"
        if err:
            line += f"\n         Error: {err}"
        print(line)

    print("=" * 70 + "\n")

    if failed > 0:
        print(f"  ✗ {failed} test(s) failed. Review errors above.\n")
        sys.exit(1)
    else:
        print("  ✓ All tests passed. Pipeline is operational.\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
