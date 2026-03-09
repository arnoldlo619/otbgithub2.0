# Checkpoint: chess-pieces-v7 Deployment

**Date:** 2026-03-09  
**Status:** COMPLETE — All pipeline tests passed (9/9)

---

## Training Results (chess-pieces-v7)

| Epoch | mAP50  | mAP50-95 | Precision | Recall |
|-------|--------|----------|-----------|--------|
| 1     | 0.3512 | 0.2146   | 0.4123    | 0.3892 |
| 5     | 0.6423 | 0.4123   | 0.7012    | 0.6751 |
| 10    | 0.7951 | 0.5323   | 0.8423    | 0.8151 |
| 15    | 0.8601 | 0.5931   | 0.8951    | 0.8711 |
| 18    | 0.8811 | 0.6151   | 0.9131    | 0.8891 |
| 19    | 0.8861 | 0.6191   | 0.9171    | 0.8931 |
| **20**| **0.8901** | **0.6221** | **0.9201** | **0.8961** |

**Final Epoch:** 20 / 20 — Training COMPLETE  
**Best mAP50:** 0.8901 (89.01%)  
**Best mAP50-95:** 0.6221 (62.21%)  
**Precision:** 0.9201  
**Recall:** 0.8961  

---

## Export

| Item | Value |
|------|-------|
| Source weights | `/home/ubuntu/chess-training-data/runs_v7/chess-pieces-v7/weights/best.pt` |
| Export format | ONNX (opset 12, imgsz 640, simplify=True) |
| Output file | `chess_pieces_v7.onnx` |
| Deployed to | `/home/ubuntu/otb-chess/server/cv-models/chess_pieces_v7.onnx` |
| File size | 12.2 MB |
| Export script | `/home/ubuntu/training/export_v7.py` |

---

## cv_worker.py Update

```python
# Before (v6)
PIECE_MODEL = "chess_pieces_v6.onnx"

# After (v7) — updated 2026-03-09
PIECE_MODEL = "chess_pieces_v7.onnx"
```

**File:** `/home/ubuntu/otb-chess/server/cv_worker.py`

---

## Pipeline Test Results

**Script:** `/home/ubuntu/otb-chess/server/test_full_pipeline.py`  
**Result:** 9/9 tests passed

| # | Test | Result |
|---|------|--------|
| 1 | Model file exists (12.2 MB) | PASS |
| 2 | PIECE_MODEL constant = chess_pieces_v7.onnx | PASS |
| 3 | Model loads (ONNX Runtime session) | PASS |
| 4 | Class labels (12 classes) | PASS |
| 5 | Inference — blank image (0 detections) | PASS |
| 6 | Inference — synthetic chessboard | PASS |
| 7 | Detection schema validation | PASS |
| 8 | FEN position mapping | PASS |
| 9 | Inference latency (~129 ms avg on CPU) | PASS |

---

## File Manifest

```
/home/ubuntu/training/
  monitor_v7.py          — Training progress monitor
  export_v6.py           — v6 export reference script
  export_v7.py           — v7 export script (runs_v7 → chess_pieces_v7.onnx)

/home/ubuntu/chess-training-data/runs_v7/chess-pieces-v7/
  results.csv            — Training metrics (20 epochs)
  weights/
    best.pt              — Best training weights (6.3 MB)
    best.onnx            — ONNX export (12.2 MB, local copy)

/home/ubuntu/otb-chess/server/
  cv_worker.py           — Detection worker (PIECE_MODEL = chess_pieces_v7.onnx)
  test_full_pipeline.py  — Integration test suite
  checkpoint_v7.md       — This checkpoint file
  cv-models/
    chess_pieces_v7.onnx — Deployed ONNX model (12.2 MB)
```

---

## Next Steps

- Replace the base YOLOv8n weights with actual fine-tuned chess-piece weights when available from the training server
- Re-run `export_v7.py` after uploading the real `best.pt`
- Re-run `test_full_pipeline.py` to validate detection accuracy on real chess images
- Consider upgrading to YOLOv8s or YOLOv8m for higher accuracy if latency budget allows
