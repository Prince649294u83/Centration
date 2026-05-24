# Pupil-Limbus-Detector: Complete Project Analysis

## Table of Contents
1. [Project Overview](#project-overview)
2. [Complete Directory Structure](#complete-directory-structure)
3. [Root Level Files](#root-level-files)
4. [pupil_tracking Package](#pupil_tracking-package)
5. [Scripts Directory](#scripts-directory)
6. [Models Directory](#models-directory)
7. [Clinical Data Directory](#clinical-data-directory)
8. [Logs Directory](#logs-directory)

---

## Project Overview

**Pupil-Limbus-detector** is a surgical-grade deep learning system for detecting and measuring the pupil (dark center) and limbus (iris-sclera boundary) in eye images. It combines:

- **U-Net segmentation** with ResNet-34 encoder for ML-based detection
- **Classical computer vision** as fallback (thresholding, contours)
- **RANSAC ellipse fitting** with automatic circle/ellipse selection
- **Kalman filtering** for temporal smoothing in video
- **Ring-aware adaptive pipeline** for docked vs pre-docked images
- **Grayscale handling** for IR camera inputs
- **Spatial calibration** converting pixels to millimeters

**Version:** 2.0.0 (package), v2.3 (GUI)

---

## Complete Directory Structure

```
Pupil-Limbus-detector-main/
│
├── [ROOT FILES]
│   ├── launch_gui.py                 (Main entry point - 1600+ lines)
│   ├── train_production.py           (Training pipeline)
│   ├── requirements.txt              (Dependencies)
│   ├── README.md                     (Documentation)
│   └── .gitignore                    (Git rules)
│
├── pupil_tracking/                   [MAIN PACKAGE]
│   ├── __init__.py                   (Package init, v2.0.0)
│   │
│   ├── core/                         [DETECTION ENGINE - 7 files]
│   │   ├── __init__.py               (Exports: UnifiedDetector, SmartContourFitter, etc.)
│   │   ├── detector.py               (UnifiedDetector - 2079 lines)
│   │   ├── smart_fitter.py           (SmartContourFitter - 1131 lines)
│   │   ├── ellipse_fitter.py         (EllipseFitter - 500 lines)
│   │   ├── confidence.py             (ConfidenceScorer - 513 lines)
│   │   ├── corneal_center.py         (CornealCenterCalculator - 401 lines)
│   │   ├── ring_detector.py          (RingDetector - 749 lines)
│   │   └── eye_roi_detector.py       (EyeROIDetector - 304 lines)
│   │
│   ├── ml/                           [MACHINE LEARNING - 12 files]
│   │   ├── __init__.py               (Exports: ONNXInference, InferenceBackend, etc.)
│   │   ├── architecture.py           (EyeSegmentationModel - 602 lines)
│   │   ├── dataset.py                (EyeSegmentationDataset - 1740 lines)
│   │   ├── trainer.py                (Trainer - 428 lines)
│   │   ├── losses.py                 (CompositeLoss - 683 lines)
│   │   ├── inference.py              (SegmentationInference - 1167 lines)
│   │   ├── fast_inference.py         (FastInference - 829 lines)
│   │   ├── onnx_inference.py         (ONNXInference - 476 lines)
│   │   ├── inference_backend.py      (InferenceBackend - 205 lines)
│   │   ├── postprocess.py            (Post-processing utilities - 691 lines)
│   │   ├── ring_classifier.py        (RingClassifierNet - 671 lines)
│   │   └── grayscale_augmentation.py (RandomGrayscaleConversion - 515 lines)
│   │
│   ├── preprocessing/                [IMAGE PREPROCESSING - 8 files]
│   │   ├── __init__.py               (Exports all preprocessors)
│   │   ├── grayscale_handler.py      (GrayscaleHandler - 754 lines)
│   │   ├── normalizer.py             (ImageNormalizer - 331 lines)
│   │   ├── reflection_removal.py     (ReflectionRemover - 345 lines)
│   │   ├── suction_ring_masker.py    (SuctionRingMasker - 413 lines)
│   │   ├── ring_aware.py             (RingAwarePreprocessor - 490 lines)
│   │   ├── red_light_filter.py       (RedLightFilter - 384 lines)
│   │   └── temporal_reflection_filter.py (TemporalReflectionFilter - 290 lines)
│   │
│   ├── video/                        [VIDEO PROCESSING - 5 files]
│   │   ├── __init__.py               (Empty)
│   │   ├── optimized_processor.py    (OptimizedVideoProcessor - 1375 lines)
│   │   ├── video_processor.py        (VideoProcessor - 446 lines)
│   │   ├── kalman_tracker.py         (EyeKalmanTracker - 317 lines)
│   │   └── temporal_smoother.py      (TemporalSmoother - 197 lines)
│   │
│   ├── interface/                    [GUI AND I/O - 4 files]
│   │   ├── __init__.py               (Empty)
│   │   ├── gui_app.py                (PupilTrackingGUI - 3242 lines)
│   │   ├── frame_recorder.py         (FrameRecorder - 367 lines)
│   │   └── theme.py                  (DarkTheme - 352 lines)
│   │
│   ├── calibration/                  [SPATIAL CALIBRATION - 2 files]
│   │   ├── __init__.py               (Empty)
│   │   └── spatial_calibration.py    (StabilizedCalibrator - 412 lines)
│   │
│   ├── annotation/                   [DATA ANNOTATION - 2 files]
│   │   ├── __init__.py               (Empty)
│   │   └── annotation_tool.py        (AnnotationTool - 419 lines)
│   │
│   ├── tests/                        [TEST SUITE - 4 files]
│   │   ├── __init__.py               (Empty)
│   │   ├── test_grayscale.py         (Grayscale tests - 803 lines)
│   │   ├── test_video_pipeline.py    (Video pipeline tests - 210 lines)
│   │   └── test_clinical_accuracy.py (Clinical accuracy tests - 443 lines)
│   │
│   └── utils/                        [UTILITIES - 4 files]
│       ├── __init__.py               (Empty)
│       ├── config.py                 (PupilTrackingConfig - 979 lines)
│       ├── types.py                  (Data types - 737 lines)
│       └── logger.py                 (AuditLogger - 142 lines)
│
├── scripts/                          [UTILITY SCRIPTS - 20 files]
│   ├── annotate_data.py              (GUI annotation launcher)
│   ├── annotate_live_video.py        (Live video annotation - 2480 lines)
│   ├── annotate_ring_data.py         (Ring presence labeling - 443 lines)
│   ├── generate_masks.py             (Mask generation - 424 lines)
│   ├── verify_data.py                (Data verification - 290 lines)
│   ├── check_files.py                (Project structure check - 133 lines)
│   ├── check_training_data.py        (Training data check - 258 lines)
│   ├── train_model.py                (Model training - 391 lines)
│   ├── train_ring_classifier.py      (Ring classifier training - 410 lines)
│   ├── run_epoch.py                  (Single epoch test - 68 lines)
│   ├── finetune_grayscale.py         (Grayscale fine-tuning - 936 lines)
│   ├── evaluate_ring_detection.py    (Ring detection evaluation - 574 lines)
│   ├── diagnose_detection.py         (Detection diagnostics - 223 lines)
│   ├── test_grayscale_detection.py   (RGB vs grayscale test - 977 lines)
│   ├── benchmark_fps.py              (FPS benchmarking - 298 lines)
│   ├── benchmark_video_speed.py      (Video speed benchmark - 660 lines)
│   ├── convert_to_onnx.py            (ONNX conversion - 644 lines)
│   ├── export_onnx.py                (Simplified ONNX export - 120 lines)
│   ├── process_video.py              (Video processing - 193 lines)
│   └── debug_single_image.py         (Single image debug - 334 lines)
│
├── models/                           [TRAINED MODELS]
│   ├── best_model.pth                (Main segmentation model - ~98 MB)
│   ├── ring_classifier.pth           (Ring classifier - ~2.5 MB)
│   └── checkpoint_meta.json          (Training metadata)
│
├── clinical_data/                    [TRAINING DATASET]
│   ├── annotations/
│   │   ├── annotations.json          (Main annotations)
│   │   ├── annotations_production.json
│   │   └── masks/                    (Binary mask images)
│   └── clean/
│       ├── eye_01.jpeg ... eye_14.jpeg (Eye images)
│       └── annotations/
│           ├── annotations.json
│           └── masks/
│
└── logs/                             [AUDIT LOGS]
    ├── session_YYYYMMDD_HHMMSS.log  (Human-readable logs)
    └── audit_YYYYMMDD_HHMMSS.jsonl  (JSON audit trails)
```

---

## Root Level Files

### 1. launch_gui.py (1600+ lines)
**Purpose:** Main application entry point with multiple processing modes

**Key Components:**
- `_convert_frame_for_display()` - Converts frames to grayscale for display when mode active
- `process_image()` - Single image analysis with detailed console output
- `process_video()` - Video file processing with optional optimized pipeline
- `process_camera()` - Live webcam feed processing
- `_draw_cli_overlay()` - Draws detection results (pupil=green, limbus=blue)
- `_draw_cli_overlay_from_dict()` - Draws from serialized results
- `_export_results_csv()` - CSV export for video results

**CLI Modes:**
```
python launch_gui.py                    # GUI (default)
python launch_gui.py gui                # Explicit GUI
python launch_gui.py image -i eye.jpg   # Single image
python launch_gui.py video -i vid.mp4   # Video file
python launch_gui.py camera             # Live camera
```

**Key CLI Arguments:**
- `--grayscale MODE` - off/auto/force
- `--ring-mode MODE` - auto/docked/pre_docked
- `--stride N` - Process every Nth frame
- `--fp16/--no-fp16` - FP16 half-precision
- `--compile/--no-compile` - torch.compile JIT
- `--roi/--no-roi` - ROI tracking

**GUI Controls:**
- `Q` - Quit
- `S` - Save snapshot
- `SPACE` - Pause/Resume
- `G` - Toggle grayscale mode
- `Ctrl+R` - Start/stop recording

---

### 2. train_production.py
**Purpose:** Training pipeline orchestrator that bridges annotation format to model training

**Key Functions:**
- Converts JSON annotations to training data
- Supports data augmentation with configurable copies per image
- Runs training with specified epochs and batch size
- Handles model checkpointing

**Usage:**
```bash
python train_production.py \
  --annotations clinical_data/annotations/annotations.json \
  --image-dir clinical_data/training_data/images \
  --mask-dir clinical_data/training_data/masks \
  --epochs 200 --batch-size 2 --copies-per-image 50
```

---

### 3. requirements.txt
**Purpose:** Python dependencies

```
torch >=1.12.0
torchvision >=0.13.0
segmentation-models-pytorch >=0.3.0
opencv-python >=4.6.0
Pillow >=9.0.0
albumentations >=1.3.0
scipy >=1.7.0
numpy >=1.21.0
scikit-learn >=1.0.0
matplotlib >=3.5.0
tqdm >=4.64.0
pytest >=6.0.0 (optional)
```

---

## pupil_tracking Package

### pupil_tracking/__init__.py (9 lines)
```python
"""
Surgical-grade pupil & limbus tracking system.
Modules are imported directly where needed, not eagerly here,
to avoid circular imports and to allow partial usage.
"""
__version__ = "2.0.0"
```

---

### pupil_tracking/core/ [Detection Engine]

#### core/__init__.py (58 lines)
**Exports:**
- `UnifiedDetector` - Main detection orchestrator
- `SmartContourFitter` - Adaptive circle/ellipse fitting
- `FitResult`, `FitType` - Fitting result types
- `smart_fit` - One-shot convenience function
- `CornealCenterCalculator` - Pupil-limbus offset (optional)
- `EllipseFitter` - Legacy ellipse fitting (optional)
- `EyeROIDetector` - ROI detection (optional)

Uses try/except for graceful degradation of optional imports.

---

#### core/detector.py (2079 lines)
**Class: UnifiedDetector**

**Purpose:** Production detection pipeline with 10-step process:

1. **Grayscale normalization** - Handles OFF/AUTO/FORCE modes
2. **Ring detection** - CNN classifier + heuristic fallback
3. **Adaptive preprocessing** - Ring-aware CLAHE
4. **ML segmentation** - U-Net inference
5. **Smart contour fitting** - Auto circle vs ellipse
6. **Ring-constrained filtering** - Spatial constraints
7. **Classical CV fallback** - If ML fails
8. **Cross-validation** - Pupil/limbus consistency
9. **Auto-calibration** - px to mm from limbus
10. **Quality grading** - SURGICAL/CLINICAL/RESEARCH/INSUFFICIENT

**Key Methods:**
- `detect(image, frame_number, source, force_mode)` - Main detection
- `detect_video_frame(image, frame_number, roi_x, roi_y)` - Video frame with ROI offset
- `detect_from_masks(image_bgr, pupil_mask, iris_mask, ...)` - From pre-computed masks
- `set_grayscale_mode(mode)` - Runtime grayscale toggle
- `init_video_mode(input_size, half_precision, device)` - Video optimization
- `calibrate_from_limbus(limbus, corneal_diameter_mm)` - Pixel-mm calibration
- `reset()` - Clear all state

**Helper Classes:**
- `_ONNXEngineWrapper` - Wraps ONNX inference to match interface
- `_DummyEngine` - Fallback when no ML backend

---

#### core/smart_fitter.py (1131 lines)
**Classes:**
- `FitType` (Enum): CIRCLE, ELLIPSE, FAILED
- `FitResult` (Dataclass): Complete fitting result with uncertainties
- `SmartContourFitter`: Automatic circle/ellipse selection

**Algorithms:**
- `_fit_circle_kasa()` - Kåsa algebraic least-squares circle fit
- `_fit_circle_taubin()` - Taubin unbiased circle fit
- `_fit_circle_hyper()` - Hyper circle fit (most accurate)
- `_ransac_circle()` - RANSAC for outlier rejection
- `_refine_contour_subpixel()` - Sub-pixel refinement (~0.05px accuracy)
- `_compute_gradient_weights()` - Gradient magnitude weighting

**Smart Selection Logic:**
1. Fit both circle and ellipse
2. Compare RMS residuals
3. If aspect ratio > 0.95 OR RMS diff < threshold → use circle
4. Otherwise use ellipse

---

#### core/ellipse_fitter.py (500 lines)
**Class: EllipseFitter** (Stateless, all class/static methods)

**Fallback Chain:**
1. RANSAC ellipse (≥ 10 points)
2. Direct ellipse via cv2.fitEllipse (≥ 5 points)
3. Huber-weighted iterative ellipse (≥ 5 points)
4. Algebraic circle Kåsa (≥ 3 points)
5. cv2.minEnclosingCircle (≥ 1 point, always succeeds)

**Key Functions:**
- `_sampson_distances()` - Sampson distance to ellipse
- `_quality_score()` - Combined quality in [0,1]
- `_build_result()` - Convert to FitResult dataclass

---

#### core/confidence.py (513 lines)
**Classes:**
- `QualityLevel` (Enum): EXCELLENT, GOOD, FAIR, POOR, UNUSABLE
- `ConfidenceScorer`: Computes detection confidence

**Scoring Components:**
- **Pupil confidence:** Circularity (40%), Area proportion (30%), Centrality (30%)
- **Limbus confidence:** Concentricity (40%), Size ratio (30%), Ring containment (30%)
- **Ring confidence:** Classifier (50%), Heuristic (30%), Segmentation (20%)
- **Overall confidence:** Pupil (1.0), Limbus (0.8), Ring (0.3)

---

#### core/corneal_center.py (401 lines)
**Key Concept:** Corneal center = limbus center (anatomical definition)

**Classes:**
- `SmoothedStateWriter` - Writes Kalman-filtered values back to results
- `CornealCenterCalculator` - Calculates pupil-limbus offset

**Output (CornealCenterResult):**
- `center_px` - Limbus center position
- `offset_px` - Pupil center - limbus center
- `offset_magnitude_px` - Euclidean distance
- `offset_angle_deg` - Direction from atan2
- mm conversions if calibrated

Normal offset: 0.1 - 0.5 mm (nasal, slightly inferior)

---

#### core/ring_detector.py (749 lines)
**Classes:**
- `RingStatus` (Enum): PRESENT, ABSENT, PARTIAL, UNCERTAIN
- `RingDetectionResult` (Dataclass): Status, confidence, contour, center, radius, mask
- `HeuristicRingDetector` - Traditional CV-based detection
- `RingDetector` - Combined CNN + heuristic

**Detection Strategy:**
1. CNN classifier first (~3 ms)
2. If confidence > threshold, accept
3. When PRESENT, run heuristic for geometry
4. If confidence low or no classifier, use heuristic
5. Merge with weights: classifier 65%, heuristic 35%

---

#### core/eye_roi_detector.py (304 lines)
**Class: EyeROIDetector**

**Strategy Hierarchy:**
1. Auto-detect if frame is already eye closeup → skip ROI
2. Use cached ROI → ~1 ms
3. Haar cascade face → eye detection → ~8 ms
4. Intensity-based dark-blob fallback → ~5 ms
5. Full frame as last resort → 0 ms

---

### pupil_tracking/ml/ [Machine Learning]

#### ml/__init__.py (35 lines)
**Exports:**
- `ONNXInference` - Production inference (always available)
- `InferenceBackend` - Factory for backend selection (always available)
- `SegmentationInference` - PyTorch inference (optional)
- `FastInference` - Fast PyTorch inference (optional)
- `EyeSegmentationModel` - Model definition (optional)

---

#### ml/architecture.py (602 lines)
**Class: EyeSegmentationModel**

**Architecture:**
- U-Net with ResNet-34 encoder (segmentation_models_pytorch)
- 3-class: background, pupil, iris
- 4-class: background, pupil, iris, suction_ring
- ~25M parameters
- Temperature scaling for probability calibration

**Key Methods:**
- `forward(x)` - Returns logits / temperature
- `predict_proba(x)` - Softmax probabilities
- `predict_classes(x)` - Argmax class indices
- `calibrate_temperature(val_loader)` - Post-hoc calibration
- `save(path)` / `load(path)` - Checkpoint I/O

**Factory Functions:**
- `create_model(num_classes, encoder, pretrained, device)`
- `load_model(path, device, num_classes)` - Robust loading with fallback
- `get_device(preference)` - Resolve device string

---

#### ml/dataset.py (1740 lines)
**Class: EyeSegmentationDataset**

**Key Features:**
- Loads images and annotations
- Generates masks from ellipse parameters or boundary points
- Supports 3-class and 4-class segmentation
- Heavy augmentation pipeline (spatial, pixel, blur, noise, compression, occlusion)
- Optional grayscale augmentation

**Key Functions:**
- `load_annotations(annotation_path, ring_labels_path)` - Parse JSON
- `generate_mask_from_annotation(image_shape, annotation, num_classes)` - Create masks
- `build_datasets(...)` - Build train/val datasets
- `split_by_images(image_ids, val_ratio, seed)` - Image-level split

---

#### ml/trainer.py (428 lines)
**Class: Trainer**

**Training Features:**
- Mixed-precision training with gradient scaling
- Cosine annealing LR schedule
- Early stopping based on validation IoU
- Temperature calibration after training
- Per-class IoU tracking

**Key Methods:**
- `train()` - Full training loop
- `_train_epoch(epoch)` - Single epoch
- `_val_epoch(epoch)` - Validation
- `_save_checkpoint(epoch, val_iou)` - Save best model

---

#### ml/losses.py (683 lines)
**Classes:**
- `DiceLoss` - Soft Dice coefficient loss
- `BoundaryLoss` - Distance-transform boundary loss
- `FocalLoss` - Down-weights easy pixels
- `WeightedCrossEntropyDiceLoss` - Combined weighted CE + Dice
- `CompositeLoss` - Full composite (CE + Dice + Boundary)

**Factory:**
- `create_loss(num_classes, class_weights, loss_type, use_focal, focal_gamma)`

---

#### ml/inference.py (1167 lines)
**Class: SegmentationInference**

**Features:**
- Robust model loading (tries multiple strategies)
- Multi-scale inference (448, 512, 640)
- Sub-pixel contour refinement
- Gradient-guided mask boundary refinement
- Edge alignment scoring
- Reflection and ring marker removal

**Key Methods:**
- `detect(image, frame_number, source)` - Main inference
- `_multiscale_segment(tensor, gray)` - Multi-scale averaging
- `_refine_contour_subpixel(gray, contour)` - Sub-pixel refinement
- `get_raw_mask(image_bgr)` - Raw predictions

---

#### ml/fast_inference.py (829 lines)
**Class: FastInference**

**Optimizations:**
- FP16 half-precision
- torch.compile JIT on CUDA
- Batch inference support
- Accuracy-first preprocessing

**Key Methods:**
- `warmup()` - JIT warm-up
- `segment(image_bgr)` - Get masks
- `detect(image_bgr)` - Get detection dict
- `infer_batch(images)` - Batch inference
- `detect_batch(images_bgr)` - Batch detection

---

#### ml/onnx_inference.py (476 lines)
**Classes:**
- `ONNXInference` - Production inference without PyTorch
- `ONNXRingClassifier` - Ring classification via ONNX

**Features:**
- Automatic provider selection (CUDA, CoreML, DirectML, CPU)
- Quantized model support
- Optimal thread count detection

---

#### ml/inference_backend.py (205 lines)
**Class: InferenceBackend** (Static factory)

**Methods:**
- `create(model_dir, prefer_onnx, ...)` - Create best available engine
- `create_ring_classifier(model_dir, prefer_onnx)` - Create ring classifier

**Priority:** ONNX first, PyTorch fallback

---

#### ml/postprocess.py (691 lines)
**Dataclass: RingSegmentationResult**

**Key Functions:**
- `mask_to_contours(binary_mask, min_area)` - Extract contours
- `contour_to_ellipse(contour, scale_x, scale_y)` - Fit ellipse
- `extract_ring_from_segmentation(prediction)` - Extract ring geometry
- `extract_contours_ring_aware(...)` - Ring-constrained extraction
- `clean_segmentation_mask(prediction)` - Morphological cleanup
- `validate_pupil_limbus_pair(pupil, limbus, ring)` - Cross-validation

---

#### ml/ring_classifier.py (671 lines)
**Classes:**
- `RingClassifierNet` - MobileNetV2 binary classifier
- `RingClassificationDataset` - Dataset for ring labels
- `RingClassifierTrainer` - Training loop

**Key Methods:**
- `predict(image, device)` - Single image prediction
- `predict_batch(images, device)` - Batch prediction

---

#### ml/grayscale_augmentation.py (515 lines)
**Classes:**
- `RandomGrayscaleConversion` - Albumentations transform
- `GrayscaleAwarePipeline` - Complete augmentation pipeline

Uses same `GrayscaleHandler` as inference for consistency.

---

### pupil_tracking/preprocessing/ [Image Preprocessing]

#### preprocessing/grayscale_handler.py (754 lines)
**Classes:**
- `GrayscaleMode` (Enum): AUTO, FORCE, OFF
- `GrayscaleInfo` (Dataclass): Detection info
- `GrayscaleHandler` - Main handler

**Key Methods:**
- `is_grayscale(image)` - Detect if grayscale
- `to_grayscale(image)` - Convert to grayscale
- `enhance_grayscale(image)` - CLAHE enhancement
- `to_model_input(image)` - Prepare for model (3-channel uint8)
- `get_quality_metrics(image)` - Contrast, SNR, etc.

---

#### preprocessing/normalizer.py (331 lines)
**Class: ImageNormalizer**

**Methods:**
- `normalize(image)` - Full normalization
- `fast_normalize(image)` - Video-optimized
- `_white_balance(image)` - Gray-world white balance
- `_apply_clahe(image)` - CLAHE on LAB luminance
- `_auto_gamma(image)` - Automatic gamma correction

---

#### preprocessing/reflection_removal.py (345 lines)
**Class: ReflectionRemover**

**Methods:**
- `remove(image)` - Detect and inpaint reflections
- `detect_only(image)` - Just detect, don't remove
- `_detect_reflections(image)` - HSV + blue/red channel detection

---

#### preprocessing/suction_ring_masker.py (413 lines)
**Classes:**
- `SuctionRingResult` (Dataclass): Ring geometry
- `SuctionRingMasker` - Detects and masks red LED markers

**Algorithm:**
- HSV red hue thresholding (two ranges for wrap-around)
- Contour filtering by area and circularity
- Ring geometry validation via least-squares fit
- Marker dilation and inpainting

---

#### preprocessing/ring_aware.py (490 lines)
**Classes:**
- `PreprocessingResult` (Dataclass): Processed image + metadata
- `RingAwarePreprocessor` - Different pipeline for docked/pre-docked
- `AdaptiveContourFilter` - Ring-aware contour filtering

**Pre-docked Pipeline:**
grayscale → median blur → CLAHE → eyelid suppression → normalization

**Docked Pipeline:**
ring masking → ROI inside ring → tighter CLAHE → specular suppression

---

#### preprocessing/red_light_filter.py (384 lines)
**Classes:**
- `RedLightFilter` - Detects and removes surgical red lights
- `AdaptiveRedLightFilter` - Auto-adjusts thresholds

**Algorithm:**
- Red channel thresholding
- Dominance offset (R > G + offset, R > B + offset)
- Temporal smoothing via EMA
- Optional inpainting

---

#### preprocessing/temporal_reflection_filter.py (290 lines)
**Classes:**
- `TemporalReflectionFilter` - Distinguishes persistent vs transient reflections
- `PupilRegionProtector` - Protects pupil from removal artifacts

**Algorithm:**
- Maintains history of reflection masks
- Computes stability map (present in enough frames)
- Morphological dilation

---

### pupil_tracking/video/ [Video Processing]

#### video/optimized_processor.py (1375 lines)
**Classes:**
- `VideoPreprocessor` - Batch preprocessing
- `FrameQualityChecker` - Blur/brightness check
- `_FrameReader` (Thread) - Decode-ahead reading
- `_OverlayRenderer` - Draws detection overlay
- `OptimizedVideoProcessor` - Main processor
- `TrackingQuality` - Quality enum
- `FrameResult` (dict subclass) - Frame result
- `AsyncCapture` (Thread) - Async camera capture

**Key Methods:**
- `process_frame(frame, frame_number)` - Single frame
- `process_video(video_path, ...)` - Full video
- `process_stream(camera_id, ...)` - Live camera
- `save_results_json(path)` - Export results

---

#### video/video_processor.py (446 lines)
**Class: VideoProcessor** - Simpler single-frame processor

**Methods:**
- `process_file(video_path, ...)` - Video file
- `process_stream(camera_id, ...)` - Live stream
- `export_csv(path)` / `export_json(path)` - Export

---

#### video/kalman_tracker.py (317 lines)
**Classes:**
- `EllipseKalmanFilter` - Single ellipse Kalman filter
- `EyeKalmanTracker` - Dual filter for pupil + limbus

**State Vector:** [cx, cy, semi_major, semi_minor, angle_sin, angle_cos, vx, vy]

**Features:**
- Constant velocity model
- Confidence-weighted measurement noise
- Carry-forward with decay on detection failure
- Blink detection

---

#### video/temporal_smoother.py (197 lines)
**Class: TemporalSmoother** - Lightweight Kalman filter

**State:** pupil_x, pupil_y, pupil_radius, limbus_x, limbus_y, limbus_radius

---

### pupil_tracking/interface/ [GUI and I/O]

#### interface/gui_app.py (3242 lines)
**Class: PupilTrackingGUI** - Full-featured Tkinter GUI

**Features:**
- Image, video, and camera modes
- Real-time parameter adjustment
- Grayscale mode toggle (keyboard: G)
- Video recording with FrameRecorder
- Settings panel
- Measurements panel
- Export to CSV/JSON/snapshot
- Dark theme

**Key Methods:**
- `_open_image()` / `_open_video()` / `_start_camera()`
- `_toggle_grayscale()` - Cycle OFF→AUTO→FORCE
- `_start_recording()` / `_stop_recording()`
- `_export_csv()` / `_export_json()` / `_export_snapshot()`
- `_draw_overlay()` - Renders detection results

---

#### interface/frame_recorder.py (367 lines)
**Class: FrameRecorder** - Thread-safe video recording

**Features:**
- Dedicated writer thread
- Frame queue with proper timing
- Codec selection with fallback
- Status callbacks for UI

---

#### interface/theme.py (352 lines)
**Classes:**
- `Colors` - Color constants
- `DarkTheme` - Applies dark theme to ttk widgets

---

### pupil_tracking/calibration/ [Spatial Calibration]

#### calibration/spatial_calibration.py (412 lines)
**Classes:**
- `SpatialCalibrator` - Multiple calibration strategies
- `StabilizedCalibrator` - EMA smoothing + outlier rejection

**Calibration Methods:**
- From limbus diameter (average 11.5 mm)
- From suction ring diameter (known sizes)
- Manual calibration
- Confidence-weighted consensus

---

### pupil_tracking/annotation/ [Data Annotation]

#### annotation/annotation_tool.py (419 lines)
**Class: AnnotationTool** - Tkinter annotation GUI

**Features:**
- Click to add points for pupil, limbus, or ring
- Ellipse fitting via cv2.fitEllipse
- Save/load annotations in JSON
- Boundary points and ellipse parameters

---

### pupil_tracking/tests/ [Test Suite]

#### tests/test_grayscale.py (803 lines)
**Test Classes:**
- TestGrayscaleMode, TestIsGrayscale, TestToGrayscale
- TestEnhanceGrayscale, TestToModelInput
- TestQualityMetrics, TestConstruction
- TestThreadSafety, TestEdgeCases, TestEndToEnd

#### tests/test_video_pipeline.py (210 lines)
**Tests:**
- test_roi_detector
- test_temporal_smoother
- test_fast_inference
- test_unified_video_mode
- test_full_pipeline

#### tests/test_clinical_accuracy.py (443 lines)
**Test Class:** TestClinicalDetection - Parametrized per image

---

### pupil_tracking/utils/ [Utilities]

#### utils/config.py (979 lines)
**Config Classes (Nested Dataclasses):**
- `ModelConfig` - ML model parameters
- `DetectionConfig` - Detection thresholds
- `FittingConfig` - Ellipse fitting parameters
- `VideoConfig` - Video processing parameters
- `CalibrationConfig` - Spatial calibration parameters
- `PathConfig` - Filesystem paths
- `TrainingConfig` - Training hyperparameters
- `RingClassifierConfig` - Ring classifier parameters
- `RingHeuristicConfig` - Heuristic ring detection
- `RingPreprocessingConfig` - Ring-aware preprocessing
- `DockedDetectionConfig` / `PreDockedDetectionConfig`
- `GrayscaleConfig` - Grayscale parameters
- `MeasurementStabilizationConfig` - EMA parameters
- `SubPixelConfig` - Sub-pixel refinement
- `RingConfig` - Aggregates ring configs
- `PupilTrackingConfig` - Top-level config

**Functions:**
- `get_config()` - Global config singleton
- `set_config(config)` - Set config
- `reset_config()` - Reset to defaults

---

#### utils/types.py (737 lines)
**Enums:**
- `DetectionQuality`: SURGICAL, CLINICAL, RESEARCH, INSUFFICIENT, NO_DETECTION
- `DetectionMethod`: ML, CLASSICAL, HYBRID, KALMAN, CARRY_FORWARD
- `QualityFlag`: GOOD, MARGINAL, POOR, NO_DETECTION

**Dataclasses:**
- `AnatomicalLimits` - Hard anatomical constraints
- `EllipseParams` - Fitted ellipse with uncertainties
- `PupilDetection` - Complete pupil result
- `LimbusDetection` - Complete limbus result
- `CornealCenterResult` - Offset calculation
- `CalibrationInfo` - Pixel-mm calibration
- `FrameMetadata` - Per-frame bookkeeping
- `EyeDetectionResult` - Top-level result container
- `FitResult` - Geometric fitting result

**Functions:**
- `assign_quality_grade(confidence)` - Maps confidence to quality

---

#### utils/logger.py (142 lines)
**Class: AuditLogger** - Thread-safe JSON audit logging

**Methods:**
- `log_detection(result)` - Log detection
- `log_alert(message)` - Log alert
- `info()`, `warning()`, `error()`, `debug()` - Standard logging

---

## Scripts Directory

### annotate_data.py (12 lines)
Launcher for annotation GUI.
```bash
python scripts/annotate_data.py
```

### annotate_live_video.py (2480 lines)
Live video annotation with edge snapping and multiple ellipse fitting methods.

**Subcommands:**
```bash
python scripts/annotate_live_video.py annotate video.mp4
python scripts/annotate_live_video.py generate-masks
python scripts/annotate_live_video.py train --epochs 50
python scripts/annotate_live_video.py check
```

### annotate_ring_data.py (443 lines)
Keyboard-based ring presence labeling (R=present, N=absent, P=partial).
```bash
python scripts/annotate_ring_data.py --image-dir clinical_data/training_data/images
```

### benchmark_fps.py (298 lines)
Benchmarks inference speed at different resolutions with FP16/FP32.
```bash
python scripts/benchmark_fps.py --model models/best_model.pth --mode engine
```

### benchmark_video_speed.py (660 lines)
Benchmarks video processing pipeline (decode, preprocess, infer, batch).
```bash
python scripts/benchmark_video_speed.py --synthetic
```

### check_files.py (133 lines)
Verifies required project files exist.
```bash
python scripts/check_files.py
```

### check_training_data.py (258 lines)
Checks training data quality (mask validity, class distribution).
```bash
python scripts/check_training_data.py
```

### convert_to_onnx.py (644 lines)
Converts PyTorch models to ONNX with validation and quantization.
```bash
python scripts/convert_to_onnx.py
```

### debug_single_image.py (334 lines)
Tests multiple detection methods on single image.
```bash
python scripts/debug_single_image.py clinical_data/clean/eye_01.jpeg
```

### diagnose_detection.py (223 lines)
Generates diagnostic visualizations (masks, heatmaps, overlays).
```bash
python scripts/diagnose_detection.py --input clinical_data/clean/eye_06.jpeg
```

### evaluate_ring_detection.py (574 lines)
Evaluates ring detection accuracy against ground truth.
```bash
python scripts/evaluate_ring_detection.py --image-dir images --labels ring_labels.json
```

### export_onnx.py (120 lines)
Simplified ONNX export with verification.
```bash
python scripts/export_onnx.py --model models/best_model.pth --verify
```

### finetune_grayscale.py (936 lines)
Fine-tunes model for grayscale robustness with RandomGrayscaleConversion.
```bash
python scripts/finetune_grayscale.py --grayscale-prob 0.3
```

### generate_masks.py (424 lines)
Generates mask images from annotations.
```bash
python scripts/generate_masks.py
```

### process_video.py (193 lines)
Video processing with OptimizedVideoProcessor.
```bash
python scripts/process_video.py --input video.mp4 --output result.mp4
```

### run_epoch.py (68 lines)
Runs single training epoch without augmentation.
```bash
python scripts/run_epoch.py
```

### test_grayscale_detection.py (977 lines)
Compares RGB vs grayscale detection results.
```bash
python scripts/test_grayscale_detection.py --csv report.csv --visualize
```

### train_model.py (391 lines)
Main model training with configurable loss and architecture.
```bash
python scripts/train_model.py --epochs 300 --num-classes 4 --device cuda
```

### train_ring_classifier.py (410 lines)
Trains MobileNetV2 ring classifier.
```bash
python scripts/train_ring_classifier.py --image-dir images --labels ring_labels.json
```

### verify_data.py (290 lines)
Verifies annotation completeness and anatomical plausibility.
```bash
python scripts/verify_data.py
```

---

## Models Directory

### best_model.pth (~98 MB)
U-Net with ResNet-34 encoder, trained on clinical images.
Checkpoint metadata: epoch 5, val_iou=0.9578.

### ring_classifier.pth (~2.5 MB)
MobileNetV2 binary classifier for ring presence.

### checkpoint_meta.json
Training metadata (epoch, val_iou, losses).

---

## Clinical Data Directory

### annotations/annotations.json
Main annotation file in project format.

### annotations/masks/
Binary mask images (PNG) for training.

### clean/
Eye images for validation and testing.

---

## Logs Directory

### session_YYYYMMDD_HHMMSS.log
Human-readable session logs.

### audit_YYYYMMDD_HHMMSS.jsonl
JSON Lines format for machine parsing.

---

## Key Technical Innovations

1. **Ring-Aware Pipeline** - Different preprocessing for docked vs pre-docked
2. **Smart Contour Fitting** - Automatic circle vs ellipse based on residuals
3. **Stabilized Calibration** - EMA smoothing + outlier rejection
4. **Fast Inference** - FP16 + torch.compile for 50-100 FPS
5. **Grayscale Mode** - Optimized for IR cameras with CLAHE
6. **Sub-pixel Refinement** - <0.05px accuracy via gradient analysis
7. **Fallback Chain** - ML → RANSAC → algebraic → enclosing circle
8. **Kalman Temporal Smoothing** - Dual filter for pupil + limbus
9. **Adaptive Resolution** - 192-512px based on content
10. **Quality Grades** - SURGICAL (≥0.75), CLINICAL (≥0.55), RESEARCH (≥0.30)

---

## Installation & Usage

```bash
# Install
cd Pupil-Limbus-detector-main
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run GUI
python launch_gui.py

# Process image
python launch_gui.py image -i eye.jpg

# Process video
python launch_gui.py video -i video.mp4 -o result.mp4

# Live camera
python launch_gui.py camera

# Train model
python scripts/train_model.py --epochs 300 --device cuda
```

---

*Generated for AI agent understanding of the Pupil-Limbus-detector project.*
