# Pupil Tracking & Limbus Detector

A comprehensive deep learning application for accurate pupil and iris (limbus) detection in eye images using advanced computer vision techniques. This project combines traditional image processing with modern deep learning to provide robust eye feature extraction for clinical and research applications.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Project Structure & File Guide](#project-structure--file-guide)
6. [Quick Start Guide](#quick-start-guide)
7. [Main Entry Points](#main-entry-points)
8. [All Scripts & Commands](#all-scripts--commands)
9. [Data Annotation Workflow](#data-annotation-workflow)
10. [Model Training](#model-training)
11. [Video & Real-time Processing](#video--real-time-processing)
12. [Testing & Validation](#testing--validation)
13. [Configuration](#configuration)
14. [Troubleshooting](#troubleshooting)
15. [Technical Architecture](#technical-architecture)

---

## Overview

This project provides an end-to-end solution for pupil and limbus detection in eye images. It features:

- **Multiple Detection Strategies**: Combines traditional image processing with deep learning
- **Deep Learning Architecture**: U-Net with ResNet-34 encoder for precise segmentation
- **Real-time Processing**: Optimized for video and camera feeds
- **Interactive GUI**: User-friendly interface for testing and annotation
- **Clinical-Grade Quality Control**: Built-in validation and audit logging

### Key Components

1. **Pupil Detection**: Identifies the dark circular region of the pupil
2. **Limbus Detection**: Detects the boundary between iris and sclera
3. **Quality Assessment**: Evaluates detection confidence and quality
4. **Geometric Fitting**: Fits ellipses to detected contours
5. **Real-time Processing**: Handles video streams and camera feeds

### Project Architecture

The project is organized into a main application module (`pupil_tracking/`) that handles all detection and processing logic, along with command-line entry points and training utilities:

```
pupil_tracking/
├── Core Detection (detection.py, preprocessing.py)
├── Machine Learning (ml/ - model architecture, training, inference)
├── Real-time Processing (video/, run_realtime.py)
├── Annotation Tools (annotation/)
├── Calibration & Metrics (calibration/, core/)
├── Logging & Configuration (logger.py, utils/)
└── Interface Layers (image_interface.py, clinical_debug.py)

Entry Points:
├── launch_gui.py           - Main application GUI
├── check_training_data.py  - Data validation utility
├── debug_single_image.py   - Single image debugging
└── scripts/                - Training, video processing, export utilities
```

---

## Features

### Core Detection Features
- ✅ **Multi-Strategy Detection**: Adaptive thresholding + deep learning segmentation
- ✅ **Robust Contour Filtering**: Removes spurious detections
- ✅ **Ellipse Fitting**: Precise geometric pupil characterization
- ✅ **Confidence Scoring**: Per-detection quality metrics
- ✅ **Reflection Handling**: Detects and manages corneal reflections

### Processing Capabilities
- ✅ **Single Image Processing**: Analyze individual eye images
- ✅ **Video Processing**: Batch process video files
- ✅ **Real-time Camera Feed**: Live camera input processing
- ✅ **Marker Annotation**: Interactive annotation tool for training data
- ✅ **Batch Processing**: Process multiple images efficiently

### Quality & Validation
- ✅ **Quality Levels**: 5-point quality classification (EXCELLENT to UNUSABLE)
- ✅ **Training Data Diagnostics**: Comprehensive data validation
- ✅ **Audit Logging**: Complete operation history
- ✅ **Configuration Management**: Centralized parameter control

---

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended for training)
- **GPU**: Optional but recommended (NVIDIA CUDA 11.0+ or Apple Metal)
- **Storage**: 2 GB for dependencies + training data

### Hardware Recommendations
- **GPU**: NVIDIA RTX 2080 or better (for training)
- **CPU**: Intel i7/AMD Ryzen 7 or equivalent
- **RAM**: 32 GB for large dataset training
- **Storage**: SSD with 50+ GB free space for large datasets

### Supported Devices
- NVIDIA CUDA GPUs (automatic detection)
- Apple Metal Performance Shaders (M1/M2 Macs)
- CPU-only mode (slower, for development/testing)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Prince649294u83/Pupil-Limbus-detector.git
cd Pupil-Limbus-detector
```

### Step 2: Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Dependencies installed:**
- `torch>=1.12.0` - Deep learning framework
- `torchvision>=0.13.0` - Computer vision utilities
- `segmentation-models-pytorch>=0.3.0` - Pre-built segmentation architectures
- `albumentations>=1.3.0` - Image augmentation library
- `opencv-python>=4.6.0` - Computer vision processing
- `numpy>=1.21.0` - Numerical computing
- `Pillow>=9.0.0` - Image handling
- `tqdm>=4.64.0` - Progress bars
- `matplotlib>=3.5.0` - Visualization
- `scikit-learn>=1.0.0` - Machine learning utilities

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Step 5 (Optional): Download Pre-trained Model

The application includes a pre-trained model at `models/best_model.pth`. If you want to re-train or use a different model, follow the [Model Training](#model-training) section.

---

## File Descriptions & Commands

### Root-Level Files

#### Entry Point Scripts

**`launch_gui.py`** — Main Application Entry Point
- **Purpose**: Unified interface for all detection modes (GUI, single image, video, camera)
- **Type**: Python executable
- **Dependencies**: Tkinter, OpenCV, PyTorch
- **Default Behavior**: Launches interactive GUI
- **Key Usage Examples**:
  ```bash
  python launch_gui.py                              # Launch GUI
  python launch_gui.py gui --stride 2               # GUI with options
  python launch_gui.py image -i eye.jpg             # Process single image
  python launch_gui.py video -i video.mp4           # Process video file
  python launch_gui.py camera --camera-id 0         # Live camera feed
  ```
- **Output**: Annotated images, CSV results, or live display
- **Lines of Code**: ~800-1000
- **Status**: Core application, well-tested

**`debug_single_image.py`** — Single Image Debug Tool
- **Purpose**: Detailed diagnostic analysis of a single eye image
- **Type**: Standalone utility for troubleshooting
- **Dependencies**: OpenCV, NumPy, Pillow
- **Usage**:
  ```bash
  python debug_single_image.py path/to/eye_image.jpg
  ```
- **Output**: Step-by-step preprocessing and detection analysis
- **When to Use**: Debugging detection failures, understanding preprocessing
- **Key Features**:
  - Visualizes each preprocessing step
  - Shows detection candidates  
  - Displays quality metrics
  - Identifies failure points
- **Status**: Development/debugging tool

**`check_training_data.py`** — Training Data Validation
- **Purpose**: Comprehensive diagnostic of training dataset quality
- **Type**: Data validation utility
- **Dependencies**: OpenCV, NumPy, Pillow
- **Usage**:
  ```bash
  python check_training_data.py
  ```
- **Output**: Validation report with warnings/errors
- **Checks Performed**:
  - Image-mask file pairing
  - Mask quality and connectivity
  - Class distribution analysis
  - Corrupted file detection
  - Size consistency
  - Data imbalance warnings
- **Required Before**: Training new models
- **Exit Codes**: 0 = pass, 1 = critical issues found
- **Status**: Essential pre-training tool

#### Configuration & Setup Files

**`requirements.txt`** — Python Dependencies
- **Purpose**: List of packages needed to run the project
- **Format**: pip requirements format
- **Contents** (current versions):
  - torch>=1.12.0 - Deep learning framework
  - torchvision>=0.13.0 - CV models and transforms
  - segmentation-models-pytorch>=0.3.0 - Segmentation architectures
  - albumentations>=1.3.0 - Image augmentation
  - opencv-python>=4.6.0 - Computer vision processing
  - numpy>=1.21.0 - Numerical computing
  - Pillow>=9.0.0 - Image file handling
  - tqdm>=4.64.0 - Progress bar display
  - matplotlib>=3.5.0 - Plotting and visualization
  - scikit-learn>=1.0.0 - ML utilities
- **Installation**:
  ```bash
  pip install -r requirements.txt
  ```
- **Note**: Versions are minimum; newer compatible versions generally work

**`README.md`** — Project Documentation (this file)
- **Purpose**: Complete project guide and reference
- **Sections**: Installation, quick start, training, API reference, troubleshooting
- **Target Audience**: New users, developers, researchers

**`.gitignore`** — Git Ignore Rules
- **Purpose**: Specifies files/folders not tracked by Git
- **Common Entries**: 
  - venv/, __pycache__/
  - *.pth (model weights)
  - logs/, output/
  - .DS_Store, *.pyc
- **Prevents**: Accidental commit of large files

#### Data Directories

**`clinical_data/`** — Training Dataset Directory
- **Purpose**: Central location for all clinical eye image data
- **Subdirectories**:
  - `annotations/` - Annotated image metadata and masks
  - `annoted/` - Original annotated eye images  
  - `clean/` - Preprocessed training images
  - `training_data/` - Organized train/val split
- **Usage**: Training, validation, and data management
- **Expected Size**: 1-50 GB depending on dataset
- **File Format**: JPEG images + PNG masks

**`models/`** — Pre-trained Model Weights
- **Purpose**: Store neural network model checkpoints
- **Key Files**:
  - `best_model.pth` (typically 98 MB) - Best performing model weights
  - `checkpoint_meta.json` - Training metadata (epochs, metrics)
- **Format**: PyTorch .pth files (compatible with torch.load)
- **Usage**: Loaded by detector for inference
- **Creation**: Generated during training via `scripts/train_model.py`

**`logs/`** — Audit and Event Logs
- **Purpose**: Comprehensive activity tracking for debugging and audit trails
- **Files**: 
  - `audit_YYYYMMDD_HHMMSS.jsonl` - One JSON object per line
  - `audit_training.jsonl` - Training-specific events
  - `audit_run_epoch.jsonl` - Per-epoch metrics
- **Format**: JSON Lines (JSONL) — one complete JSON object per line
- **Key Fields**: timestamp, event_type, details, status
- **Usage**: Debugging, performance analysis, compliance
- **Retention**: Keep for investigation; can be deleted after 1 week
- **Example Entry**:
  ```json
  {"timestamp": "2026-03-05T12:34:56", "event": "DETECTION", "image": "eye_01.jpg", "pupil_detected": true, "confidence": 0.987}
  ```

---

### Core Utility Scripts in `scripts/` Directory

#### Training & Model Management

**`scripts/train_model.py`** — Model Training Orchestrator
- **Purpose**: Primary entry point for training new detection models
- **Type**: Supervised training script with progress tracking
- **Python Path**: `pupil_tracking.ml.trainer.Trainer`
- **Command-Line Interface**:
  ```bash
  python scripts/train_model.py --help
  
  # Minimal usage (uses defaults)
  python scripts/train_model.py
  
  # Full customization
  python scripts/train_model.py \
    --epochs 300 \
    --batch-size 8 \
    --lr 0.0005 \
    --input-size 512 \
    --image-dir clinical_data/training_data/images \
    --mask-dir clinical_data/training_data/masks \
    --annotation-path clinical_data/annotations/annotations.json \
    --device cuda \
    --save-dir models/custom_run
  ```
- **Parameters**:
  - `--epochs <N>` - Number of training passes (default: 200)
  - `--batch-size <N>` - Images per batch (default: 16; use 4-8 for <8GB GPU)
  - `--lr <float>` - Learning rate (default: 0.001)
  - `--input-size <PX>` - Resolution (default: 512; smaller = faster)
  - `--image-dir <PATH>` - Training images directory
  - `--mask-dir <PATH>` - Segmentation masks directory
  - `--annotation-path <PATH>` - JSON annotations file
  - `--device <auto|cpu|cuda|mps>` - Compute device (default: auto-detect)
  - `--save-dir <PATH>` - Where to save checkpoints
- **Output**: 
  - `models/best_model.pth` - Best model checkpoint (loaded at inference)
  - `models/checkpoint_meta.json` - Training stats
  - `logs/audit_training.jsonl` - Per-epoch metrics
- **Training Time**: 2-8 hours (depending on batch size, GPU, dataset size)
- **Expected Result**: Dice coefficient 0.92-0.97 on validation set
- **Best Practices**:
  1. Run `python check_training_data.py` first
  2. Start with `--batch-size 16` on good GPU
  3. Monitor `nvidia-smi` or Activity Monitor (Mac)
  4. Results appear in `models/best_model.pth` automatically
- **Status**: Production-ready, actively maintained

**`scripts/run_epoch.py`** — Single Epoch Training
- **Purpose**: Quick validation of model and data pipeline without full training
- **Type**: Minimal training script for testing
- **Usage**:
  ```bash
  python scripts/run_epoch.py \
    --annotation-path clinical_data/annotations/annotations.json \
    --image-dir clinical_data/training_data/images \
    --mask-dir clinical_data/training_data/masks \
    --copies-per-image 5
  ```
- **Purpose**: Verify training setup works before long training runs
- **Output**: Single epoch metrics to console
- **Execution Time**: 1-5 minutes
- **Status**: Development/testing tool

**`scripts/export_onnx.py`** — Model Export for Deployment
- **Purpose**: Convert PyTorch model to ONNX format for deployment
- **Type**: Model conversion utility
- **Why Use**: ONNX models are 4x faster, don't require PyTorch at inference
- **Usage**:
  ```bash
  # Basic export
  python scripts/export_onnx.py \
    --model models/best_model.pth \
    --resolution 320 \
    --output-path models/best_model.onnx
  
  # With verification
  python scripts/export_onnx.py \
    --model models/best_model.pth \
    --resolution 512 \
    --verify
  ```
- **Parameters**:
  - `--model <PATH>` - Input PyTorch model file
  - `--resolution <PX>` - Target inference resolution (default: 320)
  - `--output-path <PATH>` - Output ONNX file path
  - `--verify` - Validate ONNX works correctly
- **Output**: `best_model.onnx` (~24 MB, 4x smaller than PyTorch)
- **Result Time**: 1-2 minutes
- **Status**: Production feature

#### Data Processing & Annotation

**`scripts/annotate_data.py`** — Interactive Annotation GUI
- **Purpose**: Human-in-the-loop graphical tool for marking eye features
- **Type**: Tkinter-based interactive GUI
- **Usage**:
  ```bash
  python scripts/annotate_data.py
  ```
- **Features**:
  - Load eye image
  - Click to mark pupil center
  - Click to define pupil radius
  - Mark iris/limbus boundary
  - Save annotations to JSON
  - Generate binary masks
  - Visual verification images
- **Output Generated**:
  - `clinical_data/annotations/annotations.json` - Coordinate metadata
  - `clinical_data/annotations/masks/*.png` - Binary pupil masks
  - `clinical_data/annotations/masks/*_limbus.png` - Iris masks
- **Keyboard Shortcuts**:
  - `SPACE` - Accept annotation
  - `U` - Undo last point
  - `N` - Next image
  - `Q` - Quit
- **Typical Workflow**: Annotate ~100-500 images per person per day
- **Status**: Production tool, data collection phase

**`scripts/generate_masks.py`** — Automated Mask Generation
- **Purpose**: Create segmentation masks from coordinate annotations
- **Type**: Batch processing utility
- **When Used**: After manual point annotation, before training
- **Usage**:
  ```bash
  python scripts/generate_masks.py \
    --image-dir clinical_data/raw_images \
    --annotation-file clinical_data/annotations/annotations.json \
    --output-dir clinical_data/annotations/masks \
    --num-classes 3
  ```
- **Parameters**:
  - `--image-dir <PATH>` - Directory with original images
  - `--annotation-file <PATH>` - JSON with point coordinates
  - `--output-dir <PATH>` - Where to save PNG masks
  - `--num-classes <N>` - Number of segmentation classes
- **Processing Time**: 1-5 seconds per image
- **Output**: Binary PNG masks (same resolution as input images)
- **Status**: Data preparation tool

#### Validation & Diagnostics

**`scripts/verify_data.py`** — Data Integrity Verification
- **Purpose**: Validate training dataset health and detect issues
- **Type**: Comprehensive data audit
- **Usage**:
  ```bash
  python scripts/verify_data.py \
    --image-dir clinical_data/training_data/images \
    --mask-dir clinical_data/training_data/masks \
    --model-path models/best_model.pth
  ```
- **Checks Performed**:
  - All images readable and valid format
  - All masks present, correct size
  - No corrupted files
  - Pixel value ranges correct
  - Class distribution analysis
  - Per-class statistics
- **Output**: Detailed report with warnings
- **Exit Code**: 0 (pass), 1 (warnings), 2 (critical issues)
- **Status**: Pre-training validation

**`scripts/check_files.py`** — File Consistency Check
- **Purpose**: Verify file organization and pairing
- **Type**: File system audit
- **Usage**:
  ```bash
  python scripts/check_files.py --path clinical_data/training_data
  ```
- **Checks**: 
  - All images have corresponding masks
  - Filenames match conventions
  - No orphaned files
  - Directory structure valid
- **Status**: Setup validation tool

**`scripts/diagnose_detection.py`** — Detection Diagnostics
- **Purpose**: Analyze detection performance on image set
- **Type**: Batch analysis utility
- **Usage**:
  ```bash
  python scripts/diagnose_detection.py \
    --image-dir clinical_data/clean \
    --model models/best_model.pth \
    --output-dir diagnostic_output
  ```
- **Computes**:
  - Per-image confidence scores
  - Quality distribution
  - Failure rate analysis
  - Detailed error logs
- **Output**: Reports and visualizations
- **Status**: Troubleshooting tool

#### Video Processing

**`scripts/process_video.py`** — Optimized Video Processor
- **Purpose**: Fast batch processing of video files or camera feeds
- **Type**: Real-time/offline processing utility
- **Usage**:
  ```bash
  # Process video file
  python scripts/process_video.py --input video.mp4
  
  # Process with output video and CSV
  python scripts/process_video.py \
    --input video.mp4 \
    --output result_annotated.mp4 \
    --csv results.csv \
    --stride 2 \
    --dev cuda
  
  # Live camera
  python scripts/process_video.py --camera 0
  
  # Benchmark performance
  python scripts/process_video.py --benchmark --input video.mp4
  ```
- **Parameters**:
  - `--input <PATH>` - Input video file path
  - `--camera <ID>` - Camera device ID (0 for webcam)
  - `--output <PATH>` - Output video file path
  - `--csv <PATH>` - Save frame-by-frame results
  - `--stride <N>` - Process every Nth frame (default: 1)
  - `--device <cuda|cpu|auto>` - Compute device
  - `--benchmark` - Profile performance metrics
  - `--preview` - Show live preview window
- **Output**:
  - Annotated video file (with detected circles)
  - CSV file with per-frame coordinates
  - Console statistics
- **Processing Speed**: 20-60 FPS (depending on stride, GPU, resolution)
- **Status**: Production-ready optimization feature

**`scripts/test_video_pipeline.py`** — Video Pipeline Tests
- **Purpose**: Verify video processing components work correctly
- **Type**: Test suite
- **Usage**:
  ```bash
  python scripts/test_video_pipeline.py
  ```
- **Tests**:
  - ROI tracking on synthetic frames
  - Temporal smoothing (Kalman filter)
  - Detection consistency
- **Status**: Development/validation tool

#### Ring Detection (Suction Cup)

**`scripts/annotate_ring_data.py`** — Ring Presence Labeling Tool
- **Purpose**: Mark which images contain suction rings (docked vs pre-docked)
- **Type**: Simple interactive window classifier
- **Why Important**: Separate training set improves ring detection accuracy
- **Usage**:
  ```bash
  # Start fresh
  python scripts/annotate_ring_data.py \
    --image-dir clinical_data/training_data/images \
    --output clinical_data/ring_labels.json
  
  # Resume existing session
  python scripts/annotate_ring_data.py \
    --image-dir clinical_data/training_data/images \
    --output clinical_data/ring_labels.json \
    --resume
  ```
- **Keyboard Controls**:
  - `R` - Image has ring (docked)
  - `N` - No ring (pre-docked)
  - `P` - Partial ring visible
  - `U` - Undo last label
  - `S` - Save progress
  - `Q` or `ESC` - Quit
- **Output Format**:
  ```json
  {
    "image_001.jpg": {"ring_present": true, "ring_visibility": "full"},
    "image_002.jpg": {"ring_present": false, "ring_visibility": "none"}
  }
  ```
- **Efficiency**: 50-100 images per hour per annotator
- **Status**: Data collection feature

**`scripts/train_ring_classifier.py`** — Ring Classifier Training
- **Purpose**: Train lightweight binary classifier for ring detection
- **Type**: Transfer learning utility
- **Architecture**: MobileNetV2-based CNN (2.5 MB)
- **Usage**:
  ```bash
  # Basic training
  python scripts/train_ring_classifier.py \
    --image-dir clinical_data/training_data/images \
    --labels clinical_data/ring_labels.json \
    --epochs 50
  
  # Advanced
  python scripts/train_ring_classifier.py \
    --image-dir clinical_data/training_data/images \
    --labels clinical_data/ring_labels.json \
    --epochs 80 \
    --batch-size 32 \
    --lr 0.0003 \
    --device cuda \
    --save-dir models/ring_classifier
  ```
- **Parameters**:
  - `--image-dir <PATH>` - Training images
  - `--labels <PATH>` - ring_labels.json file
  - `--epochs <N>` - Training passes (default: 50)
  - `--batch-size <N>` - Batch size (default: 16)
  - `--val-split <0-1>` - Validation fraction (default: 0.2)
  - `--device <cuda|cpu|auto>` - Compute device
- **Output**: `models/ring_classifier.pth` (~2.5 MB)
- **Training Time**: 5-15 minutes on GPU
- **Accuracy**: 95%+ on balanced data
- **Status**: Specialized feature

**`scripts/evaluate_ring_detection.py`** — Ring Classifier Evaluation
- **Purpose**: Measure ring classifier accuracy against ground truth
- **Type**: Performance evaluation utility
- **Usage**:
  ```bash
  # Evaluate combined CNN + heuristic detector
  python scripts/evaluate_ring_detection.py \
    --image-dir clinical_data/training_data/images \
    --labels clinical_data/ring_labels.json \
    --classifier models/ring_classifier.pth \
    --output results.json
  
  # Heuristic only (no trained model needed)
  python scripts/evaluate_ring_detection.py \
    --image-dir clinical_data/training_data/images \
    --labels clinical_data/ring_labels.json \
    --heuristic-only
  ```
- **Metrics Computed**:
  - Accuracy
  - Precision / Recall
  - F1 Score
  - Confusion Matrix
- **Output**: Classification report + CSV
- **Status**: Validation tool

#### Performance & Benchmarking

**`scripts/benchmark_fps.py`** — Speed/Quality Benchmark Tool
- **Purpose**: Measure inference speed at different settings
- **Type**: Performance profiling utility
- **Usage**:
  ```bash
  # Basic benchmark
  python scripts/benchmark_fps.py --model models/best_model.pth
  
  # Video input
  python scripts/benchmark_fps.py \
    --model models/best_model.pth \
    --source video.mp4
  
  # Camera input
  python scripts/benchmark_fps.py \
    --model models/best_model.pth \
    --source 0
  ```
- **Measures**:
  - FPS at different resolutions
  - FP16 vs FP32 performance
  - Memory usage
  - Latency P95/P99
- **Output**: Detailed table with results
- **Status**: Optimization tool

**`scripts/benchmark_video_speed.py`** — Video-Specific Benchmark
- **Purpose**: Profile end-to-end video processing performance
- **Type**: Video throughput measurement
- **Usage**:
  ```bash
  python scripts/benchmark_video_speed.py \
    --input video.mp4 \
    --model models/best_model.pth \
    --warmup 10 \
    --runs 3
  ```
- **Reports**:
  - Average FPS
  - Min/Max FPS
  - Total processing time
  - Memory peak usage
- **Status**: Optimization tool

---

### Core Package: `pupil_tracking/` Module

#### Main API (`pupil_tracking/__init__.py`)
- **Purpose**: Package initialization and version info
- **Exports**: Version number and key classes
- **Current Version**: 2.0.0
- **Key Usage**:
  ```python
  from pupil_tracking import __version__
  print(__version__)  # "2.0.0"
  ```

#### Core Detection (`pupil_tracking/core/`)

**`core/detector.py`** — Unified Feature Detector
- **Purpose**: High-level API for pupil & iris detection
- **Type**: Core detection pipeline
- **Key Class**: `UnifiedDetector`
- **Input**: Grayscale image (numpy array)
- **Output**: Detection result object with coordinates, confidence
- **Usage**:
  ```python
  from pupil_tracking.core.detector import UnifiedDetector
  detector = UnifiedDetector()
  result = detector.detect(gray_image)
  if result:
      print(f"Pupil center: ({result.center_x}, {result.center_y})")
      print(f"Confidence: {result.confidence}")
  ```
- **Key Methods**:
  - `detect(image)` - Detect pupil in single frame
  - `set_roi(roi)` - Constrain search to ROI
  - `reset()` - Clear cached state
- **Status**: Core component, well-tested

**`core/ellipse_fitter.py`** — Geometric Fitting Engine
- **Purpose**: Fit perfect ellipses to detected contours
- **Type**: Image geometry utility
- **Key Class**: `EllipseFitter`
- **Input**: Image contours (OpenCV format)
- **Output**: (center_x, center_y, semi_major, semi_minor, angle)
- **Key Methods**:
  - `fit_ellipse(contour)` - Fit single ellipse
  - `fit_circle(contour)` - Constrain to circle
  - `refine(contour)` - Iterative refinement
- **Status**: Geometric utility

**`core/corneal_center.py`** — Specular Reflection Detection
- **Purpose**: Find corneal reflection center
- **Usage**:
  ```python
  from pupil_tracking.core.corneal_center import find_corneal_center
  corneal_pos = find_corneal_center(grayscale_image)  # (x, y) or None
  ```
- **Status**: Preprocessing utility

**`core/confidence.py`** — Quality Scoring System
- **Purpose**: Assign confidence/quality scores to detections
- **Levels**: EXCELLENT (0.9+), GOOD (0.8+), OK (0.7+), POOR, UNUSABLE
- **Factors**: Shape regularity, size plausibility, anatomical consistency
- **Status**: Quality control component

**`core/adaptive_pipeline.py`** — Adaptive Detection Pipeline
- **Purpose**: Switch between detection strategies based on image content
- **Strategies**: 
  - Traditional (thresholding + contours)
  - Deep learning (segmentation model)
  - Ring-aware (detects suction cup interference)
- **Auto-Selection**: Based on ring presence confidence
- **Status**: Advanced feature

**`core/eye_roi_detector.py`** — Region of Interest Detection
- **Purpose**: Locate eye region in full face images
- **Output**: Bounding box + confidence
- **Usage**:
  ```python
  from pupil_tracking.core.eye_roi_detector import EyeROIDetector
  roi_detector = EyeROIDetector()
  roi = roi_detector.detect(face_image)  # Get eye region
  ```
- **Status**: Face detection component

**`core/ring_detector.py`** — Suction Ring Detector
- **Purpose**: Detect presence of suction cup/ring
- **Type**: Hybrid CNN + heuristic detector
- **Output**: ring_present (bool), confidence (float)
- **Status**: Specialized feature

#### Machine Learning (`pupil_tracking/ml/`)

**`ml/architecture.py`** — Neural Network Model Definition
- **Purpose**: Define the segmentation neural network
- **Architecture**: U-Net with ResNet-34 encoder
- **Model Type**: Semantic segmentation (class per pixel)
- **Classes**: 
  - 0: Background
  - 1: Pupil (dark center)
  - 2: Iris/Limbus (colored ring)
  - 3: Suction ring (if enabled)
- **Input Size**: 512×512 pixels
- **Output**: Probability map per class
- **Key Class**: `EyeSegmentationModel`
- **Status**: Core ML component

**`ml/dataset.py`** — Training Data Management
- **Purpose**: Load and prepare training data
- **Handles**:
  - Image/mask pairs
  - Annotation JSON parsing
  - Data validation
  - Train/val splitting
  - Augmentation
- **Key Functions**:
  - `load_annotations(path)` - Parse JSON
  - `generate_mask_from_annotation(...)` - Create PNG mask
  - `create_dataset(...)` - Build training dataset
- **Status**: Data pipeline

**`ml/trainer.py`** — Training Loop Manager
- **Purpose**: Orchestrate model training
- **Workflow**:
  1. Load dataset
  2. Create model
  3. For each epoch:
     - Forward pass on batch
     - Compute loss
     - Backward pass
     - Update weights
  4. Save best model
- **Key Class**: `Trainer`
- **Status**: Training core

**`ml/inference.py`** — Model Inference Engine
- **Purpose**: Run trained model on images
- **Key Class**: `SegmentationInference`
- **Input**: RGB/grayscale image
- **Output**: Segmentation mask (class per pixel)
- **Features**:
  - Batch processing
  - FP16 precision option
  - GPU acceleration
- **Status**: Inference engine

**`ml/losses.py`** — Custom Loss Functions
- **Purpose**: Define training loss equations
- **Loss Types**:
  - Dice Loss - Standard for segmentation
  - Focal Loss - Handles class imbalance
  - Composite Loss - Weighted combination
- **Status**: Training configuration

**`ml/ring_classifier.py`** — Ring Presence Classifier
- **Purpose**: Binary classification (ring yes/no)
- **Architecture**: MobileNetV2 (lightweight)
- **Input**: 224×224 RGB image
- **Output**: ring_probability (0-1)
- **Status**: Specialized model

#### Video Processing (`pupil_tracking/video/`)

**`video/video_processor.py`** — Frame-by-Frame Video Processing
- **Purpose**: Main video analysis pipeline
- **Input**: Video file or OpenCV VideoCapture
- **Output**: Per-frame detection results
- **Key Method**: `process_frame(frame)`
- **Status**: Video core

**`video/optimized_processor.py`** — High-Speed Video Processor
- **Purpose**: Fast batch video processing with optimizations
- **Optimizations**: 
  - Frame skipping (stride)
  - ROI tracking
  - FP16 precision
  - torch.compile JIT
- **Output**: Optimized detections at target FPS
- **Status**: Performance-optimized feature

**`video/kalman_tracker.py`** — Multi-Object Tracking
- **Purpose**: Smooth noisy detections using Kalman filter
- **Input**: Raw per-frame detections
- **Output**: Filtered, temporally-smooth trajectories
- **Benefits**: Removes jitter, handles brief occlusions
- **Status**: Smoothing component

**`video/temporal_smoother.py`** — Temporal Filtering
- **Purpose**: Smooth results across time
- **Methods**: Moving average, Kalman, exponential smoothing
- **Status**: Post-processing utility

#### Image Preprocessing (`pupil_tracking/preprocessing/`)

**`preprocessing/reflection_removal.py`** — Corneal Reflection Removal
- **Purpose**: Remove specular highlights
- **Type**: Image processing utility
- **Status**: Preprocessing component

**`preprocessing/roi_extractor.py`** — Region Extraction
- **Purpose**: Extract eye region from larger image
- **Status**: Preprocessing utility

**`preprocessing/normalizer.py`** — Image Normalization
- **Purpose**: Standardize image statistics (mean, std)
- **Benefits**: Improves model robustness
- **Status**: Preprocessing component

**`preprocessing/ring_aware.py`** — Ring-Aware Preprocessing
- **Purpose**: Mask out suction ring region
- **Status**: Specialized preprocessing

**`preprocessing/suction_ring_masker.py`** — Ring Masking
- **Purpose**: Create binary mask of ring region
- **Status**: Ring-detection support

#### Annotation Tools (`pupil_tracking/annotation/`)

**`annotation/annotation_tool.py`** — Interactive Annotation GUI
- **Purpose**: Mark pupil/iris coordinates in images
- **Features**:
  - Load/display images
  - Click-based marking
  - Ellipse fitting preview
  - Save to JSON format
- **Status**: Data collection tool

**`annotation/mask_generator.py`** — Mask Creation
- **Purpose**: Convert coordinates to binary masks
- **Status**: Data preparation component

#### Utilities & Configuration

**`pupil_tracking/utils/logger.py`** — Event Logging System
- **Purpose**: Audit trail and debugging logs
- **Format**: JSON Lines (JSONL)
- **Fields**: timestamp, event_type, status, details
- **Usage**:
  ```python
  from pupil_tracking.utils.logger import AuditLogger
  logger = AuditLogger("my_task")
  logger.log_event("DETECTION", status="success", confidence=0.95)
  ```
- **Output Files**: `logs/audit_*.jsonl`

**`pupil_tracking/utils/config.py`** — Configuration Management
- **Purpose**: Centralized parameter storage
- **Usage**:
  ```python
  from pupil_tracking.utils.config import get_config, se_config
  config = get_config()
  config['threshold_1'] = 100
  ```
- **Key Settings**: Thresholds, model paths, device
- **Status**: Configuration system

**`pupil_tracking/utils/types.py`** — Type Definitions
- **Purpose**: Data classes for detection results
- **Key Classes**:
  - `DetectionResult` - Single frame detection
  - `FrameMetrics` - Per-frame statistics
  - `TrainingConfig` - Training parameters
- **Status**: Type system

#### Tests (`pupil_tracking/tests/`)

**Test Suite Files:**
- `test_pipeline.py` - End-to-end pipeline tests
- `test_detection.py` - Detection algorithm tests  
- `test_calibration.py` - Geometric calibration tests
- `test_clinical_accuracy.py` - Clinical performance validation
- `test_stabilization.py` - Temporal smoothing tests
- `test_blink_detection.py` - Blink detection tests
- `test_roi_locking.py` - ROI tracking tests
- `test_adaptive_threshold.py` - Thresholding algorithm tests

**Running Tests:**
```bash
# Run all tests
python -m pytest pupil_tracking/tests/ -v

# Run specific test
python -m pytest pupil_tracking/tests/test_pipeline.py -v

# Run with coverage
python -m pytest pupil_tracking/tests/ --cov=pupil_tracking
```

#### Main Interface Files

**`pupil_tracking/image_interface.py`** — Static Image Analysis
- **Purpose**: Analyze single eye image with OpenCV window
- **Type**: Interactive analysis tool
- **Usage**:
  ```bash
  python -m pupil_tracking.image_interface image.png
  python -m pupil_tracking.image_interface folder/  # batch
  ```
- **Keyboard Controls**:
  - P - Toggle pupil overlay
  - L - Toggle limbus overlay
  - C - Toggle corneal center
  - S - Save annotated image
  - Q - Quit
- **Status**: Development tool

**`pupil_tracking/clinical_debug.py`** — Clinical Debugging Interface
- **Purpose**: Comprehensive diagnostic analysis for troubleshooting
- **Type**: Development/debugging tool
- **Status**: Debug component

---

## Quick Start Guide

### For Absolute Beginners

### 1. Installation (5 minutes)

```bash
#Clone repository
git clone https://github.com/Prince649294u83/Pupil-Limbus-detector.git
cd Pupil-Limbus-detector

# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (Mac/Linux)
source venv/bin/activate

# Install everything
pip install -r requirements.txt
```

### 2. Run the GUI (1 minute)

```bash
python launch_gui.py
```

A window will pop up. Try:
- **GUI Mode**: Default, interactive interface
- **Single Image**: Click "Open Image" and select an eye image
- **Video**: Click "Load Video" and select a video file

### 3. Process Your First Image (2 minutes)

```bash
python launch_gui.py image -i your_eye_image.jpg
```

Look for:
- Green circle around pupil
- White circle around iris/limbus
- Confidence scores in console

### 4. Common Issues

- **GPU not detected?** Run:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
  If False, CPU mode will be used (slower)

- **Model file missing?** Download from:
  ```bash
  # Assuming a models directory with best_model.pth exists
  ls -la models/
  ```

---

## Main Entry Points

### Entry Point 1: launch_gui.py (Main Application)

**Purpose**: Universal interface for all detection modes

**Modes**:

| Mode | Command | Use Case |
|------|---------|----------|
| GUI | `python launch_gui.py` | Interactive, test one image |
| Single Image | `python launch_gui.py image -i eye.jpg` | Process single file |
| Video | `python launch_gui.py video -i video.mp4` | Batch process video |
| Camera | `python launch_gui.py camera` | Live webcam stream |

### Entry Point 2: check_training_data.py (Data Validation)

#### Performance & Testing

**`scripts/benchmark_fps.py`** — Performance Benchmarking
- **Purpose**: Measure inference speed (frames per second) at different settings
- **Type**: Performance profiling utility
- **Usage**:
  ```bash
  # Benchmark with default camera
  python scripts/benchmark_fps.py --model models/best_model.pth
  
  # Benchmark with specific video
  python scripts/benchmark_fps.py \
    --model models/best_model.pth \
    --source video.mp4 \
    --resolutions 256 320 512 \
    --batch-sizes 1 4 8
  
  # Benchmark with synthetic frames (no hardware required)
  python scripts/benchmark_fps.py \
    --model models/best_model.pth \
    --synthetic \
    --duration 60
  ```
- **Parameters**:
  - `--model <PATH>` - Model file to benchmark
  - `--source <camera_id|video_path>` - Input source (default: camera 0)
  - `--resolutions` - List of resolutions to test (default: 320 512)
  - `--batch-sizes` - Batch sizes to test (default: 1)
  - `--duration <SEC>` - How long to run benchmark
  - `--synthetic` - Use synthetic frames instead of camera
- **Metrics Reported**:
  - FPS (frames per second)
  - Latency (ms per inference)
  - Memory usage
  - GPU utilization
- **Use Cases**:
  - Determine max resolution for target FPS
  - Compare GPU vs CPU speed
  - Validate optimization changes
- **Typical Results**: 30-100 FPS at 512×512 on modern GPU
- **Status**: Performance analysis tool

**`scripts/process_video.py`** — Batch Video Processing
- **Purpose**: Process video files end-to-end with optional output video generation
- **Type**: High-level video processing utility
- **Usage**:
  ```bash
  # Process video, save annotated output
  python scripts/process_video.py \
    --input video.mp4 \
    --output annotated_video.mp4 \
    --csv results.csv
  
  # Process with frame skipping for speed
  python scripts/process_video.py \
    --input video.mp4 \
    --stride 2 \
    --device cuda \
    --output output.mp4
  
  # Live camera processing
  python scripts/process_video.py \
    --camera 0 \
    --output-csv camera_results.csv
  
  # With preview window
  python scripts/process_video.py \
    --input video.mp4 \
    --preview \
    --benchmark
  ```
- **Parameters**:
  - `--input <PATH>` - Input video file (MP4, AVI, MOV)
  - `--camera <ID>` - Use camera instead of file
  - `--output <PATH>` - Save annotated video
  - `--csv <PATH>` - Save results as CSV
  - `--stride <N>` - Process every Nth frame
  - `--device <cuda|cpu>` - Compute device
  - `--preview` - Show live preview window
  - `--benchmark` - Report FPS
- **Output Files**:
  - Annotated video with visualization
  - CSV with per-frame coordinates
  - Quality report
- **Processing Time**: 5 seconds per minute of video (512×512, GPU)
- **Status**: Production video processing

**`scripts/test_video_pipeline.py`** — Video Pipeline Testing
- **Purpose**: Unit tests for video processing component
- **Type**: Test suite
- **Usage**:
  ```bash
  python scripts/test_video_pipeline.py
  ```
- **Tests**: Video reading, frame caching, processing pipeline
- **Status**: Development/CI tool

---

### Core Module Structure (`pupil_tracking/`)

#### Main Detection Module

**`pupil_tracking/detection.py`** — Pupil Candidate Detection
- **Purpose**: Multi-strategy pupil detection via contour extraction
- **Type**: Core detection algorithm
- **Key Class**: `PupilDetector`
- **Algorithm**:
  1. Apply global + adaptive thresholding
  2. Extract contours from binary image
  3. Filter by area, circularity, aspect ratio
  4. Return candidate contours sorted by score
- **Used By**: `core/detector.py` and `core/enhanced_detection.py`
- **Lines**: ~400
- **Status**: Production-grade, tested

**`pupil_tracking/preprocessing.py`** — Image Preprocessing
- **Purpose**: Normalize and prepare images for detection
- **Type**: Image processing pipeline
- **Key Class**: `ImagePreprocessor`
- **Pipeline**:
  1. Convert to grayscale
  2. Median filtering (noise removal)
  3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
  4. Optional normalization (0-1 range)
- **Configuration**: Adjustable via `config.py`
- **Used By**: Nearly all detection pipelines
- **Performance**: ~10ms per image
- **Status**: Production-ready

**`pupil_tracking/run_realtime.py`** — Real-time Camera Processing
- **Purpose**: Simple real-time eye tracking from webcam
- **Type**: Example/starter script
- **Usage**:
  ```bash
  python -m pupil_tracking.run_realtime
  ```
- **Features**:
  - Live camera capture
  - Real-time detection overlay
  - FPS display
  - Frame saving on demand
- **Dependencies**: OpenCV, haar cascades
- **Status**: Example code (see `launch_gui.py` for production version)

#### Configuration & Utilities

**`pupil_tracking/logger.py`** — Audit Logging System
- **Purpose**: Structured event logging for audit trails
- **Type**: Logging infrastructure
- **Key Class**: `PupilTrackingLogger`
- **Output**: CSV + JSON LINES formats
- **Key Methods**:
  - `log_frame()` - Log frame-level measurements
  - `log_event()` - Custom event logging
  - `finalize()` - Close session
- **Output Files**:
  - `frames.csv` - One row per frame
  - `session_summary.json` - Aggregate stats
- **Used By**: Training, inference, video processing
- **Status**: Production logging system

**`pupil_tracking/image_interface.py`** — Image I/O Wrapper
- **Purpose**: Unified interface for reading/writing images
- **Type**: I/O abstraction layer
- **Key Class**: `ImageInterface`
- **Supported Formats**: JPEG, PNG, BMP, TIFF
- **Key Methods**:
  - `read()` - Load image (auto-converts)
  - `write()` - Save image (auto-selects format)
  - `validate()` - Check image validity
- **Usage Example**:
  ```python
  from pupil_tracking.image_interface import ImageInterface
  iface = ImageInterface()
  img = iface.read("image.jpg")
  iface.write("output.png", img)
  ```
- **Status**: Utility module

**`pupil_tracking/clinical_debug.py`** — Clinical Diagnostic Tool
- **Purpose**: Visualize detection pipeline step-by-step
- **Type**: Debugging utility
- **Usage**:
  ```bash
  python -m pupil_tracking.clinical_debug clinical_eye.png
  python -m pupil_tracking.clinical_debug clinical_eye.png --save
  ```
- **Output**: Visualization showing each detection step
- **Key Use**: Troubleshooting failed detections
- **Status**: Development tool

---

### Core Detection Pipeline (`pupil_tracking/core/`)

**`core/detector.py`** — Main Detection Pipeline
- **Purpose**: Unified orchestrator combining all detection strategies
- **Type**: High-level API
- **Key Class**: `UnifiedDetector`
- **Algorithm Steps**:
  1. Preprocessing
  2. Multi-thread detection (traditional + ML)
  3. Contour filtering
  4. Geometric fitting
  5. Confidence scoring
  6. Quality assessment
- **Output**: `DetectionResult` with pupil + limbus data
- **Usage Example**:
  ```python
  from pupil_tracking.core.detector import UnifiedDetector
  detector = UnifiedDetector()
  result = detector.detect(image)
  print(f"Pupil: ({result.pupil.ellipse.center_x}, {result.pupil.ellipse.center_y})")
  ```
- **Lines**: ~600
- **Status**: Core API, well-tested

**`core/geometric_fit.py`** — Ellipse Fitting
- **Purpose**: Fit ellipse to detected pupil boundaries
- **Type**: Geometric algorithm
- **Key Class**: `GeometricFitter`
- **Algorithm**: Fitzgibbon et al. ellipse fitting
- **Output**: Ellipse parameters (center, radii, angle)
- **Accuracy**: Sub-pixel precision
- **Used By**: Pupil center/radius estimation
- **Status**: Production-grade

**`core/contour_filtering.py`** — Contour Validation
- **Purpose**: Filter spurious contours from detection phase
- **Type**: Post-processing filter
- **Key Class**: `ContourFilter`
- **Criteria**:
  - Area (min/max thresholds)
  - Circularity (0-1 score)
  - Aspect ratio (width/height)
  - Convexity
- **Removes**: ~80-90% of false positives
- **Status**: Production component

**`core/limbus_detector.py`** — Iris Boundary Detection
- **Purpose**: Detect iris-sclera boundary
- **Type**: Specialized detection
- **Key Class**: `LimbusDetector`
- **Algorithm**: Edge detection + circular Hough transform
- **Output**: Iris center and radius
- **Accuracy**: ±5-10 pixels typical
- **Status**: Core detection module

**`core/corneal_center.py`** — Reflection Detection
- **Purpose**: Identify corneal light reflections (glints)
- **Type**: Specialized detection
- **Key Class**: `CornealCenterCalculator`
- **Purpose**: Detect and label corneal reflections for filtering
- **Output**: Reflection positions and brightness
- **Used By**: Image quality assessment
- **Status**: Quality control module

**`core/confidence.py`** — Confidence Scoring
- **Purpose**: Compute detection confidence and quality metrics
- **Type**: Scoring system
- **Key Class**: `ConfidenceScorer`
- **Inputs**: Detection results, preprocessing stats
- **Output**: 
  - Confidence (0-1 float)
  - Quality level (EXCELLENT to UNUSABLE)
  - Per-module confidence scores
- **Status**: Quality assessment module

**`core/smart_fitter.py`** — Advanced Fitting (NEW)
- **Purpose**: Enhanced geometric fitting with outlier rejection
- **Type**: Robust fitting algorithm
- **Status**: Recent addition, still testing

---

### Machine Learning Module (`pupil_tracking/ml/`)

**`ml/architecture.py`** — U-Net Model Definition
- **Purpose**: Define neural network architecture
- **Type**: PyTorch nn.Module
- **Key Class**: `EyeSegmentationModel`
- **Architecture**: U-Net with ResNet-34 encoder (from segmentation-models-pytorch)
- **Input**: 512×512×3 RGB image
- **Output**: 512×512×3 probability map (3 classes)
- **Parameters**: ~25 million
- **FLOPs**: ~120 billion
- **Status**: Model definition

**`ml/dataset.py`** — PyTorch Dataset
- **Purpose**: Data loading and augmentation
- **Type**: PyTorch Dataset class
- **Key Class**: `EyeSegmentationDataset`
- **Features**:
  - Image + mask pairing
  - Automated augmentation (albumentations)
  - Lazy loading
  - Balanced sampling
- **Usage**:
  ```python
  dataset = EyeSegmentationDataset(
    image_dir="clinical_data/images",
    mask_dir="clinical_data/masks",
    image_size=512,
    augment=True
  )
  ```
- **Status**: Data loading infrastructure

**`ml/trainer.py`** — Training Loop
- **Purpose**: Execute model training with validation
- **Type**: Training orchestrator
- **Key Class**: `Trainer`
- **Features**:
  - Learning rate scheduling
  - Early stopping
  - Checkpointing
  - Metrics tracking
  - Logging
- **Output**: Best model checkpoint + metadata
- **Status**: Training infrastructure

**`ml/losses.py`** — Custom Loss Functions
- **Purpose**: Define loss functions for training
- **Type**: Loss function definitions
- **Losses Included**:
  - Weighted Cross-Entropy (primary)
  - Dice Loss
  - Focal Loss (for hard negatives)
  - Combination losses
- **Status**: Training utilities

**`ml/inference.py`** — Inference Pipeline
- **Purpose**: Run trained model on input images
- **Type**: Inference wrapper
- **Key Class**: `InferenceEngine`
- **Features**:
  - Batch processing
  - GPU/CPU support
  - Post-processing
  - Confidence scoring
- **Status**: Inference infrastructure

**`ml/fast_inference.py`** — Optimized Inference (NEW)
- **Purpose**: High-speed inference with optimizations
- **Features**:
  - FP16 precision
  - Batch processing
  - ROI tracking
  - Parallelization
- **Performance**: 2-3x speedup vs standard
- **Status**: Performance optimization

**`ml/postprocess.py`** — Post-processing Utilities
- **Purpose**: Process raw model outputs
- **Type**: Output processing
- **Functions**:
  - Argmax to class labels
  - Probability thresholding
  - Morphological operations
  - CRF refinement (optional)
- **Status**: Output processing module

---

### Video Processing Module (`pupil_tracking/video/`)

**`video/video_processor.py`** — Video File Processing
- **Purpose**: Read, process, and write video files
- **Type**: Video I/O and processing
- **Key Class**: `VideoProcessor`
- **Features**:
  - Multi-codec support (H.264, VP9, etc.)
  - Frame skipping (stride)
  - Progress tracking
  - Audio passthrough option
- **Supported Formats**: MP4, AVI, MOV, MKV
- **Status**: Video processing infrastructure

**`video/camera_processor.py`** — Live Camera Processing
- **Purpose**: Webcam capture and real-time analysis
- **Type**: Camera interface
- **Key Class**: `CameraProcessor`
- **Features**:
  - Multiple camera support
  - Resolution selection
  - FPS targeting
  - Frame saving
- **Status**: Camera processing module

**`video/optimized_processor.py`** — Optimized Pipeline (NEW)
- **Purpose**: High-performance video processing
- **Features**:
  - GPU acceleration
  - Frame batching
  - Memory optimization
  - ROI tracking
- **Performance**: 50-100 FPS on good GPU
- **Status**: Performance optimization

**`video/frame_buffer.py`** — Frame Caching
- **Purpose**: Efficient frame buffering for batch processing
- **Type**: Memory management
- **Status**: Performance utility

---

### Annotation Tools (`pupil_tracking/annotation/`)

**`annotation/annotation_tool.py`** — Interactive Annotation GUI
- **Purpose**: User-friendly interface for marking eye features
- **Type**: Tkinter GUI application
- **Key Class**: `AnnotationTool`
- **Features**:
  - Load images from directory
  - Click-based landmark marking
  - Real-time visualization
  - Batch processing
  - Mask generation
- **Output**: JSON annotations + PNG masks
- **Status**: Data collection tool

**`annotation/annotation_converter.py`** — Format Conversion
- **Purpose**: Convert between annotation formats
- **Supported Formats**: JSON, CSV, COCO, Pascal VOC
- **Status**: Data utility

**`annotation/mask_generator.py`** — Mask Generation
- **Purpose**: Create binary masks from point coordinates
- **Type**: Mask creation utility
- **Methods**:
  - Rasterize points to masks
  - Generate contours
  - Fill regions
- **Status**: Data preparation tool

---

### Calibration Module (`pupil_tracking/calibration/`)

**`calibration/calibration.py`** — Camera Calibration
- **Purpose**: Intrinsic camera parameter estimation
- **Type**: Calibration utility
- **Methods**: Checkerboard pattern detection
- **Output**: Camera matrix and distortion coefficients
- **Status**: Optional calibration tool

**`calibration/camera_calibration.py`** — Advanced Calibration
- **Purpose**: Multi-view calibration
- **Features**: Stereo calibration, distortion correction
- **Status**: Advanced tool

---

### Utility Module (`pupil_tracking/utils/`)

**`utils/config.py`** — Configuration Management
- **Purpose**: Central configuration management
- **Type**: Configuration system
- **Key Functions**:
  - `get_config()` - Get current configuration
  - `set_config(cfg)` - Update configuration
  - `load_config(path)` - Load from file
  - `save_config(path)` - Save to file
- **Configuration Options**: 200+ parameters organized by category
- **Status**: Configuration infrastructure

**`utils/logger.py`** — Logging Utilities
- **Purpose**: Structured logging for debugging
- **Type**: Logging system
- **Key Class**: `AuditLogger`
- **Features**: Thread-safe, event logging, export
- **Status**: Logging infrastructure

**`utils/helpers.py`** — Helper Functions
- **Purpose**: Common utility functions
- **Type**: Utility library
- **Functions**: Path handling, image conversion, validation
- **Status**: Utility module

---

### Advanced Scripts (`scripts/` — Extended Documentation)

#### Live Video Annotation

**`scripts/annotate_live_video.py`** — Real-Time Video Annotation Tool
- **Purpose**: Advanced interactive tool for annotating eye video frames with live feedback
- **Type**: Real-time annotation GUI with real-time training
- **Key Features**:
  - Pause/resume video playback with frame-by-frame control
  - Multiple annotation modes (pupil, limbus, circle vs ellipse)
  - Interactive drawing with edge-snapping capability
  - Live model retraining while annotating
  - Automated mask generation from annotations
  - Incremental training on new annotations
  - JSON export of all annotated frames
- **Controls**:
  - `SPACE` - Pause/resume video
  - `P` - Switch to pupil annotation mode
  - `L` - Switch to limbus annotation mode
  - `ENTER` - Confirm annotation
  - `R` - Refine fit using image edges
  - `U` - Undo last point
  - `T` - Trigger incremental model retrain
  - `S` - Save all annotations
  - `+/-` - Zoom in/out
  - `G` - Toggle edge-snap mode
  - `D` - Toggle circle/ellipse constraint
  - `ESC/Q` - Quit
- **Usage**:
  ```bash
  python scripts/annotate_live_video.py --video video.mp4
  python scripts/annotate_live_video.py --camera 0
  python scripts/annotate_live_video.py --video video.mp4 --auto-retrain
  ```
- **Output Generated**:
  - JSON annotations file with all frame coordinates
  - Binary masks for training
  - Saved model checkpoints from incremental training
  - Verification visualization images
- **Best For**: Large-scale annotation projects with incremental training
- **Status**: Advanced production tool, actively maintained

#### Performance Benchmarking

**`scripts/benchmark_fps.py`** — Frame Rate Performance Benchmark
- **Purpose**: Comprehensive performance profiling and FPS measurement
- **Type**: Performance analysis and optimization tool
- **Capabilities**:
  - Measure inference speed (FPS) at different resolutions/batch sizes
  - Compare GPU vs CPU performance
  - GPU utilization monitoring
  - Latency breakdown (decode/preprocess/infer/postprocess)
  - Synthetic frame generation for baseline testing
- **Usage**:
  ```bash
  # Basic benchmark with default camera
  python scripts/benchmark_fps.py --model models/best_model.pth
  
  # Benchmark video file with multiple resolutions
  python scripts/benchmark_fps.py \
    --model models/best_model.pth \
    --source video.mp4 \
    --resolutions 256 320 512 \
    --batch-sizes 1 4 8
  
  # Synthetic frames (no hardware required)
  python scripts/benchmark_fps.py \
    --model models/best_model.pth \
    --synthetic \
    --duration 60 \
    --device cuda
  ```
- **Parameters**:
  - `--model <PATH>` - Model to benchmark (required)
  - `--source <camera_id|video_path>` - Input source (default: camera 0)
  - `--resolutions` - Resolutions to test (e.g., 256 320 512)
  - `--batch-sizes` - Batch sizes (e.g., 1 4 8)
  - `--duration <SEC>` - Benchmark duration in seconds
  - `--synthetic` - Use synthetic frames instead of real input
  - `--device` - Force device (auto/cpu/cuda)
- **Typical Output**:
  ```
  Resolution: 512×512
    - Single frame: 25.3 FPS (39.5ms latency)
    - Batch 4: 28.1 FPS (35.7ms latency)
    - GPU util: 65%
  Resolution: 320×320
    - Single frame: 45.2 FPS (22.1ms latency)
    - Batch 4: 52.1 FPS (19.2ms latency)
    - GPU util: 48%
  ```
- **Use Cases**: Finding optimal resolution/batch size for deployment targets
- **Status**: Performance profiling tool

**`scripts/benchmark_video_speed.py`** — Video Processing Speed Benchmark
- **Purpose**: Measure video decode and processing pipeline performance
- **Type**: Video-specific performance profiling
- **Measurements**:
  - Video decode speed (frames read per second)
  - Per-frame preprocessing time
  - Inference latency (individual vs batched)
  - Overall pipeline FPS
  - GPU memory usage
  - Comparison metrics (optimized vs baseline)
- **Usage**:
  ```bash
  # Benchmark video file
  python scripts/benchmark_video_speed.py --input video.mp4 --frames 500
  
  # With detailed breakdown
  python scripts/benchmark_video_speed.py \
    --input video.mp4 \
    --frames 200 \
    --batch 4 \
    --verbose
  
  # Synthetic benchmark (no file needed)
  python scripts/benchmark_video_speed.py --synthetic --frames 300
  ```
- **Parameters**:
  - `--input <PATH>` - Video file to benchmark
  - `--frames <N>` - Number of frames to process (default: 100)
  - `--batch <N>` - Batch size for processing (default: 1)
  - `--synthetic` - Use synthetic frames
  - `--verbose` - Detailed timing breakdown
- **Output**: Comprehensive timing report with bottleneck analysis
- **Status**: Video performance analysis tool

#### Data Processing & Validation

**`scripts/generate_masks.py`** — Automated Mask Generation from Annotations
- **Purpose**: Create binary segmentation masks from coordinate annotations
- **Type**: Data preprocessing utility
- **Use Case**: After manual annotation, before training
- **Features**:
  - Batch process from JSON annotations
  - Support for multiple mask types (pupil, iris, background)
  - Automated label class assignment
  - Size consistency checking
  - Quality verification
- **Usage**:
  ```bash
  python scripts/generate_masks.py \
    --image-dir clinical_data/raw_images \
    --output-dir clinical_data/annotations/masks \
    --annotation-file clinical_data/annotations/annotations.json \
    --num-classes 3
  ```
- **Parameters**:
  - `--image-dir <PATH>` - Directory with original images
  - `--output-dir <PATH>` - Where to save PNG masks
  - `--annotation-file <PATH>` - JSON with coordinates
  - `--num-classes <N>` - Number of classes (default: 3)
  - `--overwrite` - Overwrite existing masks
- **Time**: ~1-5 seconds per image
- **Output**: Binary PNG masks matching image resolution
- **Status**: Data preparation tool

**`scripts/verify_data.py`** — Comprehensive Data Integrity Verification
- **Purpose**: Validate training dataset health and detect quality issues
- **Type**: Pre-training validation utility
- **Comprehensive Checks**:
  - Image format and readability
  - Mask presence and alignment
  - Pixel value ranges validation
  - Class distribution analysis
  - Corruption detection
  - Size consistency checking
  - Statistics per class per image
- **Usage**:
  ```bash
  python scripts/verify_data.py \
    --image-dir clinical_data/training_data/images \
    --mask-dir clinical_data/training_data/masks \
    --model-path models/best_model.pth \
    --verbose
  ```
- **Parameters**:
  - `--image-dir <PATH>` - Training images
  - `--mask-dir <PATH>` - Training masks
  - `--model-path <PATH>` - Optional: verify against model
  - `--verbose` - Detailed output
  - `--fix-issues` - Attempt to fix detected issues
- **Exit Codes**:
  - 0 = All checks passed
  - 1 = Warnings found (data usable with caution)
  - 2 = Critical issues (fix before training)
- **Output**: Detailed validation report with recommendations
- **Status**: Pre-training validation tool

**`scripts/check_files.py`** — File Organization & Consistency Verification
- **Purpose**: Verify file structure and pairing consistency
- **Type**: File system audit utility
- **Checks**:
  - All images have corresponding masks
  - Filename format validation
  - No orphaned files
  - Directory structure validation
  - File naming convention compliance
  - Duplicate detection
- **Usage**:
  ```bash
  python scripts/check_files.py \
    --path clinical_data/training_data \
    --image-ext jpg \
    --mask-ext png
  ```
- **Parameters**:
  - `--path <PATH>` - Directory to check
  - `--image-ext <EXT>` - Image extension (default: jpg)
  - `--mask-ext <EXT>` - Mask extension (default: png)
  - `--fix` - Auto-fix issues if possible
- **Status**: Setup validation tool

**`scripts/diagnose_detection.py`** — Detection Performance Analysis
- **Purpose**: Analyze and diagnose detection performance on image sets
- **Type**: Performance analysis and debugging tool
- **Analysis**:
  - Per-image confidence scores
  - Quality distribution statistics
  - Detection failure rate analysis
  - Confidence histogram
  - Failed/low-confidence image identification
  - Detailed error logs
- **Usage**:
  ```bash
  python scripts/diagnose_detection.py \
    --image-dir clinical_data/clean \
    --model models/best_model.pth \
    --output-dir diagnostic_output \
    --threshold 0.8
  ```
- **Parameters**:
  - `--image-dir <PATH>` - Images to analyze
  - `--model <PATH>` - Model to use
  - `--output-dir <PATH>` - Where to save diagnostic reports
  - `--threshold <0-1>` - Confidence threshold (default: 0.7)
  - `--visualize` - Generate visualization images
- **Output**:
  - CSV report with per-image metrics
  - Confidence histograms
  - Failed detection list
  - Recommendations for model improvement
- **Status**: Troubleshooting and analysis tool

#### Testing & Validation

**`scripts/test_video_pipeline.py`** — Video Processing Pipeline Unit Tests
- **Purpose**: Verify video processing component functionality
- **Type**: Test suite for CI/CD integration
- **Tests Covered**:
  - Video file reading (multiple formats)
  - Frame extraction and caching
  - Real-time processing pipeline
  - Batch processing
  - Error handling and edge cases
- **Usage**:
  ```bash
  python scripts/test_video_pipeline.py
  python scripts/test_video_pipeline.py --verbose
  ```
- **Output**: Test results with pass/fail status
- **When to Run**: After code changes, before deployment
- **Status**: Development and CI tool

---

## Complete Package Requirements

### Dependencies Summary

The project requires the following packages to be installed via `requirements.txt`:

```
torch>=1.12.0                              # Deep learning framework (PyTorch)
torchvision>=0.13.0                        # Vision models and transforms
segmentation-models-pytorch>=0.3.0         # Pre-built segmentation architectures
albumentations>=1.3.0                      # Advanced image augmentation
opencv-python>=4.6.0                       # Computer vision processing
numpy>=1.21.0                              # Numerical computing
Pillow>=9.0.0                              # Image file handling
tqdm>=4.64.0                               # Progress bar utilities
matplotlib>=3.5.0                          # 2D plotting and visualization
scikit-learn>=1.0.0                        # Machine learning utilities
```

### Installation & Verification

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Verify installation:**
```bash
python -c "
import torch
import cv2
import numpy as np
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ OpenCV: {cv2.__version__}')
print(f'✓ NumPy: {np.__version__}')
print(f'✓ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ CUDA Device: {torch.cuda.get_device_name(0)}')
"
```

### Package Purposes

| Package | Purpose | Used For |
|---------|---------|----------|
| `torch` | Deep learning framework | Model training, inference, tensor operations |
| `torchvision` | Vision models & transforms | Image preprocessing, model architectures |
| `segmentation-models-pytorch` | Segmentation architectures | U-Net, encoder-decoder models |
| `albumentations` | Image augmentation | Data augmentation during training |
| `opencv-python` | Computer vision processing | Image I/O, preprocessing, visualization |
| `numpy` | Numerical computing | Matrix operations, image processing |
| `Pillow` | Image manipulation | Image file handling, format conversion |
| `tqdm` | Progress tracking | Progress bars in training/processing |
| `matplotlib` | Data visualization | Training plots, result visualization |
| `scikit-learn` | ML utilities | Metrics, preprocessing, validation |

### GPU Support

**For NVIDIA GPUs (CUDA):**
```bash
# Verify CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For Apple Silicon Macs:**
```bash
# PyTorch includes Metal Performance Shaders support
python -c "import torch; print(torch.backends.mps.is_available())"
```

**For CPU-only (development):**
```bash
# Use default PyTorch installation (CPU version)
pip install -r requirements.txt
```

---

## Commands Summary Table

| Command | Purpose | Type | Example | Output |
|---------|---------|------|---------|--------|
| `python launch_gui.py` | Launch main GUI | App | `python launch_gui.py` | Interactive window |
| `python launch_gui.py image -i img.jpg` | Process single image | CLI | `python launch_gui.py image -i eye.jpg` | Annotated image + CSV |
| `python launch_gui.py video -i vid.mp4` | Process video file | CLI | `python launch_gui.py video -i video.mp4 -o out.mp4` | Video file + CSV |
| `python launch_gui.py camera` | Live camera | CLI | `python launch_gui.py camera` | Live display |
| `python check_training_data.py` | Validate training data | Utility | `python check_training_data.py` | Validation report |
| `python debug_single_image.py <img>` | Debug image detection | Debug | `python debug_single_image.py eye.jpg` | Diagnostic output |
| `python scripts/train_model.py` | Train detection model | Training | `python scripts/train_model.py --epochs 300` | Model checkpoint |
| `python scripts/annotate_data.py` | Interactive annotation | Data Tool | `python scripts/annotate_data.py` | Annotations JSON + masks |
| `python scripts/annotate_live_video.py --video vid.mp4` | Live video annotation | Data Tool | `python scripts/annotate_live_video.py --video video.mp4` | Annotations + trained model |
| `python scripts/process_video.py --input v.mp4` | Batch video processing | Utility | `python scripts/process_video.py --input video.mp4 --csv out.csv` | Video + CSV results |
| `python scripts/generate_masks.py` | Generate training masks | Data Prep | `python scripts/generate_masks.py --image-dir raw --output-dir masks` | PNG mask files |
| `python scripts/verify_data.py` | Verify dataset quality | Validation | `python scripts/verify_data.py --image-dir images --mask-dir masks` | Validation report |
| `python scripts/check_files.py` | Check file consistency | Validation | `python scripts/check_files.py --path clinical_data` | Consistency report |
| `python scripts/diagnose_detection.py` | Diagnose detection performance | Analysis | `python scripts/diagnose_detection.py --image-dir images --model model.pth` | Diagnostic report |
| `python scripts/export_onnx.py --model m.pth` | Export to ONNX | Conversion | `python scripts/export_onnx.py --model best_model.pth` | ONNX model file |
| `python scripts/benchmark_fps.py --model m.pth` | Benchmark inference speed | Profiling | `python scripts/benchmark_fps.py --model best_model.pth` | FPS results |
| `python scripts/benchmark_video_speed.py --input vid.mp4` | Benchmark video processing | Profiling | `python scripts/benchmark_video_speed.py --input video.mp4 --frames 200` | Speed report |
| `python scripts/test_video_pipeline.py` | Run pipeline tests | Testing | `python scripts/test_video_pipeline.py --verbose` | Test results |

---

## Project Structure

```
Pupil-Limbus-detector/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── launch_gui.py                      # Main GUI entry point
├── check_training_data.py             # Training data validation script
├── debug_single_image.py              # Debug/test single image
│
├── pupil_tracking/                    # Main package
│   ├── __init__.py
│   ├── detection.py                   # Pupil candidate detection
│   ├── preprocessing.py               # Image preprocessing
│   ├── image_interface.py             # Image I/O wrapper
│   ├── logger.py                      # Audit logging system
│   ├── clinical_debug.py              # Diagnostic utilities
│   ├── run_realtime.py                # Real-time processing
│   │
│   ├── ann otation/
│   │   ├── annotation_tool.py         # Interactive annotation GUI
│   │   ├── annotation_converter.py    # Format converters
│   │   └── mask_generator.py          # Automated mask generation
│   │
│   ├── calibration/
│   │   ├── calibration.py             # Camera calibration
│   │   └── camera_calibration.py      # Advanced calibration
│   │
│   ├── core/
│   │   ├── detector.py                # Main detection pipeline
│   │   ├── geometric_fit.py           # Ellipse fitting
│   │   ├── contour_filtering.py       # Contour validation
│   │   ├── corneal_center.py          # Reflection detection
│   │   ├── limbus_detector.py         # Iris boundary detection
│   │   ├── smart_fitter.py            # Advanced fitting (NEW)
│   │   └── confidence.py              # Confidence scoring
│   │
│   ├── interface/
│   │   ├── gui_app.py                 # Main GUI application
│   │   ├── api.py                     # REST API (if enabled)
│   │   └── widgets/                   # Custom GUI components
│   │
│   ├── ml/
│   │   ├── architecture.py            # U-Net model definition
│   │   ├── trainer.py                 # Training loop
│   │   ├── dataset.py                 # PyTorch Dataset class
│   │   ├── losses.py                  # Loss functions
│   │   ├── postprocess.py             # Post-processing utilities
│   │   ├── inference.py               # Inference pipeline
│   │   └── fast_inference.py          # Optimized inference (NEW)
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── filters.py                 # Image filters
│   │   └── normalization.py           # Normalization utilities
│   │
│   ├── utils/
│   │   ├── config.py                  # Configuration management
│   │   ├── logger.py                  # Logging utilities
│   │   └── helpers.py                 # Helper functions
│   │
│   ├── video/
│   │   ├── video_processor.py         # Video file processing
│   │   ├── camera_processor.py        # Live camera handling
│   │   ├── frame_buffer.py            # Frame caching
│   │   └── optimized_processor.py     # Performance optimizations (NEW)
│   │
│   └── tests/
│       ├── test_detection.py
│       ├── test_training.py
│       └── test_preprocessing.py
│
├── scripts/                           # Standalone utilities
│   ├── train_model.py                 # Model training entry point
│   ├── run_epoch.py                   # Single epoch training test
│   ├── annotate_data.py               # Batch annotation tool
│   ├── annotate_live_video.py         # Live video annotation with retraining
│   ├── generate_masks.py              # Generate training masks
│   ├── check_files.py                 # File validation & consistency
│   ├── verify_data.py                 # Data integrity check
│   ├── diagnose_detection.py          # Detection diagnostics
│   ├── benchmark_fps.py               # Performance benchmarking (NEW)
│   ├── benchmark_video_speed.py       # Video speed benchmarking (NEW)
│   ├── process_video.py               # Batch video processing
│   ├── export_onnx.py                 # Model export for inference
│   └── test_video_pipeline.py         # Video pipeline unit tests
│
├── clinical_data/                     # Training dataset
│   ├── annotations/
│   │   ├── annotations.json           # Annotation metadata
│   │   └── masks/                     # Segmentation masks
│   ├── annoted/                       # Annotated images
│   ├── clean/                         # Cleaned/processed images
│   ├── diagnostic_output/             # Diagnostics results
│   └── training_data/                 # Train/val split
│
├── models/
│   ├── best_model.pth                 # Pre-trained model weights
│   └── checkpoint_meta.json           # Training metadata
│
├── logs/
│   └── audit_*.jsonl                  # Audit log files
│
└── diagnostic_output/                 # Generated diagnostics
```

---

## Quick Start Guide

### Option 1: Launch the GUI Application (Recommended for Beginners)

```bash
python launch_gui.py
```

This opens an interactive GUI where you can:
- Load and analyze single images
- Process videos or camera feeds
- View detection results with visualizations
- Adjust detection parameters in real-time
- Export results

### Option 2: Process a Single Image

```bash
python launch_gui.py image -i path/to/eye_image.jpg
```

Output includes:
- Detected pupil center and radius
- Limbus boundary
- Quality assessment
- Confidence scores

### Option 3: Process a Video File

```bash
python launch_gui.py video -i path/to/video.mp4
```

Outputs:
- Frame-by-frame detection results
- Video file with annotations
- CSV with detection coordinates

### Option 4: Live Camera Processing

```bash
python launch_gui.py camera
```

Features:
- Real-time detection from webcam
- Live visualization
- Frame saving on demand
- Statistics tracking

---

## Data Annotation Workflow

### Overview

The annotation process converts raw eye images into training data consisting of:
1. Original eye images
2. Segmentation masks (pupil, iris, background)
3. Annotation metadata (coordinates, quality notes)

### Step 1: Prepare Raw Images

1. Collect eye images in a directory:
```
clinical_data/raw_images/
├── eye_01.jpg
├── eye_02.jpg
├── eye_03.jpg
└── ...
```

**Image Requirements:**
- Format: JPEG, PNG, or BMP
- Resolution: 512x512 or higher recommended
- Clear eye visibility
- Consistent lighting
- No extreme close-ups or distant views

### Step 2: Launch Annotation Tool

Start the interactive annotation GUI:

```bash
python scripts/annotate_data.py
```

Or directly:

```python
from pupil_tracking.annotation.annotation_tool import launch_annotation_tool
launch_annotation_tool()
```

### Step 3: Manual Annotation Interface

In the annotation tool GUI:

1. **Load Image**: Click "Load Image" and select an eye image
2. **Mark Pupil Center**: Click the center of the pupil
3. **Mark Pupil Boundary**: Click around the pupil edge to define radius
4. **Mark Limbus**: Click to define iris boundary points
5. **Verify**: System shows detected contours
6. **Save**: Click "Save Annotation" - creates:
   - `annotations.json` - metadata with coordinates
   - `masks/eye_XX.png` - binary pupil mask
   - `masks/eye_XX_limbus.png` - limbus mask
   - `verify/eye_XX_verify.png` - visual verification image

### Step 4: Alternative - Automated Mask Generation

If you have manual pixel coordinates, generate masks programmatically:

```bash
python scripts/generate_masks.py \
    --image-dir clinical_data/raw_images \
    --output-dir clinical_data/annotations/masks \
    --annotation-file clinical_data/annotations/annotations.json
```

**Command Explanation:**
- `--image-dir`: Directory containing original eye images
- `--output-dir`: Where to save generated mask files
- `--annotation-file`: JSON with coordinate data

### Step 5: Validate Annotations

Check data quality before training:

```bash
python check_training_data.py
```

This performs:
- ✓ Image existence validation
- ✓ Mask-image pairing verification
- ✓ Mask quality checks (connectivity, size)
- ✓ Class distribution analysis
- ✓ Identifies corrupted or misaligned data
- ✓ Provides correction suggestions

**Output Example:**
```
============================================================
TRAINING DATA DIAGNOSTIC
============================================================
✓ Image directory: clinical_data/annoted (87 images)
✓ Mask directory: clinical_data/annotations/masks (87 masks)
✓ Class distribution:
  - Pupil:     4,234,567 pixels (12.3%)
  - Iris:      7,892,341 pixels (22.8%)
  - Background: 21,873,092 pixels (64.9%)
✓ All images paired correctly
⚠ Warning: 2 masks have scattered small regions (eye_15, eye_23)
  
OVERALL: READY FOR TRAINING
```

### Step 6: Organize Training Data

Structure for training:

```
clinical_data/training_data/
├── images/
│   ├── eye_01.jpg
│   ├── eye_02.jpg
│   └── ...
├── masks/
│   ├── eye_01.png
│   ├── eye_02.png
│   └── ...
└── annotations.json
```

---

## Model Training

### Overview

The project uses a U-Net architecture with ResNet-34 encoder for segmenting eye regions (pupil, iris, background).

### Architecture Details

**Model: U-Net with ResNet-34 Encoder**
- **Encoder**: ResNet-34 pretrained on ImageNet
- **Decoder**: 4-level upsampling with skip connections
- **Input**: 512×512 RGB images
- **Output**: 3-class segmentation (background, pupil, iris)
- **Parameters**: ~25M trainable parameters

**Key Features:**
- Multi-scale feature extraction
- Skip connections for detail preservation
- Cross-entropy loss with class weighting
- Learning rate scheduling
- Early stopping mechanism

### Step 1: Prepare Training Data

Ensure your data is organized (see Data Annotation section):

```
clinical_data/training_data/
├── images/
├── masks/
└── annotations.json
```

Run diagnostic:

```bash
python check_training_data.py
```

### Step 2: Configure Training Parameters

Edit or override training configuration:

```bash
# Create/edit pupil_tracking/utils/config.py or use command-line args
python scripts/train_model.py --help
```

**Available Parameters:**

```
--epochs                  Number of training epochs (default: 200)
--batch-size             Batch size (default: 16)
--lr                     Learning rate (default: 0.001)
--input-size             Model input resolution (default: 512)
--annotation-path        Path to annotations.json
--image-dir              Directory with training images
--mask-dir               Directory with mask images
--device                 Device to train on [auto|cpu|cuda|mps]
```

### Step 3: Start Training

**Basic Training:**
```bash
python scripts/train_model.py
```

**Custom Configuration:**
```bash
python scripts/train_model.py \
    --epochs 300 \
    --batch-size 8 \
    --lr 0.0005 \
    --input-size 512 \
    --image-dir clinical_data/training_data/images \
    --mask-dir clinical_data/training_data/masks \
    --device cuda
```

**Command Explanation:**
- `--epochs 300`: Train for 300 epochs
- `--batch-size 8`: Process 8 images per batch
- `--lr 0.0005`: Set learning rate to 0.0005
- `--input-size 512`: Input resolution 512×512
- `--device cuda`: Use NVIDIA GPU

### Training Process Details

**Phase 1: Initialization**
- Loads ResNet-34 backbone pretrained on ImageNet
- Initializes decoder layers randomly
- Sets up optimizer (Adam) and scheduler
- Prints model summary and parameter count

**Phase 2: Training Loop (per epoch)**
```
For each batch:
  1. Load batch of images and masks
  2. Forward pass through U-Net
  3. Compute segmentation loss (weighted cross-entropy)
  4. Backward pass and gradient update
  5. Update running metrics
  6. Display progress bar

After epoch:
  - Validate on holdout set
  - Check early stopping criteria
  - Save checkpoint if improved
  - Adjust learning rate if plateau
```

**Phase 3: Validation**
- Evaluates on validation set every epoch
- Computes metrics:
  - **Segmentation Loss**: Cross-entropy loss
  - **Dice Coefficient**: Overlap metric per class
  - **IoU**: Intersection over Union per class
  - **Accuracy**: Pixel-level accuracy

**Phase 4: Checkpointing**
- Saves best model weights to `models/best_model.pth`
- Saves training metadata to `models/checkpoint_meta.json`
- Early stopping if no improvement for 20 epochs

### Step 4: Training Monitoring

**View Real-time Progress:**
```
Epoch 001/300 [████░░░░░░░░░░░░░░░░░░░░░░] 15/200
  Loss: 0.842  Val Loss: 0.756  Dice: 0.834  Time: 2.3s
```

**Training Artifacts:**
- `logs/audit_*.jsonl` - Detailed event logs
- `models/best_model.pth` - Best checkpoint weights
- `models/checkpoint_meta.json` - Training metadata

### Step 5: Evaluate Trained Model

```bash
python scripts/verify_data.py --model-path models/best_model.pth
```

**Evaluation Metrics:**
- Dice coefficient per class
- Intersection over Union (IoU)
- Sensitivity (recall)
- Specificity
- Per-image and per-class statistics

### Step 6: Export Model (Optional)

Export to ONNX format for deployment:

```bash
python scripts/export_onnx.py \
    --model-path models/best_model.pth \
    --output-path models/best_model.onnx
```

This creates a model usable in production without PyTorch dependency.

### Training Best Practices

1. **Data Preparation**
   - Ensure masks are clean (no small scattered regions)
   - Check image-mask pairing is correct
   - Verify class balance (~60% background, ~23% iris, ~12% pupil)

2. **Batch Size**
   - Start with 8-16 for VRAM < 8GB
   - Use 32+ for 24GB+ VRAM
   - Larger batches = more stable gradients

3. **Learning Rate**
   - Start with 0.001 for new training
   - Reduce by 0.5x if loss oscillates
   - Use 0.0001 for fine-tuning

4. **Epochs**
   - 200-300 typical for convergence
   - Monitor validation loss for early stopping
   - Stop if no improvement for 20+ epochs

5. **Augmentation**
   - Rotation: ±15 degrees
   - Brightness/Contrast: ±10%
   - Slight zoom: 0.9-1.1x
   - Horizontal flip: 50% probability

6. **GPU Memory Tips**
   - Reduce batch size if OOM errors
   - Use gradient accumulation
   - Monitor with `nvidia-smi` (NVIDIA) or `Activity Monitor` (Mac)

---

## Running the Application

### Main GUI Application

#### Basic Launch

```bash
python launch_gui.py
```

Opens the main Tkinter GUI with tabs for:
- **Single Image Analysis**
- **Batch Processing**
- **Video Processing**
- **Real-time Camera**
- **Settings & Configuration**

#### Command-Line Modes

**GUI Mode (Default):**
```bash
python launch_gui.py gui
```

**Single Image Processing:**
```bash
python launch_gui.py image -i path/to/image.jpg
```

Options:
- `-i, --input`: Path to input image
- `--model`: Custom model path (default: models/best_model.pth)
- `--output`: Save results directory

**Video Mode:**
```bash
python launch_gui.py video -i path/to/video.mp4
```

Options:
- `-i, --input`: Path to video file
- `-o, --output`: Output video path
- `--start-frame`: Start processing from frame N
- `--end-frame`: Stop at frame N
- `--stride`: Process every Nth frame

**Camera Mode:**
```bash
python launch_gui.py camera --camera-id 0
```

Options:
- `--camera-id`: Webcam index (0 for default)
- `--resolution`: Output resolution (default: 1280x720)
- `--fps`: Target FPS (default: 30)

### GUI Application Features

#### Image Tab
1. **Load Image**: Select JPEG/PNG/BMP file
2. **Analyze**: Runs detection pipeline
3. **View Results**:
   - Pupil center and radius
   - Limbus boundary
   - Quality assessment
   - Confidence scores
4. **Export**: Save annotated image or CSV results

#### Video Tab
1. **Load Video**: Select MP4/AVI/MOV file
2. **Configure**:
   - Processing stride (every Nth frame)
   - Resolution
   - Detection parameters
3. **Process**: Batch analysis
4. **Results**:
   - Output video with overlays
   - CSV with per-frame coordinates
   - Statistics summary

#### Real-time Camera Tab
1. **Select Camera**: Choose input device
2. **Adjust Parameters**: Real-time preview
3. **Capture**: Save frames on spacebar press
4. **Statistics**: Live FPS and quality metrics

#### Settings Tab
1. **Detection Parameters**:
   - Threshold values
   - Morphological operations
   - Contour filtering
2. **Model Settings**:
   - Device selection (CPU/GPU)
   - Input resolution
   - Confidence threshold
3. **Output Options**:
   - Save annotated images
   - Export formats (CSV, JSON)
   - Verbosity level

### Output Files

After processing, the application generates:

```
output/
├── results.json              # Structured detection data
├── results.csv               # Tabular format
├── annotated_image.jpg       # Visualization
├── detection_log.txt         # Text summary
└── video_results/            # Per-frame data
    └── frame_001_results.json
```

**Sample CSV Output:**
```csv
filename,pupil_center_x,pupil_center_y,pupil_radius,limbus_x,limbus_y,quality,confidence
eye_01.jpg,256.3,512.1,48.2,280.5,535.8,EXCELLENT,0.988
eye_02.jpg,258.1,510.4,47.8,282.1,533.2,EXCELLENT,0.985
```

---

## Inference & Processing

### Programmatic Inference

#### Basic Detection

```python
from pupil_tracking.core.detector import UnifiedDetector
import cv2

# Load detector
detector = UnifiedDetector(model_path="models/best_model.pth")

# Load image
image = cv2.imread("eye_image.jpg")

# Detect
result = detector.detect(image, source="eye_image.jpg")

# Access results
if result.pupil.detected:
    print(f"Pupil: ({result.pupil.ellipse.center_x}, " 
          f"{result.pupil.ellipse.center_y})")
    print(f"Radius: {result.pupil.ellipse.radius} px")
    print(f"Confidence: {result.pupil.confidence:.3f}")
    print(f"Quality: {result.pupil.quality.value}")

if result.limbus.detected:
    print(f"Limbus center: ({result.limbus.center_x}, "
          f"{result.limbus.center_y})")
```

#### Batch Processing

```python
from pathlib import Path
from pupil_tracking.core.detector import UnifiedDetector
import cv2
import json

detector = UnifiedDetector()
results = {}

for image_path in Path("images/").glob("*.jpg"):
    image = cv2.imread(str(image_path))
    result = detector.detect(image, source=str(image_path))
    
    results[image_path.name] = {
        "pupil": {
            "center": (result.pupil.ellipse.center_x, 
                      result.pupil.ellipse.center_y),
            "radius": result.pupil.ellipse.radius,
            "confidence": float(result.pupil.confidence),
        },
        "quality": result.overall_quality.value
    }

# Save results
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
```

#### Video Processing

```python
from pupil_tracking.video.video_processor import VideoProcessor
from pupil_tracking.core.detector import UnifiedDetector

detector = UnifiedDetector()
processor = VideoProcessor(detector)

# Process video
output_path = processor.process_video(
    input_path="video.mp4",
    output_path="output_video.mp4",
    stride=1,  # Process every frame
    resize=(1280, 720),  # Target resolution
)

# Results include frame-by-frame coordinates
```

#### Real-time Camera

```python
from pupil_tracking.video.camera_processor import CameraProcessor
from pupil_tracking.core.detector import UnifiedDetector

detector = UnifiedDetector()
processor = CameraProcessor(detector, camera_id=0)

# Start live processing
processor.run(
    resolution=(1280, 720),
    fps=30,
    flip_horizontal=False,
    display_fps=True,
)
```

### Result Structure

Detection results follow this schema:

```python
class DetectionResult:
    # Pupil detection
    pupil.detected: bool                # Whether pupil detected
    pupil.confidence: float             # 0-1 confidence score
    pupil.quality: QualityEnum          # EXCELLENT to UNUSABLE
    pupil.ellipse.center_x: float       # X coordinate (pixels)
    pupil.ellipse.center_y: float       # Y coordinate (pixels)
    pupil.ellipse.radius: float         # Radius (pixels)
    pupil.ellipse.minor_axis: float     # Secondary axis length
    pupil.ellipse.angle: float          # Rotation in degrees
    
    # Iris/Limbus detection
    limbus.detected: bool               # Iris boundary detected
    limbus.center_x: float              # Limbus center X
    limbus.center_y: float              # Limbus center Y
    limbus.radius: float                # Limbus radius
    
    # Quality metrics
    overall_quality: QualityEnum        # Overall assessment
    overall_confidence: float           # Mean confidence
```

### Quality Levels

Results include a 5-point quality assessment:

| Level | Criteria |
|-------|----------|
| **EXCELLENT** | High contrast, centered, clear boundaries |
| **GOOD** | Minor reflections/shadows, slightly off-center |
| **FAIR** | Moderate artifacts, low contrast regions |
| **POOR** | Significant occlusion, heavy artifacts |
| **UNUSABLE** | Cannot reliably detect pupil/iris |

---

## Configuration

### Configuration File

Configuration is managed through `pupil_tracking/utils/config.py`:

```python
from pupil_tracking.utils.config import get_config, set_config

cfg = get_config()

# Access configuration
print(cfg.model.input_size)           # 512
print(cfg.training.batch_size)        # 16
print(cfg.paths.log_dir)              # logs/

# Modify configuration
cfg.model.device = "cuda"
cfg.detection.threshold_value = 50
set_config(cfg)
```

### Key Configuration Parameters

**Model Settings:**
```python
cfg.model.encoder = "resnet34"          # Backbone architecture
cfg.model.num_classes = 3               # Segmentation classes
cfg.model.input_size = 512              # Input resolution
cfg.model.device = "auto"               # auto/cpu/cuda/mps
cfg.model.pretrained = True             # Use ImageNet weights
```

**Training Settings:**
```python
cfg.training.epochs = 200               # Number of epochs
cfg.training.batch_size = 16            # Batch size
cfg.training.learning_rate = 0.001      # Initial LR
cfg.training.weight_decay = 1e-4        # L2 regularization
cfg.training.patience = 20              # Early stopping patience
```

**Detection Settings:**
```python
cfg.detection.threshold_value = 40              # Primary threshold
cfg.detection.adaptive_block_size = 51         # Adaptive threshold window
cfg.detection.adaptive_c = 12.0                # Adaptive constant
cfg.detection.min_contour_area = 150          # Minimum contour pixels
cfg.detection.max_contour_area = 50000        # Maximum contour pixels
```

**Preprocessing:**
```python
cfg.preprocessing.median_blur_kernel = 5      # Noise reduction
cfg.preprocessing.normalize = True             # 0-1 normalization
cfg.preprocessing.clahe_enabled = True        # Contrast enhancement
```

**Paths:**
```python
cfg.paths.log_dir = "logs/"                    # Audit logs
cfg.paths.model_dir = "models/"                # Model checkpoints
cfg.paths.data_dir = "clinical_data/"          # Dataset
cfg.paths.output_dir = "diagnostic_output/"    # Results
```

### Override via Command Line

```bash
# Training
python scripts/train_model.py --epochs 300 --batch-size 8 --lr 0.0005

# Inference
python launch_gui.py image -i image.jpg --device cuda
```

### Override Programmatically

```python
from pupil_tracking.utils.config import get_config, set_config

cfg = get_config()
cfg.training.epochs = 500
cfg.model.device = "cuda"
set_config(cfg)
```

---

## API Reference

### Core Detection API

#### UnifiedDetector

Main detection pipeline combining traditional and deep learning approaches.

```python
from pupil_tracking.core.detector import UnifiedDetector

detector = UnifiedDetector(
    model_path: str = "models/best_model.pth",
    device: str = "auto",
    config: Config = None
)
```

**Methods:**

```python
# Single image detection
result = detector.detect(
    image: np.ndarray,              # Input image
    source: str = ""                # Image source path
) -> DetectionResult

# Batch detection
results = detector.detect_batch(
    images: List[np.ndarray],
    sources: List[str] = None
) -> List[DetectionResult]

# Get model info
info = detector.get_model_info() -> Dict
```

#### Dataset API

```python
from pupil_tracking.ml.dataset import EyeSegmentationDataset
import torch

dataset = EyeSegmentationDataset(
    image_dir="clinical_data/images",
    mask_dir="clinical_data/masks",
    image_size=512,
    augment=True
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# Iterate
for images, masks in loader:
    # images: shape (16, 3, 512, 512)
    # masks: shape (16, 512, 512)
    pass
```

#### Trainer API

```python
from pupil_tracking.ml.trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

# Train
history = trainer.train()  # Returns dict with loss/metrics

# Save/Load
trainer.save_checkpoint("models/checkpoint.pth")
trainer.load_checkpoint("models/checkpoint.pth")
```

### Utility APIs

#### Logging

```python
from pupil_tracking.utils.logger import AuditLogger

logger = AuditLogger(log_dir="logs/", session_id="inference_v1")

logger.log_event(
    event_type="DETECTION",
    details={
        "image": "eye_01.jpg",
        "pupil_detected": True,
        "confidence": 0.987,
    }
)

logs = logger.get_logs(event_type="DETECTION")
```

#### Image Interface

```python
from pupil_tracking.image_interface import ImageInterface

img_interface = ImageInterface()

# Read image (auto-converts formats)
image = img_interface.read("eye_image.jpg")

# Write image
img_interface.write("output.jpg", image, format="JPEG")

# Validate
is_valid = img_interface.validate(image)
```

---

## Troubleshooting

### Installation Issues

**Problem: CUDA not available despite having NVIDIA GPU**

```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # GPU name
```

**Solution:**
1. Install NVIDIA drivers (latest)
2. Install CUDA toolkit matching PyTorch version
3. Reinstall PyTorch with CUDA support:
   ```bash
   pip install torch torchvision -f https://download.pytorch.org/whl/cu118/torch_stable.html
   ```

**Problem: "ModuleNotFoundError: No module named 'torch'"**

```bash
# Verify virtual environment is activated
which python  # Should show path inside venv/

# Reinstall
pip install --upgrade pip
pip install -r requirements.txt
```

### Training Issues

**Problem: Out of Memory (OOM) errors during training**

```
RuntimeError: CUDA out of memory
```

**Solution:**
1. Reduce batch size: `--batch-size 4`
2. Reduce input size: `--input-size 256`
3. Use CPU: `--device cpu` (slower)
4. Monitor GPU memory: `nvidia-smi`

**Problem: Loss not decreasing / model not training**

**Causes & Solutions:**
- Learning rate too high: Reduce to 0.0001
- Learning rate too low: Increase to 0.001
- Bad data: Run `python check_training_data.py`
- Gradient explosion: Add weight decay `--decay 1e-4`

**Problem: Training is very slow**

**Solution:**
- Use GPU: `--device cuda`
- Increase batch size: `--batch-size 32`
- Reduce input size: `--input-size 256`
- Enable mixed precision (if supported)

### Inference Issues

**Problem: Poor detection quality / Low confidence scores**

1. **Check Image Quality:**
   - High contrast? Low = preprocessing issue
   - Clear pupil boundary? Blurry = acquisition issue
   - Appropriate lighting? Too dark = gain settings

2. **Verify Preprocessing:**
   ```python
   from pupil_tracking.preprocessing import Preprocessor
   
   preprocessor = Preprocessor()
   cleaned_image = preprocessor.preprocess(image)
   
   # Visual inspection
   cv2.imshow("Preprocessed", cleaned_image)
   ```

3. **Adjust Detection Parameters:**
   ```python
   cfg.detection.threshold_value = 35  # Lower = more sensitive
   cfg.detection.min_contour_area = 100
   ```

4. **Re-train with Better Data:**
   - Collect more diverse eye images
   - Improve annotation accuracy
   - Balance class distribution

**Problem: Detection fails on certain image types**

**Solution:**
- Generate synthetic training data for that condition
- Fine-tune model on problematic images
- Use ensemble of multiple models

---

## Testing & Validation

### Running Tests

The project includes a comprehensive test suite in `pupil_tracking/tests/`:

```bash
# Run all tests
python -m pytest pupil_tracking/tests/ -v

# Run specific test file
python -m pytest pupil_tracking/tests/test_detection.py -v

# Run with coverage report
python -m pytest pupil_tracking/tests/ --cov=pupil_tracking --cov-report=html

# Run single test
python -m pytest pupil_tracking/tests/test_detection.py::test_pupil_detection -v
```

### Test Suites

| Test File | Purpose | Commands |
|-----------|---------|----------|
| `test_pipeline.py` | End-to-end pipeline | `pytest test_pipeline.py -v` |
| `test_detection.py` | Detection algorithms | `pytest test_detection.py -v` |
| `test_training.py` | Training pipeline | `pytest test_training.py -v` |
| `test_preprocessing.py` | Image preprocessing | `pytest test_preprocessing.py -v` |
| `test_clinical_accuracy.py` | Clinical validation | `pytest test_clinical_accuracy.py -v` |
| `test_calibration.py` | Geometric calibration | `pytest test_calibration.py -v` |
| `test_stabilization.py` | Temporal smoothing | `pytest test_stabilization.py -v` |
| `test_blink_detection.py` | Blink detection | `pytest test_blink_detection.py -v` |

### Validation Scripts

**Quick Data Validation:**
```bash
python check_training_data.py
```

**Comprehensive Diagnostics:**
```bash
python scripts/verify_data.py --image-dir clinical_data/clean
```

**Detection Performance:**
```bash
python scripts/diagnose_detection.py --image-dir clinical_data/clean --model models/best_model.pth
```

**Ring Classifier Evaluation:**
```bash
python scripts/evaluate_ring_detection.py \
    --image-dir clinical_data/training_data/images \
    --labels clinical_data/ring_labels.json \
    --classifier models/ring_classifier.pth
```

### Performance Benchmarking

**Inference Speed:**
```bash
python scripts/benchmark_fps.py --model models/best_model.pth
```

**Video Processing Speed:**
```bash
python scripts/benchmark_video_speed.py --input sample_video.mp4 --model models/best_model.pth
```

**Expected Performance (GPU):**
- Single image: 50-100 FPS at 512×512
- Video processing: 30-60 FPS at 1080p
- Training: 100-200 images/second on RTX 3090

---

## Advanced Features

### Ring (Suction Cup) Detection

The system includes specialized detection for suction rings in surgical microscopy:

### 1. Ring Labeling Workflow

```bash
# Launch interactive ring labeling tool
python scripts/annotate_ring_data.py \
    --image-dir clinical_data/training_data/images \
    --output clinical_data/ring_labels.json
```

**Controls:**
- `R` - Mark as docked (ring present)
- `N` - Mark as pre-docked (no ring)
- `P` - Mark as partial ring
- `U` - Undo last label
- `S` - Save progress
- `Q` - Save and quit

### 2. Train Ring Classifier

```bash
python scripts/train_ring_classifier.py \
    --image-dir clinical_data/training_data/images \
    --labels clinical_data/ring_labels.json \
    --epochs 50 \
    --batch-size 32 \
    --device cuda
```

### 3. Evaluate Ring Detection

```bash
python scripts/evaluate_ring_detection.py \
    --image-dir clinical_data/training_data/images \
    --labels clinical_data/ring_labels.json \
    --classifier models/ring_classifier.pth
```

**Output Metrics:**
- Classification accuracy
- Precision/Recall per class
- F1 scores
- Confusion matrix

### 4. Using Ring Detection at Inference

```python
from pupil_tracking.core.ring_detector import RingDetector

# Detect ring presence
ring_detector = RingDetector(classifier_path="models/ring_classifier.pth")
result = ring_detector.detect_ring(image)

print(f"Ring present: {result.ring_present}")
print(f"Confidence: {result.confidence}")
print(f"Visibility: {result.visibility}")  # FULL, PARTIAL, NONE

# Adapt detection strategy based on ring
if result.ring_present:
    # Use ring-aware preprocessing
    preprocessed = apply_ring_masking(image)
else:
    # Standard preprocessing
    preprocessed = standard_preprocessing(image)
```

### ONNX Model Export & Deployment

Convert PyTorch model to ONNX for deployment without PyTorch dependency:

```bash
# Export model
python scripts/export_onnx.py \
    --model models/best_model.pth \
    --resolution 320 \
    --verify
```

**Using ONNX Model:**

```python
import onnxruntime
import cv2
import numpy as np

# Load ONNX model
sess = onnxruntime.InferenceSession("models/best_model.onnx")

# Prepare input
image = cv2.imread("eye.jpg")
image_resized = cv2.resize(image, (320, 320))
image_normalized = image_resized.astype(np.float32) / 255.0
image_batch = np.expand_dims(image_normalized, axis=0)  # Add batch dimension

# Run inference
outputs = sess.run(None, {"input": image_batch})
segmentation_mask = outputs[0]

print(f"Segmentation shape: {segmentation_mask.shape}")
```

**ONNX Benefits:**
- 4x smaller file size
- Faster inference (50-100% speedup)
- No PyTorch dependency
- Hardware acceleration (TensorRT, CoreML, etc.)
- Cross-platform compatibility

### Live Video Annotation with Incremental Retraining

Interactive annotation tool that trains model in real-time:

```bash
python scripts/annotate_live_video.py \
    --video clinical_video.mp4 \
    --model models/best_model.pth \
    --auto-retrain \
    --retrain-interval 100
```

**Controls:**
- `SPACE` - Pause/resume and annotate
- `P` - Switch to pupil annotation
- `L` - Switch to limbus annotation
- `ENTER` - Save annotation
- `T` - Trigger incremental retrain
- `S` - Save annotations
- `ESC` - Quit

**Features:**
- Real-time annotation
- Incremental model retraining every N frames
- Active learning - focuses on hard examples
- Continuous model improvement
- Automatic augmentation

### Custom Model Architecture

Train with different backbone architectures:

```bash
# Using ResNet-50 instead of ResNet-34
python scripts/train_model.py \
    --encoder resnet50 \
    --input-size 512 \
    --epochs 300

# Using EfficientNet (lighter)
python scripts/train_model.py \
    --encoder efficientnet-b2 \
    --input-size 360 \
    --epochs 250
```

**Available Architectures:**
- ResNet-18, 34, 50, 101
- EfficientNet-b0 through b7
- DenseNet-121, 169
- VGG-16, 19 (legacy)
- EfficientNet-lite (mobile)

### Multi-Class Segmentation

Train 4-class model that also detects suction rings:

```bash
python scripts/train_model.py \
    --num-classes 4 \
    --annotation-path clinical_data/annotations/annotations.json \
    --ring-labels clinical_data/ring_labels.json \
    --loss-type composite \
    --use-focal
```

**Classes:**
- Class 0: Background
- Class 1: Pupil (dark center)
- Class 2: Iris/Limbus (colored ring)
- Class 3: Suction ring (surgical device)

### ROI Tracking for Video

Enable ROI (Region of Interest) tracking for faster video processing:

```bash
python scripts/process_video.py \
    --input video.mp4 \
    --roi \
    --roi-expand-factor 1.2 \
    --output result.mp4
```

**Benefits:**
- 2-3x faster processing
- Reduced memory usage
- Smoother tracking across frames
- Better handling of motion

### Temporal Smoothing & Kalman Filter

Smooth noisy detections across frames:

```python
from pupil_tracking.video.temporal_smoother import TemporalSmoother

smoother = TemporalSmoother(method="kalman", window_size=5)

# Process detections
for frame_idx, detection in enumerate(detections):
    smoothed = smoother.smooth(detection)
    print(f"Frame {frame_idx}: smoothed_center = {smoothed.center}")
```

**Smoothing Methods:**
- `kalman` - Kalman filter (best for tracking)
- `moving_avg` - Moving average (simple, fast)
- `exponential` - Exponential smoothing (adaptive)
- `median` - Median filter (robust to outliers)

### Data Augmentation Pipeline

View and customize augmentation:

```python
from pupil_tracking.ml.dataset import get_augmentation

# Get default augmentation
aug = get_augmentation(input_size=512)

# Apply augmentation
augmented = aug(image=original_image, mask=original_mask)
aug_image = augmented["image"]
aug_mask = augmented["mask"]
```

**Default Augmentations:**
- Rotation: ±15  degrees
- Brightness/Contrast: ±10%
- Elastic distortion: slight warping
- Zoom: 0.9-1.1x
- Horizontal flip: 50%
- Shift: ±10% of image size

---

## Performance Optimization Tips

### 1. GPU Acceleration

**Check GPU Usage:**
```bash
# NVIDIA GPU
nvidia-smi -l 1

# Apple Metal
python -c "import torch; print(f'Metal available: {torch.backends.mps.is_available()}')"
```

**Enable GPU:**
```bash
python launch_gui.py --device cuda
```

### 2. Batch Processing

Process multiple images efficiently:

```bash
python scripts/process_video.py \
    --input video.mp4 \
    --stride 2 \
    --batch-size 32 \
    --device cuda
```

### 3. Resolution Trade-offs

| Resolution | FPS (RTX2080) | Memory | Accuracy |
|------------|---------------|--------|----------|
| 256×256    | 120+ FPS      | 2 GB   | 94%      |
| 320×320    | 80-100 FPS    | 3 GB   | 96%      |
| 512×512    | 40-60 FPS     | 6 GB   | 98%      |
| 768×768    | 20-30 FPS     | 12 GB  | 98.5%    |

### 4. Model Compilation

Use torch.compile for up to 2x speedup:

```bash
python launch_gui.py --compile
```

### 5. Mixed Precision Training

Train faster with FP16:

```bash
python scripts/train_model.py \
    --mixed-precision \
    --batch-size 32
```

**Benefits:**
- 2x speedup
- 50% memory reduction
- Negligible accuracy loss

---

## Deployment Options

### Option 1: PyTorch Model (Current)

```python
detector = UnifiedDetector(model_path="models/best_model.pth")
```

### Option 2: ONNX (Recommended)

```bash
# Export
python scripts/export_onnx.py --model models/best_model.pth

# Use
detector = ONNXDetector(model_path="models/best_model.onnx")
```

### Option 3: Docker Container

```dockerfile
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "launch_gui.py"]
```

### Option 4: Real-time System (C++ Deployment)

Export and use with TensorRT for maximum performance:

```bash
# Convert ONNX to TensorRT
python -c "from torch2trt import torch2trt; ..."
```

---

## Frequently Asked Questions (FAQ)

**Q: What's the minimum dataset size for training?**
A: Start with 50-100 annotated images. More data (500+) significantly improves accuracy.

**Q: Can I use this on mobile devices?**
A: Yes, export to ONNX and use ONNX Runtime Mobile.

**Q: How do I improve detection accuracy?**
A: 1) More diverse training data 2) Better annotation quality 3) Longer training 4) Tune thresholds

**Q: What if I don't have a GPU?**
A: The system works on CPU (slower). Use stride=2 to reduce processing time.

**Q: Can I detect both eyes simultaneously?**
A: Process each eye separately, or modify dataset to include stereo pairs.

**Q: How do I integrate this into my application?**
A: Use the programmatic API (see API Reference section) or REST API if configured.

---

## References & Further Reading

### Key Papers

1. **U-Net Architecture:**
   - Ronneberger et al. (2015) - "U-Net: Convolutional Networks for Biomedical Image Segmentation"

2. **ResNet Backbone:**
   - He et al. (2015) - "Deep Residual Learning for Image Recognition"

3. **Semantic Segmentation:**
   - Long et al. (2015) - "Fully Convolutional Networks for Semantic Segmentation"

### Related Projects

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [Albumentations - Image Augmentation](https://albumentations.ai/)
- [OpenCV - Computer Vision](https://opencv.org/)
- [PyTorch - Deep Learning](https://pytorch.org/)

### Documentation

- [PyTorch Documentation](https://pytorch.org/docs/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [Segmentation Models Documentation](https://smp.readthedocs.io/)

---

## Version History

**v2.0.0 (March 2026)** — Current Release
- Complete rewrite with modular architecture
- Ring detection system
- Optimized video processing
- ONNX export support
- Comprehensive documentation
- Production-ready testing

**v1.5.0 (January 2026)**
- ROI tracking
- Temporal smoothing
- Performance optimizations

**v1.0.0 (Initial Release)**
- Basic pupil/iris detection
- Training pipeline
- GUI application

---

## Contact & Support

**Author:** Prince J Rathnakara

**How to Get Help:**
1. Check this README (especially Troubleshooting section)
2. Search GitHub issues for similar problems
3. Open a new GitHub issue with:
   - Error message (full stack trace)
   - Your system info (OS, Python version, GPU?)
   - Steps to reproduce
   - Expected vs actual behavior

**Report Bugs:**
Include:
- Test image (if possible)
- Full error message
- Code snippet that reproduces issue
- System specifications

---

## Acknowledgments

This project combines state-of-the-art deep learning (U-Net architecture) with classical computer vision (contour detection, geometric fitting) to provide a robust solution for clinical eye tracking applications.

Thank you to the open-source communities behind PyTorch, OpenCV, and Segmentation Models.

---

**Project Status:** Active Development ✅

**Last Updated:** March 12, 2026

**Version:** 2.0.0

**Python:** 3.8+

**License:** MIT

**Problem: Real-time processing is slow (low FPS)**

1. **Profile Performance:**
   ```bash
   python scripts/benchmark_fps.py --model models/best_model.pth
   ```

2. **Optimize Resolution:**
   - Use 256×256 for 60 FPS
   - Use 512×512 for 30 FPS (current)
   - Depends on GPU

3. **Skip Frames:**
   ```python
   processor.process_video(
       input_path="video.mp4",
       stride=2,  # Process every 2nd frame
   )
   ```

4. **Use Optimized Model:**
   ```bash
   python scripts/export_onnx.py --model models/best_model.pth
   # Then use ONNX runtime for 2-3x speedup
   ```

### Annotation Issues

**Problem: Mask generation produces invalid masks**

**Solution:**
```bash
# Validate generated masks
python scripts/verify_data.py --mask-dir clinical_data/annotations/masks
```

**Problem: Annotation tool crashes when loading image**

**Solution:**
- Ensure image format is JPEG/PNG
- Check image not corrupted: `file image.jpg`
- Verify image has reasonable size (< 50MB)
- Try: `python debug_single_image.py image.jpg`

### Logging & Debugging

**Enable Verbose Logging:**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Check Audit Logs:**

```bash
# View latest audit log
tail -f logs/audit_*.jsonl
```

**Debug Single Image:**

```bash
python debug_single_image.py path/to/image.jpg
```

Produces detailed diagnostic output including:
- Preprocessing steps
- Detection statistics
- Fitting results
- Quality scores

---

## Technical Architecture

### Detection Pipeline

```
Input Image
    ↓
[PREPROCESSING]
  - Median filtering (noise removal)
  - Normalization (0-1 range)
  - CLAHE (contrast enhancement)
    ↓
[MULTI-STRATEGY DETECTION]
  Strategy 1: Adaptive + Otsu Thresholding
  Strategy 2: Deep Learning Segmentation (U-Net)
  Strategy 3: Morphological Operations
    ↓
[CONTOUR EXTRACTION & FILTERING]
  - Area gating
  - Aspect ratio filtering
  - Circularity assessment
    ↓
[CANDIDATE SELECTION]
  - Top-N by confidence
  - Geometrically valid
    ↓
[GEOMETRIC FITTING]
  - Ellipse fitting (Fitzgibbon algorithm)
  - Parameter validation
    ↓
[QUALITY ASSESSMENT]
  - Confidence scoring
  - Quality level assignment
    ↓
[LIMBUS DETECTION]
  - Iris boundary detection
  - Iris center estimation
    ↓
OUTPUT: DetectionResult
  - Pupil center, radius, confidence
  - Limbus center, radius
  - Quality level, overall confidence
```

### Model Architecture

**U-Net with ResNet-34 Encoder:**

```
Input: 512×512×3 RGB Image
    ↓
[ENCODER - ResNet-34]
  Block 1: 64 filters, 256×256    (stride 2)
  Block 2: 128 filters, 128×128   (stride 2)
  Block 3: 256 filters, 64×64     (stride 2)
  Block 4: 512 filters, 32×32     (stride 2)
    ↓
[BOTTLENECK]
  Atrous Spatial Pyramid Pooling (ASPP)
    ↓
[DECODER]
  Upsample 4: 256 filters, 64×64   + skip from encoder block 3
  Upsample 3: 128 filters, 128×128 + skip from encoder block 2
  Upsample 2: 64 filters, 256×256  + skip from encoder block 1
  Upsample 1: 32 filters, 512×512
    ↓
[OUTPUT LAYER]
  3 channels (background, pupil, iris)
    ↓
OUTPUT: 512×512×3 Probability Maps
```

**Training Loss:**

Weighted Cross-Entropy:
```
L = -Σ w_c * y_c * log(ŷ_c)

where:
  w_background ≈ 0.5 (majority class)
  w_pupil ≈ 1.0 (minority class)
  w_iris ≈ 0.8 (intermediate)
```

### Performance Characteristics

| Metric | Value |
|--------|-------|
| **Model Size** | ~98 MB (PyTorch), ~24 MB (ONNX) |
| **Inference Time (512×512)** | ~50-100ms (GPU), ~500ms (CPU) |
| **Training Memory** | ~4-6 GB (batch size 16) |
| **FPS (Real-time)** | 10-20 FPS (512×512, GPU) |
| **Accuracy (Dice)** | 0.92-0.95 (well-annotated data) |

### Dependencies Graph

```
torch, torchvision
    ↓
segmentation-models-pytorch (smp)
    ↓
pupil_tracking.ml.architecture
    ↓
pupil_tracking.ml.trainer
    ↓
pupil_tracking.core.detector
    ↓
pupil_tracking.interface.gui_app
```

---

## Contributing

Issues and pull requests are welcome! Please ensure:

1. Code follows PEP 8 style guide
2. All tests pass: `pytest pupil_tracking/tests/`
3. New features include tests
4. Update documentation

## License

This project is distributed under the MIT License. See LICENSE file for details.

## Citation

If you use this project in research, please cite:

```bibtex
@software{pupil_limbus_2026,
  title={Pupil Tracking and Limbus Detector},
  author={Prince J Rathnakara},
  year={2026},
  url={https://github.com/Prince649294u83/Pupil-Limbus-detector}
}
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing issues for similar problems
- Review the Troubleshooting section above

---

**Last Updated:** March 3, 2026
**Version:** 2.0.0
