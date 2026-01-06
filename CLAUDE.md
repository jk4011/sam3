# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

SAM 3 (Segment Anything Model 3) is a unified foundation model for promptable segmentation in images and videos. It can detect, segment, and track objects using text or visual prompts (points, boxes, masks). The model has 848M parameters and consists of a detector and tracker that share a vision encoder.

## Installation and Setup

### Environment Setup
There already exists a python environment named `ludivg` for this project. 

```bash
# Create conda environment
conda create -n ludvig python=3.12
conda activate ludvig

# Install PyTorch with CUDA
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install package
pip install -e .

# For notebooks
pip install -e ".[notebooks]"

# For development and training
pip install -e ".[dev,train]"
```

### Hugging Face Authentication
Before using SAM 3, you must request access to checkpoints on the [SAM 3 Hugging Face repo](https://huggingface.co/facebook/sam3) and authenticate:
```bash
huggingface-cli login
```

## Common Development Commands

### Code Formatting
```bash
# Format code with ufmt (uses ruff-api backend)
ufmt format .
```

### Running Tests
```bash
# Run tests (test files in tests/ directory)
pytest

# Run specific test file
pytest tests/test_*.py

# Run with coverage
pytest --cov
```

### Running Examples
```bash
# Start Jupyter notebook
jupyter notebook examples/sam3_image_predictor_example.ipynb

# Available example notebooks:
# - sam3_image_predictor_example.ipynb: Text and visual box prompts on images
# - sam3_video_predictor_example.ipynb: Text prompts on videos with interactive refinement
# - sam3_image_batched_inference.ipynb: Batched inference on images
# - sam3_agent.ipynb: SAM 3 Agent for complex text prompts
# - sam3_for_sam1_task_example.ipynb: SAM 1 task compatibility
# - sam3_for_sam2_video_task_example.ipynb: SAM 2 video task compatibility
```

### Training
```bash
# Local single GPU training
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 1

# Local multi-GPU training
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 0 --num-gpus 4

# Cluster training with SLURM
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml --use-cluster 1 --partition gpu_partition --num-gpus 8 --num-nodes 2

# Training on ODinW13 dataset
python sam3/train/train.py -c configs/odinw13/odinw_text_only_train.yaml
```

### Evaluation
```bash
# Evaluate on Roboflow dataset
python sam3/train/train.py -c configs/roboflow_v100/roboflow_v100_eval.yaml

# Evaluate on ODinW13 dataset
python sam3/train/train.py -c configs/odinw13/odinw_text_only.yaml

# Evaluate on SA-Co benchmarks
python scripts/eval/gold/eval_sam3.py
```

## Architecture Overview

### Core Components

**Model Builder (`sam3/model_builder.py`)**
- Central factory for creating SAM 3 models
- Main entry points:
  - `build_sam3_video_predictor()`: For video segmentation and tracking
  - `build_sam3_image_model()`: For image segmentation
- Handles checkpoint loading from Hugging Face
- Configures multi-GPU distribution

**Video Predictor (`sam3/model/sam3_video_predictor.py`)**
- Session-based API for video segmentation
- Request types: `start_session`, `add_prompt`, `propagate_in_video`, `reset_session`, `close_session`
- Handles both MP4 videos and JPEG frame folders
- Supports text and point prompts

**Image Predictor (`sam3/model/sam3_image.py`, `sam3/model/sam3_image_processor.py`)**
- Processor-based API for image segmentation
- Supports text, point, and box prompts
- Can run batched inference

**Dual Architecture**
- **Detector**: DETR-based model conditioned on text, geometry, and exemplars
- **Tracker**: Inherits SAM 2 transformer encoder-decoder for video segmentation
- **Shared Vision Encoder**: ViT-based backbone (1024 dim, 32 layers) with features like RoPE, global attention

### Key Model Components

**Vision Backbone (`sam3/model/vitdet.py`)**
- ViT with 1024 embedding dim, 32 layers, 16 heads
- Image size: 1008x1008, patch size: 14
- Uses RoPE (Rotary Position Embedding) and global attention blocks

**Text Encoder (`sam3/model/text_encoder_ve.py`, `sam3/model/tokenizer_ve.py`)**
- Handles text prompt encoding
- BPE tokenizer with vocab file at `assets/bpe_simple_vocab_16e6.txt.gz`

**Geometry Encoders (`sam3/model/geometry_encoders.py`)**
- Encodes point, box, and mask prompts
- Converts spatial prompts to sequence embeddings

### Training System (`sam3/train/`)

**Architecture**
- Uses Hydra for configuration management
- Supports both local and SLURM cluster execution
- Multi-node distributed training via `submitit`

**Key Files**
- `train.py`: Main training entry point
- `trainer.py`: Training loop implementation
- `configs/`: Hydra configuration files organized by dataset/task
- `loss/`: Loss functions including focal loss, mask sampling
- `data/`: Dataset loaders and transforms
- `matcher.py`: Hungarian matcher for object detection
- `nms_helper.py`: Non-maximum suppression utilities

**Configuration Structure**
- Configs in `sam3/train/configs/` organized by dataset:
  - `roboflow_v100/`: Roboflow 100-VL dataset configs
  - `odinw13/`: ODinW13 dataset configs
  - `gold_image_evals/`, `silver_image_evals/`, `saco_video_evals/`: SA-Co benchmark configs
- Job arrays supported for sweeping across datasets

## Important Implementation Details

### Video Inference Session Pattern
Video inference uses a session-based API pattern:
1. Start session with `start_session` → returns `session_id`
2. Add prompts with `add_prompt` (text, points, boxes)
3. Propagate through video with `propagate_in_video` (streaming)
4. Always close session with `close_session` to free resources

### Coordinate Systems
- Point prompts can be in absolute pixels or relative [0, 1] coordinates
- Auto-detection: if `max(points) < 1`, treats as relative coordinates
- Internally converts to relative coordinates for the model

### Multi-GPU Distribution
- `gpus_to_use` parameter controls GPU allocation
- Default: uses single GPU (`torch.cuda.current_device()`)
- For multi-GPU: pass list of GPU IDs (e.g., `[0, 1, 2, 3]`)

### Logging
- Logger configured in `sam3/logger.py`
- To disable logging in scripts, set `logging.basicConfig(level=logging.ERROR)`
- Training logs saved to `experiment_log_dir/` with TensorBoard support

### Checkpoints
- Model checkpoints loaded from Hugging Face Hub
- Default model: `facebook/sam3`
- Checkpoint paths configurable via model builder parameters

## Dataset Formats

### SA-Co Benchmarks
Three benchmarks for open-vocabulary segmentation:
- **SA-Co/Gold**: High-quality image benchmark with 270K unique concepts
- **SA-Co/Silver**: Larger image benchmark
- **SA-Co/VEval**: Video benchmark

Available on HuggingFace and Roboflow Universe.

### Training Datasets
- **Roboflow 100-VL**: 100 datasets organized by supercategory in `roboflow_vl_100_root/`
- **ODinW13**: 13 datasets for object detection, organized in `odinw_data_root/`

Each dataset should have `train/`, `valid/`, `test/` subdirectories.

## Special Files

**BPE Vocabulary**
- Located at `assets/bpe_simple_vocab_16e6.txt.gz`
- Required for text tokenization
- Automatically included in package via `MANIFEST.in`

**Custom Inference Wrapper**
- `sam3/sam3_inference.py`: Simplified inference API with caching
- Main function: `sam3_inference(video_path=..., text=..., points=...)`
- Returns binary masks as `torch.Tensor[num_frames, H, W]`
- Supports optional `jhutil.cache_output` decorator for result caching

## Package Structure

```
sam3/
├── model/              # Core model implementations
│   ├── sam3_video_predictor.py     # Video inference API
│   ├── sam3_image.py               # Image model
│   ├── sam3_image_processor.py     # Image inference API
│   ├── sam3_tracking_predictor.py  # Tracking components
│   ├── vitdet.py                   # Vision backbone
│   ├── text_encoder_ve.py          # Text encoder
│   ├── decoder.py                  # Transformer decoder
│   ├── encoder.py                  # Transformer encoder
│   └── geometry_encoders.py        # Prompt encoders
├── train/              # Training system
│   ├── train.py                    # Training entry point
│   ├── trainer.py                  # Training loop
│   ├── configs/                    # Hydra configs
│   ├── data/                       # Dataloaders
│   ├── loss/                       # Loss functions
│   └── transforms/                 # Data augmentations
├── eval/               # Evaluation tools
├── agent/              # SAM 3 Agent for complex prompts
├── sam/                # SAM 1/2 compatibility components
├── model_builder.py    # Model factory
└── sam3_inference.py   # Simplified inference wrapper
```

## Development Notes

### Python Version
- Requires Python 3.12 or higher (per README)
- pyproject.toml shows compatibility back to Python 3.8

### CUDA and Performance
- CUDA 12.6 or higher recommended
- TensorFloat-32 automatically enabled on Ampere GPUs (compute capability ≥ 8.0)
- Model supports bfloat16 precision for faster inference

### NumPy Version Pinning
- Pinned to numpy>=1.26,<2 (see recent commit about numpy 1.26.X)
- Important for compatibility with dependent packages