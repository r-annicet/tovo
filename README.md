# TOVO: Tone-oriented vision optimization for efficient low-light image enhancement

> 🚀 Official PyTorch Implementation

This repository contains the **official implementation** of:

> **TOVO: Tone-oriented vision optimization for efficient low-light image enhancement**  
> *Signal Processing (Elsevier), 2026*  
> 🔗 https://doi.org/10.1016/j.sigpro.2026.110540

## Overview

TOVO is a **training-free low-light image enhancement** method. It improves visibility and brightness in low-illumination images **without requiring any pretrained weights or training data**.

Unlike learning-based approaches, TOVO can be applied **directly at inference time** to enhance images, making it lightweight, fast, and easy to deploy.

### Key Features

- **No training required** — works out of the box  
- **Lightweight & fast** — suitable for real-time use  
- **Simple pipeline** — minimal setup and dependencies  
- **Stable results** — enhances brightness while preserving natural colors  

### Where It Is Useful

TOVO can be used in scenarios such as:
- Low-light photography enhancement  
- Surveillance and security imaging  
- Preprocessing for computer vision pipelines  
- Mobile and embedded vision applications  
## Visual Results

### Multi-Scene Enhancement Results

![TOVO Results](assets/Figure_6.png)

*Qualitative comparison on multiple indoor and outdoor scenes. TOVO produces consistent brightness enhancement while preserving natural colors and avoiding artifacts.*

---

### Results on LOL Dataset (with Ground Truth)

![LOL Results](assets/Figure_5.png)

*Comparison on LOL dataset with ground truth. TOVO achieves visually balanced enhancement close to the reference.*

---

### Detail Comparison

![Detail Comparison](assets/Figure_4.png)

*Zoomed-in comparison highlighting detail preservation and artifact reduction.*
## Installation

### 1. Create Virtual Environment (Recommended)

```bash
python -m venv tovo
```

Activate the environment:

- Linux / Mac:
```bash
source tovo/bin/activate
```

- Windows:
```bash
tovo\Scripts\activate
```

---

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install torch torchvision kornia pillow
```

---

### 3. CUDA Support (Optional)

For faster inference, you can use a GPU with CUDA.

Install PyTorch with CUDA support from the official website:  
https://pytorch.org/

Run with GPU:
```bash
--device cuda
```

Run on CPU:
```bash
--device cpu
```
## Quick Start (TL;DR)

Enhance a single image with default settings:

```bash
python inference.py --mode single --input input.jpg --output output.jpg --iteration 5 --device cuda --color_space hsv
```

- Replace `input.jpg` with your image path  
- The enhanced result will be saved as `output.jpg`  
- Use `--device cpu` if CUDA is not available  
## Usage

### 1. Single Image

Enhance a single image:

```bash
python inference.py \
  --mode single \
  --input path/to/input.jpg \
  --output path/to/output.jpg \
  --iteration 5 \
  --device cuda \
  --color_space hsv
```

---

### 2. Folder

Enhance all images in a folder:

```bash
python inference.py \
  --mode folder \
  --input path/to/input_folder \
  --output path/to/output_folder \
  --iteration 5 \
  --device cuda \
  --color_space hsv
```

---

### 3. Dataset Mode

Dataset mode enhances all images inside the selected dataset directory.

```bash
python inference.py \
  --mode data \
  --dataset VV \
  --model TOVO \
  --iteration 5 \
  --device cuda \
  --color_space hsv
```

#### Dataset layout (recommended)

You can keep all datasets under a single root folder, for example:

```bash
datasets/
├── VV/
├── DICM/
├── NPE/
├── MEF/
├── LIME/
├── LOL/
│   └── low/
├── LOLv2real/
│   └── low/
└── LOLv2synthetic/
    └── low/
```

**Notes**
- For most datasets (e.g., `VV`, `DICM`, `NPE`, `MEF`, `LIME`), images can be directly inside the dataset folder.
- For `LOL`, `LOLv2real`, and `LOLv2synthetic`, the script expects images under `low/` (because it uses `<dataset>/low` internally).

#### Output

Results are saved under:

```bash
<model>/<dataset>/fake/
```

Example:

```bash
TOVO/VV/
TOVO/LOL/fake/
```

---

### 4. Batch Experiments (run_batch_experiments.py)

Use this script to automatically run multiple experiments across:
- multiple datasets
- multiple iteration values
- both `hsv` and `rgb` enhancement modes

```bash
python run_batch_experiments.py
```

#### Requirement (only for batch script)

`run_batch_experiments.py` calls `inference.py --mode data --dataset <name>`.  
So you must ensure the dataset folders listed inside the script exist and are accessible from where you run it.

A common setup is to run it from the parent directory that contains all datasets, e.g.:

```bash
datasets/
  VV/
  DICM/
  NPE/
  ...
run_batch_experiments.py
inference.py
model.py
```

Inside `run_batch_experiments.py`, datasets/iterations/color_spaces are configured like:

```python
datasets = ['VV', 'DICM', 'NPE', 'MEF', 'LOL', 'LOLv2real', 'LOLv2synthetic', 'LIME']
iterations = [1, 2, 3, 4, 5, 6]
color_spaces = ['hsv', 'rgb']
```
## Arguments / Configuration

| Argument         | Description | Default | Recommendation |
|------------------|------------|---------|----------------|
| `--mode`       | Execution mode: `single`, `folder`, or `data` | `data` | Use `single` for quick testing |
| `--input`        | Path to input image or folder | None | Required for `single` and `folder` |
| `--output`       | Path to save results | None | Required for `single` and `folder` |
| `--dataset`      | Dataset name (used in `data` mode) | `VV` | Match folder name exactly |
| `--model`        | Output directory name | `TOVO` | Use custom names for experiments |
| `--iteration`    | Number of enhancement iterations | `5` | `5` gives best balance |
| `--device`       | `cuda` or `cpu` | `cuda` | Use `cuda` if available |
| `--color_space`  | `hsv` or `rgb` | `hsv` | Use `hsv` for better results |

---

## Repository Structure

```bash
.
├── model.py                    # Core TOVO implementation
├── inference.py                # Main script for image enhancement
├── run_batch_experiments.py    # Batch processing across datasets/settings
└── README.md
```

---

## Notes / Implementation Details

- TOVO is **training-free** — no pretrained weights are required  
- The model runs entirely at inference time  
- Input images are **normalized automatically** before enhancement  
- Default processing uses the **HSV color space** (recommended)  
- RGB mode is available but may produce less stable color results  
- Increasing `--iteration` increases brightness but may lead to over-enhancement  
- Works on both **CPU and GPU**  

### Tips

- Use `--iteration 5` for best visual balance  
- Use `--color_space hsv` for natural colors  
- For experiments, use different `--model` names to organize outputs  

---

## Citation

If you use this work, please cite:

```bibtex
@article{RAZAFINDRATOVOLAHY2026110540,
title = {TOVO: Tone-oriented vision optimization for efficient low-light image enhancement},
journal = {Signal Processing},
volume = {244},
pages = {110540},
year = {2026},
issn = {0165-1684},
doi = {https://doi.org/10.1016/j.sigpro.2026.110540},
url = {https://www.sciencedirect.com/science/article/pii/S016516842600054X},
author = {Annicet Razafindratovolahy and Yunbo Rao and Jean Clément Tovolahy and Mona Afanga and Albert Mutale},
keywords = {Low-Light image enhancement, Image processing, Monotonic transformation, Non-Linear image enhancement, Real-Time image processing}
```

---

## License

This project is licensed under the MIT License.

