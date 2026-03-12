## PancreasSubRegion-Segmentation

PancreasSubRegion-Segmentation is a small extension on top of **nnU-Net v2** that implements:

- A custom trainer, `nnUNetSubregionTrainer`, which trains a dedicated **subregion UNet** architecture.
- A **clean, Python-only inference entrypoint** for a single NIfTI file (`simple_inference.py`), without any shell wrappers.
- A **fixed export pipeline** to ensure correct NIfTI orientation and spacing during inference.

It is intended for pancreas subregion segmentation experiments, but can be adapted to other 3D medical image segmentation tasks that follow the nnU-Net v2 conventions.

---

### Pretrained weights

Pretrained weights (`checkpoint_best.pth`) for this project are available on Google Drive:

- [`PancreasPartsSegmentation` weights folder](https://drive.google.com/drive/folders/1-zfIPTpWc44a1RzEla7pSO52B3pvPJbg?usp=sharing)

Download `checkpoint_best.pth` from this folder and place it into your chosen `model_folder` (see below).

---

### Project structure

- `simple_inference.py`  
  Clean, self-contained nnU-Net v2 inference script for a **single NIfTI input**.  
  Uses `nnunetv2.inference.nnUNetPredictor` internally, but is **hard-wired** to:
  - Use `nnUNetSubregionTrainer` as the trainer definition.
  - Build the network architecture via `nnUNetSubregionTrainer.build_network_architecture(...)`.
  - Load a model folder that contains `checkpoint_best.pth`, `dataset.json`, and `plans.json`.

- `export_prediction_fixed.py`  
  Provides `export_prediction_from_logits(...)`, a **fixed** version of nnU-Net’s export utility.  
  It:
  - Calls `convert_predicted_logits_to_segmentation_with_correct_shape(...)`.
  - Preserves and uses orientation metadata from `properties_dict` to avoid flipped / mis-oriented outputs.
  - Returns the final segmentation array to be saved by `simple_inference.py`.

- `nnunetv2/`  
  A local copy / fork of nnU-Net v2 for training and inference (utilities, trainers, architectures, etc.).

- `nnUNetTrainer/`  
  Local trainer extension package:
  - `nnUNetTrainer/nnUNetSubregionTrainer.py`  
    - Subclasses `nnunetv2.training.nnUNetTrainer.nnUNetTrainer`.
    - Overrides `get_tr_and_val_datasets` to **filter training/validation cases** by dataset identifiers:
      - Keeps only identifiers containing any of:  
        `['mca', 'nu', 'nyu', 'cad', 'iu', 'ahn', 'emc', 'mcf', 'northwestern']`.
    - Overrides `build_network_architecture(...)` as a `@staticmethod` that ignores the configuration’s network class and directly calls:
      - `nnunetv2.network_architecture.UNet_subregion.get_network()`
    - Sets `self.num_epochs = 100` and logs that it is using the fixed subregion trainer.
  - `nnUNetTrainer/__init__.py` exposes `nnUNetSubregionTrainer`.

- `figs/`  
  Folder containing qualitative example figures of the pancreas subregion segmentation results (e.g., overlays of predictions on input images). These are meant for visualization and for use in presentations or manuscripts.

- `environment.yml`  
  Conda environment specification with:
  - Python 3.13
  - nnUNet 1.x, MONAI, PyTorch 2.8 + CUDA 12.6 stack
  - Common scientific imaging stack: `nibabel`, `SimpleITK`, `pydicom`, `dicom2nifti`, `scikit-image`, `scikit-learn`, etc.

- `__init__.py`  
  Marks the repository as a Python package and documents it as an “Extension package for nnUNet inference. Contains custom trainer and network architecture.”

---

### Installation

1. **Clone** this repository:

```bash
git clone <your-pancreas-subregion-repo-url>
cd PancreasSubRegion-Segmentation
```

2. **Create and activate** the conda environment:

```bash
conda env create -f environment.yml
conda activate base  # or rename/change as you prefer
```

3. **Install in editable mode** (recommended):

```bash
pip install -e .
```

Make sure that your CUDA drivers match the versions in `environment.yml` if you want GPU inference.

---

### Model folder layout

`simple_inference.py` expects a **model folder** with the standard nnU-Net v2 layout:

```text
model_folder/
  checkpoint_best.pth
  dataset.json
  plans.json
```

These files should be produced by training with nnU-Net v2 using a configuration that is compatible with the subregion architecture (e.g. where `UNet_subregion` was used as the network). Alternatively, you can use the pretrained `checkpoint_best.pth` from the Google Drive link above and pair it with the appropriate `dataset.json` and `plans.json`.

---

### Command-line inference

To run inference on a **single NIfTI file**:

```bash
python simple_inference.py \
  -i /path/to/input_image.nii.gz \
  -o /path/to/output_segmentation.nii.gz \
  -m /path/to/model_folder \
  -d cuda \
  -v
```

**Arguments:**

- `-i, --input`  
  Path to the input NIfTI volume (`.nii.gz`).

- `-o, --output`  
  Desired output segmentation path.  
  Internally, the script strips the extension, appends the dataset-specific `file_ending` (from `dataset.json`, defaulting to `.nii.gz`), and writes using the correct orientation and spacing.

- `-m, --model`  
  Path to the **model folder** containing `checkpoint_best.pth`, `dataset.json`, `plans.json`. Default: `Model`.

- `-d, --device`  
  Device: `cpu`, `cuda`, or `mps`.  
  When `cuda` is used, most operations are kept on the GPU for speed.

- `-v, --verbose`  
  If set, prints detailed logs:
  - Image shape and spacing.
  - Whether `nibabel_stuff` / `sitk_stuff` is present in properties.
  - Shapes at each preprocessing / prediction step.
  - Orientation checks before and after export.

---

### Programmatic inference

You can also call the inference API from Python:

```python
from simple_inference import run_inference

run_inference(
    input_file="/path/to/input_image.nii.gz",
    model_folder="/path/to/model_folder",
    output_file="/path/to/output_segmentation.nii.gz",
    device="cuda",   # or "cpu", "mps"
    verbose=True,
)
```

Or use the `CleanInference` class directly:

```python
from simple_inference import CleanInference
import torch

engine = CleanInference(
    model_folder="/path/to/model_folder",
    device=torch.device("cuda"),
    verbose=True,
)

engine.predict_single_file(
    input_file="/path/to/input_image.nii.gz",
    output_file="/path/to/output_segmentation.nii.gz",
)
```

---

### Training with nnUNetSubregionTrainer

Training is done via nnU-Net v2 but with this custom trainer:

- The trainer:
  - Filters training and validation datasets to a curated subset (mainly multi-center pancreas-related cohorts).
  - Enforces the use of `UNet_subregion.get_network()` for the network architecture, instead of relying on `plans.json`’s `network_arch_class_name`.
  - Runs for `num_epochs = 100`.

Typical high-level steps (simplified):

1. **Prepare data** in nnU-Net v2 format (datasets, preprocessed data, plans, etc.).
2. **Configure** nnU-Net v2 to use `nnUNetTrainer.nnUNetSubregionTrainer` as trainer class.
3. **Run training** using nnU-Net v2’s training entrypoints, ensuring that:
   - This repo is on `PYTHONPATH` so that `nnUNetTrainer.nnUNetSubregionTrainer` can be imported.
   - The `UNet_subregion` architecture is available in `nnunetv2.network_architecture`.

(Exact training commands depend on your local nnU-Net v2 setup and are not included here.)

---

### Orientation and export fixes

`simple_inference.py` and `export_prediction_fixed.py` jointly address several common pitfalls in nnU-Net inference:

- **Preserving orientation metadata**  
  When reading images, the predictor’s `image_reader_writer_class` is used (typically `NibabelIOWithReorient`). The associated `properties` dict includes `nibabel_stuff` or `sitk_stuff`, which is propagated through preprocessing and export.

- **Converting logits to final segmentation**  
  `export_prediction_from_logits` calls  
  `convert_predicted_logits_to_segmentation_with_correct_shape(...)`  
  to obtain a segmentation in nnU-Net’s internal orientation.

- **Re-orienting to original space**  
  `simple_inference.py`:
  - Applies a final transpose to map from nnU-Net internal axes to NIfTI `(x, y, z)`.
  - Uses orientation transforms based on `io_orientation` and `ornt_transform` to reorient the segmentation back to the original acquisition space before saving.
  - Restores voxel spacing from the original properties.

This prevents common issues like left-right flips, swapped axes, or mis-scaled segmentations.

---

### License and citation

This repository builds on **nnU-Net v2** and related work.  
Please also cite the original nnU-Net and any related architecture papers (e.g. subregion UNet / Primus, if used) when using this code in academic work.

If you use the **subregion trainer** or **inference pipeline** in a publication, please cite your own project and the nnU-Net / related papers accordingly.

---

### Contact

This code is primarily intended for internal research use.  
Adapt paths, trainer configuration, and dataset filtering logic as needed for your own setup.

