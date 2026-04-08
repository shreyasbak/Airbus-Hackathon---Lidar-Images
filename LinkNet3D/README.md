
# 🔷 LinkNet3D: A 3D-OD Model

This repository provides the **LinkNet3D** model implementation.

This repo is built from the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) 3D object detection framework(Apache 2.0 License). We thank the team of OpenPCDet for their contribution.

LinkNet3D is designed to be integrated **without changing any core file names** — it can easily be inserted directly inside the BEV backbone module.

---
## 📊 KITTI Benchmark Results

| **Benchmark**              | **Easy** | **Moderate** | **Hard** |
|-----------------------------|----------|--------------|----------|
| Car (3D Detection)          | 87.22 %  | 78.54 %      | 74.36 %  |
| Car (Bird's Eye View)       | 91.71 %  | 88.16 %      | 84.85 %  |
| Cyclist (3D Detection)      | 76.11 %  | 60.47 %      | 53.77 %  |
| Cyclist (Bird's Eye View)   | 79.47 %  | 64.64 %      | 57.99 %  |

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `2DCRM.py` | The LinkNet3D backbone class |
| `LinkNet3D.yaml` | Configuration for training on the KITTI dataset |
| `README.md` | Setup and usage instructions |

---

## ⚙️ Setup Instructions

### ✅ Step 1: Clone OpenPCDet

```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
```
Install dependencies and compile CUDA ops:
---

## 📂 Dataset Preparation

```bash
Download the official **KITTI 3D Object Detection** dataset from https://www.cvlibs.net/datasets/kitti/index.php and organize it as follows
LinkNet3D
├── OpenPCDet
|    ├── data
|    │   ├── kitti
|    │   │   │── ImageSets
|    │   │   │── training
|    │   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
|    │   │   │── testing
|    │   │   │   ├──calib & velodyne & image_2
|    ├── pcdet
|    ├── tools
```

### 🔧 Step 2: Add LinkNet3D Backbone
1. Copy `2DCRM.py` to:

```
OpenPCDet/pcdet/models/backbones_2d/bev_backbone/
```

2. Open this file for editing:

```
OpenPCDet/pcdet/models/backbones_2d/bev_backbone/__init__.py
```

3. Add this line at the top (with other imports):

```python
from .2DCRM import 2DCRM
```

> ✅ This will register the class so that it can be used in your config file.

---
Note: This implementation assumes basic familiarity with OpenPCDet and 3D object detection workflows. LinkNet3D is built on OpenPCDet’s existing codebase.

### 📝 Step 3: Add the Config File

Copy `LinkNet3D.yaml` to:

```
OpenPCDet/tools/cfgs/kitti_models/
```

You can now train using this config.

---

## 🚀 Training

Run training using:

```bash
python train.py --cfg_file cfgs/kitti_models/LinkNet3D.yaml
```

You can add arguments like:

```bash
python train.py --cfg_file cfgs/kitti_models/LinkNet3D.yaml --epochs 80 
```

---

## 🧪 Evaluation

After training is complete, evaluate the checkpoint:

```bash
python test.py --cfg_file cfgs/kitti_models/LinkNet3D.yaml --ckpt <path_to_your_checkpoint.pth>
```

---

## 📊 TensorBoard

To monitor training:

```bash
tensorboard --logdir=output
```

Logged metrics include:
- `loss`
- `learning rate`

You can optionally log `accuracy` by modifying the training loop in `train_utils/train_utils.py`.

---

## 🧠 Acknowledgements

- 🚗 Framework: [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
- 📄 Paper Inspiration: [SECOND: Sparsely Embedded Convolutional Detection](https://arxiv.org/abs/1811.10092)

---

## 📄 License

This project inherits the license of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (Apache 2.0).

---
