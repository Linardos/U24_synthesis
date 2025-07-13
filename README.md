# U24 Breast‑Cancer Synthesis & Classification

A reproducible research framework for **training, evaluating, and analysing diffusion‑based synthetic mammography data** and its impact on breast‑cancer classification.

The project offers:

* end‑to‑end **data handling** for public (DDSM, InBreast, CMMD, etc.) and private datasets;
* scalable **training pipelines** in vanilla PyTorch *and* PyTorch Lightning;
* utilities to **blend real & synthetic samples** with precise class‑balance control;
* ready‑made **evaluation & visualisation** scripts for AUC, balanced accuracy, sensitivity/specificity, and FID;
* a **tmux launcher** that runs five random seeds in parallel.

---

## Directory layout

```
linardos‑u24_synthesis/
├── config.yaml                # simple classifier config (PyTorch)
├── data_loaders.py            # core Dataset + real/synthetic mixer
├── holdout_train.py           # single hold‑out split training script
├── main.py                    # k‑fold CV trainer (default)
├── models.py                  # backbone factory (ResNet, DenseNet, …)
├── requirements.txt           # Python deps
├── tmux_initiate_fiveseeds.sh # convenience launcher
├── trim_synthetic.py          # down‑sample synthetic folders per class
├── read_metrics.py            # quick CSV/PKL reader
│
├── basic_trials/              # minimal UNet + GAN baselines (PL)
├── data_handling/             # dataset‑specific preprocessing helpers
├── lightning_synthesis/       # full diffusion/GAN training workflows (PL)
├── plots_for_paper/           # reusable plotting scripts
└── utils/                     # misc helpers
```

---

## Quick‑start (conda)

1. **Create & activate** a fresh environment (Python 3.9 recommended)

   ```bash
   conda create -n U24 python=3.12 -y
   conda activate U24 
   ```

2. **Install PyTorch** with CUDA *or* CPU only (choose the line that matches your system)

   ```bash
   # CUDA 12.1 (find the one for your version though
   pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121

   ```

3. **Install project requirements**

   ```bash
   pip install -r requirements.txt
   ```

   > 💡  `xformers` wheels are pre‑built for common CUDA versions. If installation fails, remove it from
   > `requirements.txt` and skip flash‑attention features.

4. **(Optional) Jupyter support**

   ```bash
   pip install jupyterlab
   ```

---

## Data preparation

All training scripts expect a directory hierarchy such as

```
ROOT/
 ├── original/               # real NIfTI files organised by class
 │     ├── benign/
 │     └── malignant/
 └── synthetic_guide8.0/     # CFG‑DDPM outputs (same sub‑folder layout)
```

Pre‑processing helpers for each public dataset live under `data_handling/`.  Edit
`config.yaml → root_dir` to point at your *train* split.

---

## Training recipes

### 1. Five‑fold cross‑validation (default)

```bash
python main.py         # uses seed from config.yaml
python main.py 123     # override random seed
```

Key options are centralised in **`config.yaml`**:

| Field             | Purpose                                                   |
| ----------------- | --------------------------------------------------------- |
| `real_percentage` | Fraction of real samples in the mixed training set (0–1). |
| `num_classes`     | 2 (binary) · 3 · 4‑class setups supported.                |
| `model_name`      | `resnet50`, `densenet121`, `efficientnet_b0`, …           |
| `k_folds`         | 0 → hold‑out; ≥2 → k‑fold CV.                             |

### 2. Hold‑out split

```bash
python holdout_train.py
```

### 3. Launch 5 seeds in tmux

```bash
bash tmux_initiate_fiveseeds.sh 01  # creates session U24_session_01
```

---

## License & citation

Forthcoming
---

