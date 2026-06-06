# HSF-Net

Official implementation of **HSF-Net: Hierarchical Structure-Semantics Fusion Network for RGB-DSM Remote Sensing Semantic Segmentation**.

## ⚙️ Installation & Dependencies

```bash
git clone 
cd HSF-Net

conda env create -f requirements.yml
conda activate HSF-Net
```

## 🔗 Pretrained Backbone

Please download the SegFormer MiT-B4 pretrained backbone and place it under:

```text
pretrained/
└── mit_b4.pth
└── open_clip_pytorch_model.bin
```

Download links:

| Backbone | Link |
|---|---|
| MiT-B4 | [Google Drive](https://github.com/qubvel/segmentation_models.pytorch/releases/download/v0.0.2/mit_b4.pth) |
| CLIP ViT-B-32 | [Google Drive](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) |

Then set the path in `config.yaml` and `moedls/prompt_semantic.py`:

```yaml
model:
  pretrained_backbone: "/path/to/pretrained/mit_b4.pth"
```
```py
clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained="/path/to/pretrained/open_clip_pytorch_model.bin",
        )
```

## 🛰️ Datasets

Experiments are conducted on three RGB-DSM semantic segmentation datasets:

| Dataset | Type | Download |
|---|---|---|
| ISPRS Vaihingen | Public benchmark | [Download Dataset](https://www.isprs.org/education/benchmarks/UrbanSemLab/) |
| ISPRS Potsdam | Public benchmark | [Download Dataset](https://www.isprs.org/education/benchmarks/UrbanSemLab/) |
| Ordos-OPM | Open-pit mining RGB-DSM dataset | Coming Soon |

Please organize the datasets as follows:

```text
datasets/
├── Vaihingen/
├── Potsdam/
└── Ordos/
```

Then modify the dataset root path in `configs/config.yaml`:

```yaml
folder: "/path/to/datasets"
```

## 🚀 Training

### Train on ISPRS Vaihingen

```bash
bash train_Vaihingen.sh 0
```

or:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py training_dataset=Vaihingen
```

### Train on ISPRS Potsdam

```bash
bash train_Potsdam.sh 0
```

or:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py training_dataset=Potsdam
```

### Train on Ordos-OPM

```bash
bash train_Ordos.sh 0
```

or:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py training_dataset=Ordos
```



## 📦 Checkpoints

| Dataset | Backbone | Checkpoint |
|---|---|---|
| ISPRS Vaihingen | MiT-B4 | Coming Soon |
| ISPRS Potsdam | MiT-B4 | Coming Soon |
| Ordos-OPM | MiT-B4 | Coming Soon |



