# Efficient ResNet for CIFAR-10 (≤ 5M trainable parameters)

**Deep Learning Project 1, Fall 2026** · Author: Hakan Gulec · Kaggle team: Hakan Gulec

A 24-layer ResNet with **4,697,162 trainable parameters**, trained from scratch on the 50,000 CIFAR-10 training images
provided for the course's Kaggle competition. It reaches **96.60%** on 5,000 held-out training images. Our two final
Kaggle submissions (Cutout and CutMix variants) score 0.889 and 0.888 on the public leaderboard and 0.864 and 0.866 on
the private leaderboard; the team's final private score is **0.866**. 

**Refer to the Results section for the accuracies (the ones with final submission) and Model section for the total number of parameters.**

## Results

All four runs use the same architecture, seed and training code. Each notebook below has been executed and
keeps its outputs, so you can see the code run, the training log, the plots, the parameter count and the accuracy.

| Notebook | Training images | Regulariser | Validation accuracy (single view / mirror-averaged) | Kaggle public | Kaggle private |
| --- | --- | --- | --- | --- | --- |
| `resnet_cifar10_training_main_run.ipynb` | 45,000 | Cutout | 96.46% / 96.60% | 0.880 | – |
| `resnet_cifar10_training_full_run.ipynb` (**final submission**) | 50,000 | Cutout | no validation split | **0.889** | 0.864 |
| `resnet_cifar10_cutmix.ipynb` | 45,000 | CutMix | 96.58% / 96.60% | not submitted | – |
| `resnet_cifar10_cutmix_full.ipynb` (**final submission**) | 50,000 | CutMix | no validation split | 0.888 | **0.866** |

- **Validation accuracy** is measured on 5,000 training images (500 per class) that were held out and never trained on.
- **Test accuracy** is the Kaggle leaderboard score; the test labels are hidden. Both full-data models were selected as
  final submissions; the private leaderboard ranks the team by the better of the two (0.866). Private scores are listed
  for those two submissions.
- Kaggle scores were typed into each notebook's last cell by hand after submitting. The full-run notebook's last cell
  prints 88.00% because it still held the main run's score when that run started; its own public score is 0.889
  (a note above that cell says so).

## Model

| Part | Output (H×W×C) | Content | Parameters |
| --- | --- | --- | --- |
| Stem | 32×32×64 | 3×3 conv, BN, ReLU | 1,856 |
| Stage 1 | 32×32×64 | 4 residual blocks | 295,936 |
| Stage 2 | 16×16×128 | 4 residual blocks (first: stride 2, 1×1 projection shortcut) | 1,116,416 |
| Stage 3 | 8×8×256 | 3 residual blocks (first: stride 2, 1×1 projection shortcut) | 3,280,384 |
| Head | 10 | 8×8 average pool, linear layer | 2,570 |
| **Total** | | | **4,697,162** |

Each block computes ReLU(S(x) + F(x)) with F = conv3×3 → BN → ReLU → conv3×3 → BN. Training: SGD with Nesterov
momentum 0.9, weight decay 5e-4, batch 128, 200 epochs, 5-epoch warm-up to lr 0.1 then cosine decay, label smoothing
0.1, random crop + horizontal flip + TrivialAugment + Cutout (or CutMix). Predictions average each test image with its
horizontal mirror. Details, design alternatives and lessons learned are in the report.

## Repository layout

```
README.md
requirements.txt
resnet_cifar10_training_main_run.ipynb    main run: 45,000 training images + 5,000 validation
resnet_cifar10_training_full_run.ipynb    final model: all 50,000 training images
resnet_cifar10_cutmix.ipynb               CutMix experiment with validation
resnet_cifar10_cutmix_full.ipynb          CutMix experiment on all 50,000 images
outputs/
    <run>_best.pt                         trained weights of each run (selected by validation accuracy,
                                          or the final epoch when there is no validation split)
    <run>_history.json                    per-epoch loss, accuracy, learning rate and time, plus the run's settings
    <run>_log.txt                         the printed training log
    <run>_submission.csv                  the Kaggle predictions of each run (id,label)
    figures/                              every plot shown in the notebooks and the report
```

The competition data (`data.zip`) is not included, because it belongs to the course.

## How to reproduce a run

1. Use Python 3.9 or newer. The notebook's first cell installs any missing package from `requirements.txt`
   (PyTorch, torchvision, NumPy, pandas, Matplotlib, Pillow).
2. Put the course's `data.zip` in this folder. The notebook unzips it into `data/` on the first run.
3. Open a notebook and run all cells. All settings are in the configuration cell of Section 1; each notebook already
   holds the settings of its run. Set `SMOKE_TEST = True` first for a 2-minute end-to-end check on 2,000 images.
4. The notebook uses an NVIDIA GPU (CUDA) or an Apple-silicon GPU (MPS) automatically. One 200-epoch run took about
   4 hours on an Apple-silicon MacBook Pro. Re-running a notebook overwrites that run's files in `outputs/`.
   GPU non-determinism can move results by a few tenths of a point.

To load trained weights without retraining, run the notebook's data and model cells (Sections 1–4), then:

```python
ckpt = torch.load('outputs/full_best.pt', map_location='cpu', weights_only=False)
model = ResNet(tuple(ckpt['config']['channels']), tuple(ckpt['config']['blocks']))
model.load_state_dict(ckpt['model'])
```

## References

- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- K. Liu. *pytorch-cifar*, https://github.com/kuangliu/pytorch-cifar. Our basic-block code follows this repository.
- T. DeVries, G. W. Taylor. *Improved Regularization of Convolutional Neural Networks with Cutout.* arXiv:1708.04552, 2017.
- S. Yun et al. *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features.* ICCV 2019.
- S. G. Müller, F. Hutter. *TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation.* ICCV 2021.
- I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017.
- P. Goyal et al. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.* arXiv:1706.02677, 2017.
- C. Szegedy et al. *Rethinking the Inception Architecture for Computer Vision* (label smoothing). CVPR 2016.
- B. Recht, R. Roelofs, L. Schmidt, V. Shankar. *Do CIFAR-10 Classifiers Generalize to CIFAR-10?* arXiv:1806.00451, 2018.
- A. Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.
