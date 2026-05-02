# Introduction
Implementation of the paper [Structured Semantic Modeling and Cross-Modal Interaction for Robust Image-Text Matching]. The contributing journals are The Visual Computer.
The full source code will be released publicly upon acceptance of the paper.
# Prerequisites
## Environment
### Environment
- Python 3.10.4
- PyTorch 1.13.0
- CUDA 11.7.0
### Installation
```bash
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
## Pretrained checkpoints
All pretrained checkpoints will be uploaded [here](#).
- `f30k_gru_scratch.pth`: on Flickr30K, GRU, trained from scratch.
---

## Training

All config files can be found in the `config` folder. To run training, find the config file corresponding to the settings you want to run, make adjustments for your environment, hyperparameters, etc., and launch training with:
```bash
# Flickr30K, GRU as text encoder, train with 4 GPUs
python -m main.train --cfg config/f30/gru.yaml \
    DISTRIBUTED.world_size 4 TRAIN.num_workers 12
