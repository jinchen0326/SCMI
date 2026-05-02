import argparse
import glob
import json
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.utils.clip_grad import clip_grad_norm_

from config import cfg

from datasetsmodel.image_caption import BaseDataset, ImageCaptionDataset
from datasetsmodel.image_caption import collate_fn as my_collate_fn

from utils.average_meter import AverageMeter
from utils.count_parameters import count_parameters
from utils.evaluation import i2t, t2i
from utils.record_hyperparams import record_hyperparams

import models.cora_model as cora_model


@torch.no_grad()
def validate(val_loader, model, model2, device, cfg, split="testall"):
    model.eval()
    model2.eval()

    saved_images = 0
    attn1_list = []
    cross_attn_list = []
    img_paths_list = []

    for i, batch in enumerate(val_loader):
        if saved_images >= 5:
            break

        # batch 取出
        images, captions, img_paths = batch
        images = images.to(device)
        captions = captions.to(device)
        bs = images.size(0)

        # 前向计算
        _ = model(images, captions)

        # 取注意力矩阵
        attn1 = None
        cross_attn = None

        if getattr(model.image_encoder, "attn1", None) is not None:
            attn1 = model.image_encoder.attn1.detach().cpu().clone().numpy()  # (B, H, N, N) 或 (H, N, N)

        if getattr(model.image_encoder, "cross_attn", None) is not None:
            cross_attn = model.image_encoder.cross_attn.detach().cpu().clone().numpy()  # (B, H, N, N) 或 (H, N, N)

        # 遍历 batch 内样本
        for j in range(bs):
            if saved_images >= 5:
                break

            # 如果 attn1 有 batch 维度
            if attn1 is not None:
                if attn1.ndim == 4:   # (B, H, N, N)
                    attn1_list.append(attn1[j])
                elif attn1.ndim == 3: # (H, N, N) → 没有 batch 维度，只能整份保存
                    attn1_list.append(attn1)

            if cross_attn is not None:
                if cross_attn is not None and cross_attn.dim() in (3, 4):
                    cross_attn_list.append(cross_attn[j])
                elif cross_attn.ndim == 3:
                    cross_attn_list.append(cross_attn)

            img_paths_list.append(img_paths[j])
            saved_images += 1

    # 保存文件
    os.makedirs("saved_attn", exist_ok=True)

    np.save("saved_attn/attn1.npy", np.array(attn1_list, dtype=object))
    np.save("saved_attn/cross_attn.npy", np.array(cross_attn_list, dtype=object))
    with open("saved_attn/img_paths.json", "w") as f:
        json.dump(img_paths_list, f, indent=2)

    print(f"已保存 {len(img_paths_list)} 张图片的注意力矩阵")


    # === 评估部分保持不变 ===
    if split == 'testall':
        img_embeds = torch.cat(img_embeds, dim=0)
        cap_embeds = torch.cat(cap_embeds, dim=0)

        results = []
        for i in range(5):
            img_embs_shard = img_embeds[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embeds[i * 5000:(i + 1) * 5000]
            start = time.time()
            sims = torch.matmul(img_embs_shard, cap_embs_shard.t())
            end = time.time()
            print("calculate similarity time: {}".format(end - start))
            print(sims.shape, flush=True)
            sims = sims.numpy()
            npts = sims.shape[0]
            (r1, r5, r10, medr, meanr) = i2t(npts, sims)
            (r1i, r5i, r10i, medri, meanri) = t2i(npts, sims)

            ar = (r1 + r5 + r10) / 3
            ari = (r1i + r5i + r10i) / 3
            rsum = r1 + r5 + r10 + r1i + r5i + r10i
            results += [[r1, r5, r10, medr, meanr] +
                        [r1i, r5i, r10i, medri, meanri] +
                        [ar, ari, rsum]]
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print(f'rsum: {mean_metrics[12]:.1f}')
        print(f'Avg i2t: {mean_metrics[10]:.1f}')
        print(f'Image to text: {mean_metrics[0]:.1f} {mean_metrics[1]:.1f} {mean_metrics[2]:.1f} {mean_metrics[3]:.1f} {mean_metrics[4]:.1f}')
        print(f'Avg t2i: {mean_metrics[11]:.1f}')
        print(f'Text to image: {mean_metrics[5]:.1f} {mean_metrics[6]:.1f} {mean_metrics[7]:.1f} {mean_metrics[8]:.1f} {mean_metrics[9]:.1f}')
        return mean_metrics[12]
    else:
        img_embeds = torch.cat(img_embeds, dim=0)
        img_embeds = torch.cat([img_embeds[i].unsqueeze(0) for i in range(0, len(img_embeds), 5)])
        cap_embeds = torch.cat(cap_embeds, dim=0)

        img_embeds2 = torch.cat(img_embeds2, dim=0)
        img_embeds2 = torch.cat([img_embeds2[i].unsqueeze(0) for i in range(0, len(img_embeds2), 5)])
        cap_embeds2 = torch.cat(cap_embeds2, dim=0)

        start = time.time()
        sims = torch.matmul(img_embeds, cap_embeds.t())
        sims2 = torch.matmul(img_embeds2, cap_embeds2.t())
        end = time.time()
        if device == 'cuda:0':
            print("calculate similarity time: {}".format(end - start))
        sims = (sims + sims2) / 2

        sims = sims.numpy()

        # caption retrieval
        npts = sims.shape[0]
        (r1, r5, r10, medr, meanr) = i2t(npts, sims)
        if device == 'cuda:0':
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                  (r1, r5, r10, medr, meanr), flush=True)

        # image retrieval
        (r1i, r5i, r10i, medri, meanri) = t2i(npts, sims)
        if device == 'cuda:0':
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                  (r1i, r5i, r10i, medri, meanri), flush=True)

        currscore = r1 + r5 + r10 + r1i + r5i + r10i
        if device == 'cuda:0':
            print('Current rsum is {}'.format(currscore), flush=True)

        return currscore



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True,
                        help='path to config file')
    parser.add_argument('--split', type=str, required=True,
                        help="'dev' or 'test'")
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='modify config file from terminal')
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    print(cfg)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    torch.cuda.set_device(0)
    device = 'cuda:0'

    # Prepare dataset & dataloader.
    print('Prepare dataset', flush=True)
    base_dataset = BaseDataset(cfg, split=args.split)
    dataset = ImageCaptionDataset(base_dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

    if cfg.MODEL.word_embed_source not in ['bert', 'bert-prefix']:
        vocab_size = len(base_dataset.vocab.word2idx)
    else:
        vocab_size = -1

    print('Prepare model', flush=True)
    model = cora_model.build_cora_model(cfg, vocab_size=vocab_size)
    model.setup_training_loss(cfg)
    model.to(device)
    model2 = cora_model.build_cora_model(cfg, vocab_size=vocab_size)
    model2.setup_training_loss(cfg)
    model2.to(device)

    if cfg.MODEL.weights != '':
        print('Load weights from checkpoint at %s' % cfg.MODEL.weights)
        weights_dict = torch.load(cfg.MODEL.weights, map_location=device)
        weights_dict2 = torch.load(cfg.MODEL.weights2, map_location=device)
        model_state_dict = model.state_dict()
        unloaded_dict = {
            k: v for k, v in weights_dict.items()
            if (k not in model_state_dict) or (v.shape != model_state_dict[k].shape)}
        weights_dict = {
            k: v for k, v in weights_dict.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape}
        missing_dict = {
            k: v for k, v in model_state_dict.items()
            if (k not in weights_dict) or (v.shape != weights_dict[k].shape)}
        if device == 'cuda:0':
            print('-----------------------------')
            print('Unable to load the following weights')
            print(unloaded_dict.keys())
            print('Missing the following weights')
            print(missing_dict.keys())
        model.load_state_dict(weights_dict, strict=True)
        model2.load_state_dict(weights_dict2, strict=True)

    torch.backends.cudnn.benchmark = True

    print('Start evaluating', flush=True)

    score = validate(dataloader, model, model2, device, cfg, split=args.split)


if __name__ == "__main__":
    main()
