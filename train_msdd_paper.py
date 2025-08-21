# 文件：train_msdd_paper.py
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diarize.msdd_modules import MSDDDecoder
from datasets.msdd_paper_dataset import MSDDPaperDataset
from diarize.utils import Dict2ObjParser

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", type=str, default="conf/config.yaml")
    p.add_argument("--log_csv", type=str, default="train_log.csv")
    p.add_argument("--tb_dir", type=str, default="runs/msdd")
    return p.parse_args()

def bce_f1(prob, Y, thr=0.5):
    """返回 BCE loss 和 micro-F1"""
    loss = nn.BCELoss()(prob, Y.float())
    with torch.no_grad():
        pred = (prob > thr).int()
        tp = ((pred == 1) & (Y == 1)).sum().item()
        fp = ((pred == 1) & (Y == 0)).sum().item()
        fn = ((pred == 0) & (Y == 1)).sum().item()
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return loss, f1

def main():
    args = parse_args()

    # === 读取配置 ===
    with open(args.cfg_file, "r") as f:
        cfg = Dict2ObjParser(yaml.safe_load(f)).parse()
    device = cfg.misc.device
    K      = len(cfg.embedding.win_length_list)
    Ne     = int(getattr(cfg.msdd, "emb_dim", 192))
    bs     = int(getattr(cfg.msdd, "batch_size", 1))
    lr     = float(getattr(cfg.msdd, "lr", 3e-4))
    epochs = int(getattr(cfg.msdd, "max_epochs", 10))
    ckpt   = getattr(cfg.msdd, "ckpt", "checkpoints/msdd.pt")
    infer_thr = float(getattr(cfg.msdd, "infer_thr", 0.5))

    # === 数据集 / DataLoader ===
    ds = MSDDPaperDataset(cfg)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0,
                    collate_fn=MSDDPaperDataset.collate)

    # === 模型 / 优化器 ===
    model = MSDDDecoder(
        K=K, Ne=Ne,
        lstm_hidden=cfg.msdd.lstm_hidden,
        lstm_layers=cfg.msdd.lstm_layers
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # === 可视化：TensorBoard & CSV ===
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    writer = SummaryWriter(args.tb_dir)
    # 准备 CSV 头
    if not os.path.isfile(args.log_csv):
        with open(args.log_csv, "w") as f:
            f.write("epoch,loss,f1\n")

    best_f1 = 0.0

    for ep in range(epochs):
        model.train()
        tot_loss, tot_f1, n_batches = 0.0, 0.0, 0

        # tqdm 进度条（总步数=可用 batch 数；无法预知时直接不设置总数也可）
        pbar = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}", dynamic_ncols=True)

        for batch in pbar:
            if batch is None:
                continue
            U_seq, V1, V2, Y = batch  # [B,T,K,Ne], [B,K,Ne], [B,K,Ne], [B,T,2]
            U_seq, V1, V2, Y = U_seq.to(device), V1.to(device), V2.to(device), Y.to(device)

            prob = model(U_seq, V1, V2)              # [B,T,2]
            loss, f1 = bce_f1(prob, Y, thr=infer_thr)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # 累计
            tot_loss += loss.item()
            tot_f1 += f1
            n_batches += 1

            # tqdm 上显示即时指标
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "f1": f"{f1:.4f}"})

        if n_batches == 0:
            print("本轮无有效样本（可能是 RTTM 缺失或簇数为1）。请检查数据或聚类阈值。")
            continue

        avg_loss = tot_loss / n_batches
        avg_f1   = tot_f1 / n_batches

        # 控制台输出
        print(f"[Epoch {ep}] loss={avg_loss:.4f}  F1={avg_f1:.4f}")

        # TensorBoard
        writer.add_scalar("loss/train", avg_loss, ep)
        writer.add_scalar("f1/train", avg_f1, ep)

        # CSV
        with open(args.log_csv, "a") as f:
            f.write(f"{ep},{avg_loss:.6f},{avg_f1:.6f}\n")

        # 保存最好
        if avg_f1 > best_f1:
            best_f1 = avg_f1
            torch.save(model.state_dict(), ckpt)
            print(f"Saved best to {ckpt} (F1={best_f1:.4f})")

    writer.close()
    print("训练完成。")

if __name__ == "__main__":
    main()
