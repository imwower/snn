#!/usr/bin/env python3
from __future__ import annotations

import argparse, os, numpy as np
from typing import Tuple

from snn.trainers.train_numpy import (
    ReadoutMLP, FPTBlock, ThreeCompTStep,
    load_dataset, compute_or_load_norm, normalize,
)


def build_models_from_npz(path: str):
    data = np.load(path)
    mode = "fpt" if int(data.get("mode", np.array([1]))[0]) == 1 else "tstep"
    # Infer dims from head
    W1 = data["head_W1"]; b1 = data["head_b1"]
    W2 = data["head_W2"]; b2 = data["head_b2"]
    in_dim = W1.shape[0]; hidden = W1.shape[1]; out_dim = W2.shape[1]
    # dummy rng not used for evaluation
    rng = np.random.default_rng(123)
    head = ReadoutMLP(in_dim, hidden, out_dim, rng, layer_norm=True, dropout_prob=0.0, learn_logit_scale=True)
    head.W1[...] = W1; head.b1[...] = b1
    head.W2[...] = W2; head.b2[...] = b2
    if "ln_gamma" in data and hasattr(head, "ln_gamma") and head.ln_gamma is not None:
        head.ln_gamma[...] = data["ln_gamma"]
    if "ln_beta" in data and hasattr(head, "ln_beta") and head.ln_beta is not None:
        head.ln_beta[...] = data["ln_beta"]
    if "logit_scale" in data:
        head.logit_scale[...] = data["logit_scale"]

    if mode == "fpt":
        core = FPTBlock(in_dim, hidden, seed=123)
        core.Wxh[...] = data["Wxh"]; core.Whh[...] = data["Whh"]; core.bh[...] = data["bh"]
    else:
        core = ThreeCompTStep(in_dim, hidden, {}, seed=123)
        core.Wxb[...] = data["Wxb"]; core.Wxa[...] = data["Wxa"]
        core.Whb[...] = data["Whb"]; core.Wha[...] = data["Wha"]
        for k in ("g","tau_m","tau_a","tau_b","v_th","beta","T"):
            if k in data:
                setattr(core, k, float(data[k][0]) if k != "T" else int(data[k][0]))
    return mode, core, head


def eval_checkpoint(npz_path: str, dataset_root: str, dataset_name: str = "MNIST", batch: int = 256) -> Tuple[float, float]:
    Xtr, ytr, Xte, yte, in_dim, num_classes = load_dataset(dataset_name, dataset_root)
    mean, std = compute_or_load_norm(dataset_name, dataset_root, Xtr)
    Xte = normalize(Xte, mean, std)

    mode, core, head = build_models_from_npz(npz_path)
    total = Xte.shape[0]
    seen = 0; correct = 0; top5_sum = 0.0
    for i in range(0, total, batch):
        xb = Xte[i:i+batch]
        yb = yte[i:i+batch]
        if mode == "tstep":
            _, Z = core.forward_states(xb)
        else:
            h0 = np.zeros((xb.shape[0], head.in_dim), np.float32)
            Z, *_ = FPTBlock.anderson_solve(
                xb, h0, core.Wxh, core.Whh, core.bh,
                K=10, tol=1e-5, m=10, beta=0.5,
                solver="anderson", line_search=True, ridge=1e-4,
                fp_err_guard=1e-1, damping=0.9,
            )
        logits, _ = head.forward(Z, training=False)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= np.clip(np.sum(probs, axis=1, keepdims=True), 1e-9, None)
        pred = np.argmax(probs, axis=1)
        correct += int((pred == yb).sum()); seen += yb.shape[0]
        idx = np.argpartition(-probs, min(4, probs.shape[1]-1), axis=1)[:, :5]
        top5_sum += float(np.mean((idx == yb[:, None]).any(axis=1))) * yb.shape[0]
    acc = correct / max(1, seen)
    top5 = top5_sum / max(1, seen)
    return acc, top5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/np_trainer_best.npz")
    ap.add_argument("--data_root", default=".data/datasets")
    ap.add_argument("--dataset", default="MNIST")
    args = ap.parse_args()
    acc, top5 = eval_checkpoint(args.ckpt, args.data_root, args.dataset)
    print(f"EVAL: acc={acc:.4f} top5={top5:.4f}")


if __name__ == "__main__":
    main()

