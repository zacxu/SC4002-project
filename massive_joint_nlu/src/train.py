"""Training loop: dev-joint checkpoint selection, schedules, OOV simulation, EMA, extra regularisers."""
import copy
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import guard
from .config import DEVICE, CKPT_DIR, RESULTS_DIR, seed_everything
from .data import make_collate
from .evaluate import predict, to_device
from .models import count_params


def make_loaders(assets, train_ds, dev_ds, batch_size, seed, test_ds=None):
    g = torch.Generator()
    g.manual_seed(seed)
    collate = make_collate(assets)
    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, generator=g)
    dl = DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=collate)
    if test_ds is None:
        return tl, dl
    return tl, dl, DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate)


def corrupt_batch(b, tc, unk_id, counts, starts=None):
    """Simulated OOV, training only: lexical tokens become <UNK> or their frozen Word2Vec row."""
    if not (tc.word_dropout or tc.slot_dropout or tc.rare_alpha or tc.swap_prob or tc.mention_dropout):
        return b
    ids = b["word_ids"]
    b["orig_word_ids"] = ids
    live = b["lexical_mask"] & b["attention_mask"]
    in_slot = b["slot_ids"] > 0                     # label 0 is O, padding is -100
    rate = torch.full(ids.shape, float(tc.word_dropout), device=ids.device)
    if tc.slot_dropout > 0:
        rate = rate + tc.slot_dropout * in_slot.float()
    if tc.rare_alpha > 0:
        c = counts[ids.clamp(max=counts.numel() - 1)]
        rate = rate + tc.rare_alpha / (tc.rare_alpha + c)
    chosen = live & (torch.rand(ids.shape, device=ids.device) < rate)
    if tc.mention_dropout > 0:                      # a whole gold span at once
        span_id = (starts[b["slot_ids"].clamp_min(0)] & in_slot).long().cumsum(1) * in_slot.long()
        kmax = int(span_id.max())
        if kmax > 0:
            drop = torch.rand(ids.size(0), kmax + 1, device=ids.device) < tc.mention_dropout
            drop[:, 0] = False
            chosen = chosen | (drop.gather(1, span_id) & live)
    new_ids = ids.masked_fill(chosen, unk_id)
    has_row = b["swap_ids"] >= 0
    if tc.swap_frac > 0:
        to_row = chosen & has_row & (torch.rand(ids.shape, device=ids.device) < tc.swap_frac)
        new_ids = torch.where(to_row, b["swap_ids"], new_ids)
    sim = chosen
    if tc.swap_prob > 0:
        extra = live & ~chosen & has_row & (torch.rand(ids.shape, device=ids.device) < tc.swap_prob)
        new_ids = torch.where(extra, b["swap_ids"], new_ids)
        sim = sim | extra
    b["word_ids"] = new_ids
    b["sim_mask"] = sim
    return b


class EMA:
    """Exponential moving average of the trainable parameters."""

    def __init__(self, model, decay):
        self.decay = decay
        self.names = {n for n, p in model.named_parameters() if p.requires_grad}
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.names:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)


def _sym_kl(p_logits, q_logits):
    logp, logq = F.log_softmax(p_logits, -1), F.log_softmax(q_logits, -1)
    return 0.5 * (F.kl_div(logp, logq, reduction="batchmean", log_target=True)
                  + F.kl_div(logq, logp, reduction="batchmean", log_target=True))


def _forward_loss(model, b, tc):
    il, sl = model(b)
    loss = model.loss(il, sl, b, tc)
    biaffine = model.cfg.slot_head == "biaffine"
    mask = b["attention_mask"]
    if tc.rdrop_alpha > 0:  # R-Drop: a second pass with other dropout masks, pulled together
        il2, sl2 = model(b)
        loss = 0.5 * (loss + model.loss(il2, sl2, b, tc))
        kl = _sym_kl(il, il2)
        if not biaffine:
            kl = kl + _sym_kl(sl[mask], sl2[mask])
        loss = loss + tc.rdrop_alpha * kl
    if tc.cons_alpha > 0 and "orig_word_ids" in b:  # the word-dropped view follows the clean view
        with torch.no_grad():
            il_c, sl_c = model(dict(b, word_ids=b["orig_word_ids"]))
        kl = F.kl_div(F.log_softmax(il, -1), F.softmax(il_c, -1), reduction="batchmean")
        if not biaffine:
            kl = kl + F.kl_div(F.log_softmax(sl[mask], -1), F.softmax(sl_c[mask], -1), reduction="batchmean")
        loss = loss + tc.cons_alpha * kl
    return loss


def _train_step(model, b, tc, opt, sched, ema, params):
    opt.zero_grad(set_to_none=True)
    loss = _forward_loss(model, b, tc)
    loss.backward()

    if tc.adv_eps > 0:  # adversarial embedding noise (Miyato et al. 2017), size relative to the embeddings
        emb, g = model._emb, model._emb.grad
        m = b["attention_mask"].unsqueeze(-1).float()
        g = g.detach() * m
        scale = (emb.detach() * m).flatten(1).norm(dim=1, keepdim=True)
        direction = g.flatten(1) / (g.flatten(1).norm(dim=1, keepdim=True) + 1e-12)
        delta = (tc.adv_eps * scale * direction).reshape(g.shape)
        model._emb = None
        il, sl = model(b, emb_delta=delta)
        model.loss(il, sl, b, tc).backward()

    if tc.sam_rho > 0:  # sharpness-aware minimisation (Foret et al. 2021)
        grads = [p.grad for p in params if p.grad is not None]
        norm = torch.norm(torch.stack([g.norm() for g in grads])) + 1e-12
        saved = []
        with torch.no_grad():
            for p in params:
                if p.grad is None:
                    continue
                e = tc.sam_rho * p.grad / norm
                p.add_(e)
                saved.append((p, e))
        opt.zero_grad(set_to_none=True)
        _forward_loss(model, b, tc).backward()
        with torch.no_grad():
            for p, e in saved:
                p.sub_(e)

    torch.nn.utils.clip_grad_norm_(params, tc.clip)     # the sparse paths are not clipped
    opt.step()
    sched.step()
    if ema is not None:
        ema.update(model)
    return loss.item()


def save_atomic(obj, path):
    """A crash during saving leaves the previous file intact."""
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "r+b") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def train_model(model, assets, train_ds, dev_ds, tc, run_name, meta=None, verbose=True):
    """Train; keep the best dev-joint weights on disk as soon as they appear."""
    assert not (tc.adv_eps > 0 and tc.sam_rho > 0), "use one of adv_eps / sam_rho"
    assert not (tc.adv_eps > 0 and (tc.cons_alpha > 0 or tc.rdrop_alpha > 0)), "adv_eps is not combined with cons_alpha or rdrop_alpha"
    seed_everything(tc.seed)
    guard.check(run_name, need_vram=True)
    if tc.emb_lr_mult == 0:
        model.embedding.weight.requires_grad_(False)
    model = model.to(DEVICE)
    model._track_emb = tc.adv_eps > 0
    model._wide_dropout = tc.wide_dropout
    train_loader, dev_loader = make_loaders(assets, train_ds, dev_ds, tc.batch_size, tc.seed)
    counts = torch.as_tensor(assets.word_counts, device=DEVICE)
    starts = torch.tensor([t[0] in "BS" for t in train_ds.tagset.labels], device=DEVICE)

    def is_sparse(n):
        return n.startswith("wide_") or n.startswith("wslot_")

    emb_params = [p for n, p in model.named_parameters() if n == "embedding.weight" and p.requires_grad]
    sparse = [p for n, p in model.named_parameters() if is_sparse(n) and p.requires_grad]
    other = [p for n, p in model.named_parameters()
             if n != "embedding.weight" and not is_sparse(n) and p.requires_grad]
    params = other + emb_params
    groups = [{"params": other, "lr": tc.lr, "weight_decay": tc.weight_decay}]
    if emb_params:
        groups.append({"params": emb_params, "lr": tc.lr * tc.emb_lr_mult, "weight_decay": tc.weight_decay})
    if sparse:
        groups.append({"params": sparse, "lr": tc.wide_lr, "weight_decay": tc.wide_wd})
    opt = torch.optim.AdamW(groups)

    steps_per_epoch = len(train_loader)
    total_steps = tc.epochs * steps_per_epoch
    warm = int(tc.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if warm and step < warm:
            return (step + 1) / warm
        if tc.schedule == "cosine":
            prog = (step - warm) / max(1, total_steps - warm)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = EMA(model, tc.ema_decay) if tc.ema_decay > 0 else None

    ckpt_path = CKPT_DIR / f"{run_name}.pt"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    total, trainable = count_params(model)
    best = {"score": -1.0, "epoch": 0, "state": None, "dev": None}

    def write_checkpoint(history, elapsed):
        save_atomic({
            "run_name": run_name,
            "model_state_dict": best["state"],
            "model_config": model.cfg.to_dict(),
            "train_config": tc.to_dict(),
            "best_epoch": best["epoch"],
            "epochs_run": len(history) - 1,
            "dev": best["dev"],
            "history": history,
            "elapsed_s": elapsed,
            "params_total": total,
            "params_trainable": trainable,
            "vocab": {"word_vocab": assets.word_vocab, "slot_labels": assets.slot_labels,
                      "intent_labels": assets.intent_labels},
            "meta": meta or {},
        }, ckpt_path)

    e0 = predict(model, dev_loader, assets)
    history = [{"epoch": 0, "train_loss": np.nan, "dev_loss": e0["loss"], "dev_intent_acc": e0["intent_accuracy"],
                "dev_slot_f1": e0["slot_micro_f1"], "dev_joint": e0["joint_score"], "elapsed_s": 0.0}]
    if verbose:
        print(f"[{run_name}] epoch 00 | dev {e0['loss']:.4f} | intent {e0['intent_accuracy']:.2f} | "
              f"slot {e0['slot_micro_f1']:.2f} | joint {e0['joint_score']:.2f}", flush=True)

    patience = tc.patience
    t0 = time.time()
    train_ds.mention_replace, train_ds.lwtr, train_ds.sis = tc.mention_replace, tc.lwtr, tc.sis
    try:
        for epoch in range(1, tc.epochs + 1):
            model.train()
            losses = []
            for raw in train_loader:
                b = corrupt_batch(to_device(raw), tc, assets.UNK_ID, counts, starts)
                losses.append(_train_step(model, b, tc, opt, sched, ema, params))

            if ema is not None:
                live = copy.deepcopy(model.state_dict())
                model.load_state_dict(ema.shadow)
            r = predict(model, dev_loader, assets)
            row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_loss": r["loss"],
                   "dev_intent_acc": r["intent_accuracy"], "dev_slot_f1": r["slot_micro_f1"],
                   "dev_joint": r["joint_score"], "elapsed_s": time.time() - t0}
            history.append(row)
            if verbose:
                print(f"[{run_name}] epoch {epoch:02d} | train {row['train_loss']:.4f} | dev {row['dev_loss']:.4f} | "
                      f"intent {row['dev_intent_acc']:.2f} | slot {row['dev_slot_f1']:.2f} | "
                      f"joint {row['dev_joint']:.2f} | {row['elapsed_s']:.0f}s", flush=True)
            if row["dev_joint"] > best["score"]:
                best.update(score=row["dev_joint"], epoch=epoch,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            dev={k: r[k] for k in ("intent_accuracy", "slot_micro_f1", "joint_score",
                                                   "exact_match", "loss")})
                patience = tc.patience
                write_checkpoint(history, time.time() - t0)
            else:
                patience -= 1
            if ema is not None:
                model.load_state_dict(live)
            if patience <= 0:
                if verbose:
                    print(f"[{run_name}] early stop", flush=True)
                break
            t0 += guard.check(run_name)     # cooling pauses do not count as training time
    finally:
        train_ds.mention_replace = train_ds.lwtr = train_ds.sis = 0.0
    elapsed = time.time() - t0

    model.load_state_dict(best["state"])
    model._track_emb = False
    write_checkpoint(history, elapsed)
    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / "history.csv", index=False)
    plot_history(hist, run_dir / "curve.png", run_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if verbose:
        print(f"[{run_name}] best epoch {best['epoch']} | dev intent {best['dev']['intent_accuracy']:.2f} | "
              f"slot {best['dev']['slot_micro_f1']:.2f} | joint {best['dev']['joint_score']:.2f} | "
              f"params {total:,} (trainable {trainable:,}) | {elapsed:.0f}s | peak RAM {guard.peak_ram_mb()} MB | "
              f"ckpt {ckpt_path}", flush=True)
    return model, ckpt


def plot_history(hist, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = hist[hist["epoch"] > 0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["epoch"], h["train_loss"], marker="o", label="train loss")
    ax[0].plot(h["epoch"], h["dev_loss"], marker="o", label="dev loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("joint loss"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(h["epoch"], h["dev_intent_acc"], marker="o", label="dev intent acc")
    ax[1].plot(h["epoch"], h["dev_slot_f1"], marker="o", label="dev slot F1")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("%"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
