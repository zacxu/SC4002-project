"""Plot every result together: the KP ladder, the notebook's recorded run, the reruns, and all test scores.

Writes plots/all_dev_results.png, plots/reruns_sam_embeddings.png, plots/systems_dev_test.png and the table
behind them, plots/all_results.csv. Reads saved results only.

Usage: python -I plot_all_results.py
"""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(sys.argv[0]).resolve().parents[1]
PROJECT = ROOT.parent
OUT = ROOT / "plots"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
KP, NOTEBOOK, RERUN = "#2a78d6", "#eb6834", "#1baf7a"
FINAL_DEV_JOINT = 85.49     # KP strict single, the ladder's final system
# zacxu/SC4002-project results/revised_transformer/results.csv and the notebook's analysis table; dev only, RTX 3060 Ti
RECORDED = [("Notebook Stage 1, recorded", 84.16, 69.56, 34.13), ("Notebook Stage 2, recorded", 85.83, 77.76, 43.70)]
RUNG_NAMES = {"s": "Stage 1 baseline", "A": "A  Recipe and Viterbi", "B": "B  Word features", "C": "C  Slot heads", "D": "D  Encoders",
              "R": "R  Regularisers", "H": "H  Random search", "F": "F  80 epochs", "E": "E  Post-hoc decoding"}
LABELS = {"r0_word2vec_base": "r0 Word2Vec", "r1_word2vec_sam": "r1 Word2Vec + SAM", "r2_fasttext_base": "r2 fastText", "r3_glove_base": "r3 GloVe",
          "r4_fasttext_sam": "r4 fastText + SAM", "r5_glove_sam": "r5 GloVe + SAM"}

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": AXIS, "xtick.labelcolor": INK2,
                     "ytick.left": False, "text.color": INK})


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load():
    """One record per plotted result. Ladder families average their seeds, as the KP summary does."""
    records, families = [], {}
    for row in read_csv(PROJECT / "results" / "leaderboard.csv"):
        if not row["family"].startswith("main_"):      # notebook retrains are not ladder runs
            families.setdefault(row["family"], []).append(row)
    if len(families) != 103 or sum(map(len, families.values())) != 149:
        raise ValueError("Ladder size differs from the KP summary (103 families, 149 runs)")
    for family, runs in families.items():
        mean = lambda key: round(sum(float(run[key]) for run in runs) / len(runs), 2)
        records.append({"source": "KP ladder", "group": family[0], "name": family, "split": "dev", "seeds": len(runs), "needs_approval": runs[0]["flagged"] == "True",
                        "intent": mean("dev_intent"), "slot_f1": mean("dev_slot_f1"), "joint": mean("dev_joint"), "slot_oov_f1": mean("f1_slot_oov")})
    for row in read_csv(PROJECT / "results" / "posthoc_leaderboard.csv"):
        records.append({"source": "KP ladder", "group": "E", "name": row["system"], "split": "dev", "seeds": 1, "needs_approval": row["flagged"] == "True",
                        "intent": float(row["dev_intent"]), "slot_f1": float(row["dev_slot_f1"]), "joint": float(row["dev_joint"]), "slot_oov_f1": float(row["f1_slot_oov"])})
    names = {"Stage1": "KP Stage 1", "strict_single": "KP strict single", "flagged_single": "KP flagged single"}
    for row in read_csv(PROJECT / "results" / "final" / "final_comparison.csv"):
        intent, slot = float(row["intent_accuracy"]), float(row["slot_micro_f1"])
        records.append({"source": "KP ladder", "group": "final", "name": names[row["system"]], "split": row["split"], "seeds": 1, "needs_approval": row["system"] == "flagged_single",
                        "intent": intent, "slot_f1": slot, "joint": round((intent + slot) / 2, 2), "slot_oov_f1": float(row["slot_f1_slot_oov"])})
    for name, intent, slot, oov in RECORDED:
        records.append({"source": "Notebook, recorded", "group": "recorded", "name": name, "split": "dev", "seeds": 1, "needs_approval": False,
                        "intent": intent, "slot_f1": slot, "joint": round((intent + slot) / 2, 2), "slot_oov_f1": oov})
    for row in read_csv(ROOT / "results_summary.csv"):
        records.append({"source": "Notebook, reruns", "group": row["stage"], "name": LABELS[row["run"]], "split": "dev", "seeds": 1, "needs_approval": row["embedding"] != "word2vec",
                        "intent": float(row["intent"]), "slot_f1": float(row["slot_f1"]), "joint": float(row["joint"]), "slot_oov_f1": float(row["slot_oov_slot_f1"])})
    for folder in ("test_r0_stage2", "test_r1_stage2"):
        report = json.loads((ROOT / folder / "test_report.json").read_text(encoding="utf-8"))
        records.append({"source": "Notebook, reruns", "group": "stage2", "name": LABELS[report["run"]], "split": "test", "seeds": 1, "needs_approval": False,
                        "intent": report["intent"], "slot_f1": report["slot_f1"], "joint": report["joint"], "slot_oov_f1": report["groups"]["slot_oov"]["slot_f1"]})
    return records


def dot(ax, x, y, color, hollow=False, marker="o", size=7.5):
    ax.plot([x], [y], marker=marker, ms=size, ls="none", mfc=SURFACE if hollow else color, mec=color if hollow else SURFACE,
            mew=1.4 if hollow else 0.8, zorder=3, clip_on=False)


def legend_dot(label, color, hollow=False, marker="o"):
    return Line2D([], [], marker=marker, ms=7.5, ls="none", mfc=SURFACE if hollow else color, mec=color if hollow else SURFACE, mew=1.4 if hollow else 0.8, label=label)


def heading(fig, title, subtitle, top=0.975, gap=0.03):
    fig.text(0.02, top, title, fontsize=15, fontweight="semibold", va="top")
    fig.text(0.02, top - gap, subtitle, fontsize=10.5, color=INK2, va="top")


def clean(ax):
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def swarm(values, gap=0.22, step=0.24):
    """Vertical offsets that stop dots in a row overlapping, nearest the row's centre first."""
    placed, offsets = [], {}
    for index in sorted(range(len(values)), key=lambda i: values[i]):
        for level in (0, 1, -1, 2, -2, 3, -3, 4, -4):
            if all(abs(values[index] - x) >= gap for x, other in placed if other == level):
                break
        placed.append((values[index], level))
        offsets[index] = level * step
    return [offsets[i] for i in range(len(values))]


def overview(records):
    dev = [r for r in records if r["split"] == "dev"]
    layout = [("KP ladder", [(RUNG_NAMES[key], [r for r in dev if r["source"] == "KP ladder" and r["group"] == key], KP) for key in "sABCDRHFE"]),
              ("Notebook, recorded run", [("Stage 1 and Stage 2", [r for r in dev if r["group"] == "recorded"], NOTEBOOK)]),
              ("Notebook, reruns", [("Stage 1", [r for r in dev if r["group"] == "stage1"], RERUN), ("Stage 2", [r for r in dev if r["group"] == "stage2"], RERUN)])]
    fig, ax = plt.subplots(figsize=(12.5, 11.8))
    fig.subplots_adjust(left=0.17, right=0.77, top=0.89, bottom=0.1)
    y, ticks, labels = 0.0, [], []
    for group, rows in layout:
        ax.text(-0.235, y, group, transform=ax.get_yaxis_transform(), fontweight="semibold", va="center", clip_on=False)
        y += 0.75
        for label, items, color in rows:
            offsets = swarm([item["joint"] for item in items])
            half = max(0.3, max(map(abs, offsets)))     # dense rows take more height, so rows never run together
            y += half
            ticks.append(y)
            labels.append(label)
            for item, offset in zip(items, offsets):
                dot(ax, item["joint"], y + offset, color, hollow=item["needs_approval"])
            allowed = max((item for item in items if not item["needs_approval"]), key=lambda item: item["joint"])
            note = f"{allowed['name'].removeprefix('Notebook ').removesuffix(', recorded')}  {allowed['joint']:.2f}"
            if allowed["name"] == "strict_ensemble":
                note += ", not adopted"
            ax.text(1.015, y, note, transform=ax.get_yaxis_transform(), color=INK2, va="center", clip_on=False)
            y += half + 0.45
        y += 0.2
    ax.set_yticks(ticks, labels, color=INK2)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.45, -1.0)
    ax.set_xlim(73.5, 87)
    ax.set_xticks(range(74, 88, 2))
    ax.set_xlabel("Dev joint score", color=INK2)
    clean(ax)
    ax.axvline(FINAL_DEV_JOINT, color=INK2, lw=0.9, zorder=1)
    ax.text(FINAL_DEV_JOINT - 0.1, -0.55, f"KP strict single, final system: {FINAL_DEV_JOINT:.2f}", color=INK2, ha="right", va="center")
    ax.text(1.015, -0.55, "Best within the rules", transform=ax.get_yaxis_transform(), fontweight="semibold", va="center", clip_on=False)
    heading(fig, "The KP ladder stays ahead of the revised notebook on dev, even with SAM and new vectors",
            "Dev joint score: the mean of intent accuracy and slot F1. Each dot is one experiment family or one run.")
    fig.legend(handles=[legend_dot("KP ladder", KP), legend_dot("Notebook, recorded run", NOTEBOOK), legend_dot("Notebook, reruns", RERUN),
                        legend_dot("Within the assignment rules", MUTED), legend_dot("Needs lecturer approval", MUTED, hollow=True)],
               loc="upper left", bbox_to_anchor=(0.012, 0.925), ncol=5, frameon=False, handletextpad=0.2, columnspacing=1.6)
    fig.text(0.02, 0.022, "Ladder dots average the 1 or 3 seeds run for that family. The notebook's runs use one seed and 20 epochs; the ladder ran 12 to 80 epochs.\n"
             "The ladder scores exact token spans and the notebook exact character spans, so small gaps between the two sources aren't exact. Selected dev scores are optimistic.",
             color=MUTED, fontsize=9, va="bottom")
    fig.savefig(OUT / "all_dev_results.png", dpi=200)
    plt.close(fig)


def sam_effect(records):
    runs = {(r["group"], r["name"]): r for r in records if r["source"] == "Notebook, reruns" and r["split"] == "dev"}
    tests = {r["name"]: r["joint"] for r in records if r["source"] == "Notebook, reruns" and r["split"] == "test"}
    recorded = {"stage1": RECORDED[0], "stage2": RECORDED[1]}
    pairs = [("Word2Vec", "r0 Word2Vec", "r1 Word2Vec + SAM", False), ("fastText", "r2 fastText", "r4 fastText + SAM", True), ("GloVe", "r3 GloVe", "r5 GloVe + SAM", True)]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.875, top=0.72, bottom=0.17, wspace=0.42)
    for ax, stage, title, limits in zip(axes, ("stage1", "stage2"), ("Stage 1", "Stage 2"), ((75, 79), (80, 84))):
        _, intent, slot, _ = recorded[stage]
        dot(ax, (intent + slot) / 2, 0, NOTEBOOK)
        ax.text(1.02, 0, f"Recorded  {(intent + slot) / 2:.2f}", transform=ax.get_yaxis_transform(), color=INK2, va="center", clip_on=False)
        for y, (_, plain, sam, hollow) in enumerate(pairs, 1):
            before, after = runs[stage, plain]["joint"], runs[stage, sam]["joint"]
            ax.plot([before, after], [y, y], color=AXIS, lw=2, solid_capstyle="round", zorder=2)
            dot(ax, before, y, MUTED, hollow=hollow)
            dot(ax, after, y, RERUN, hollow=hollow)
            change = after - before
            tie = ", tie" if abs(change) < 0.8 else ""
            ax.text(1.02, y, f"{before:.2f} → {after:.2f}\nSAM {change:+.2f}{tie}".replace("-", "−"), transform=ax.get_yaxis_transform(), color=INK2, va="center", clip_on=False, linespacing=1.25)
        if stage == "stage2":
            before, after = tests["r0 Word2Vec"], tests["r1 Word2Vec + SAM"]
            ax.plot([before, after], [4, 4], color=AXIS, lw=2, solid_capstyle="round", zorder=2)
            dot(ax, before, 4, MUTED, marker="D", size=6.8)
            dot(ax, after, 4, RERUN, marker="D", size=6.8)
            ax.text(1.02, 4, f"{before:.2f} → {after:.2f}\nSAM {after - before:+.2f}", transform=ax.get_yaxis_transform(), color=INK2, va="center", clip_on=False, linespacing=1.25)
        else:
            ax.text(1.02, 4, "Not tested", transform=ax.get_yaxis_transform(), color=INK2, va="center", clip_on=False)
        ax.set_xlim(*limits)
        ax.set_xticks(range(limits[0], limits[1] + 1))
        ax.set_title(title, loc="left", fontweight="semibold", fontsize=11)
        ax.set_xlabel("Joint score", color=INK2)
        clean(ax)
    axes[0].set_yticks(range(5), ["Word2Vec,\nrecorded run", "Word2Vec", "fastText", "GloVe", "Word2Vec,\ntest split"], color=INK2)
    axes[0].set_ylim(4.5, -0.5)
    axes[0].tick_params(axis="y", length=0)
    heading(fig, "SAM lowers Stage 1 on every embedding and lifts Stage 2 on two of the three",
            "Reruns of the revised notebook, one seed. Dev scores except the test row. Each axis covers 4 points. Under 0.8 points counts as a tie.", gap=0.065)
    fig.legend(handles=[legend_dot("No SAM", MUTED), legend_dot("SAM, rho 0.2", RERUN), legend_dot("Notebook's recorded run, no SAM", NOTEBOOK),
                        legend_dot("Test", MUTED, marker="D"), legend_dot("Hollow: needs lecturer approval", MUTED, hollow=True)],
               loc="upper left", bbox_to_anchor=(0.012, 0.87), ncol=5, frameon=False, handletextpad=0.2, columnspacing=1.6)
    fig.savefig(OUT / "reruns_sam_embeddings.png", dpi=200)
    plt.close(fig)


def systems(records):
    rows = [("KP Stage 1", KP), ("KP strict single", KP), ("KP flagged single", KP), ("Notebook Stage 1, recorded", NOTEBOOK), ("Notebook Stage 2, recorded", NOTEBOOK)]
    rows += [(LABELS[key], RERUN) for key in LABELS]
    stage2 = [r for r in records if r["source"] == "Notebook, reruns" and r["group"] == "stage2"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 6.6), sharey=True)
    fig.subplots_adjust(left=0.175, right=0.985, top=0.79, bottom=0.135, wspace=0.09)
    panels = (("intent", "Intent accuracy", (80, 90), 2), ("slot_f1", "Slot F1", (55, 85), 5), ("slot_oov_f1", "Slot F1, utterances with an unseen slot word", (20, 80), 10))
    for ax, (key, title, limits, step) in zip(axes, panels):
        for y, (name, color) in enumerate(rows):
            dev = next((r for r in (stage2 if color == RERUN else records) if r["name"] == name and r["split"] == "dev"), None)
            test = next((r for r in (stage2 if color == RERUN else records) if r["name"] == name and r["split"] == "test"), None)
            if test:
                ax.plot([dev[key], test[key]], [y, y], color=AXIS, lw=2, solid_capstyle="round", zorder=2)
                dot(ax, test[key], y, color, hollow=test["needs_approval"], marker="D", size=6.8)
                above = test[key] > dev[key]    # keep the label clear of the dev dot
                ax.annotate(f"{test[key]:.2f}", (test[key], y), xytext=(8 if above else -8, 0), textcoords="offset points", ha="left" if above else "right", va="center", color=INK2, fontsize=9)
            dot(ax, dev[key], y, color, hollow=dev["needs_approval"])
        ax.set_xlim(*limits)
        ax.set_xticks(range(limits[0], limits[1] + 1, step))
        ax.set_title(title, loc="left", fontweight="semibold", fontsize=11)
        clean(ax)
        ax.set_xlabel("Percent", color=INK2)
    axes[0].set_yticks(range(len(rows)), [name for name, _ in rows], color=INK2)
    axes[0].set_ylim(len(rows) - 0.5, -0.5)
    axes[0].tick_params(axis="y", length=0)
    heading(fig, "On test, the KP strict single leads the notebook with SAM, mainly on slots",
            "Five systems have test scores, labelled on the chart. The notebook's recorded run and r2 to r5 are dev only. Reruns shown are Stage 2.", gap=0.045)
    fig.legend(handles=[legend_dot("KP ladder", KP), legend_dot("Notebook, recorded run", NOTEBOOK), legend_dot("Notebook, reruns", RERUN), legend_dot("Dev", MUTED),
                        legend_dot("Test", MUTED, marker="D"), legend_dot("Hollow: needs lecturer approval", MUTED, hollow=True)],
               loc="upper left", bbox_to_anchor=(0.012, 0.895), ncol=6, frameon=False, handletextpad=0.2, columnspacing=1.6)
    fig.text(0.02, 0.018, "The ladder scores exact token spans and the notebook exact character spans; the unseen-word groups have the same sizes in both (dev 215, test 341). One seed per system.",
             color=MUTED, fontsize=9, va="bottom")
    fig.savefig(OUT / "systems_dev_test.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    records = load()
    overview(records)
    sam_effect(records)
    systems(records)
    with (OUT / "all_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(len(records), "results written to", OUT)


if __name__ == "__main__":
    main()
