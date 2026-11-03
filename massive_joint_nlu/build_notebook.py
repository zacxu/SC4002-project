"""Assemble a self-contained main.ipynb from src/ (single source of truth) + the ladder results.

Which systems the notebook trains is read from results/final_systems.json and results/leaderboard.csv,
both produced with dev-only selection. The notebook retrains Stage 1 and every member of those systems
with the same seeds, saves their checkpoints, reloads them and evaluates dev and test.
"""
import json
import pprint
import re
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
LB = ROOT / "results" / "leaderboard.csv"
SYSTEMS = ROOT / "results" / "final_systems.json"


def module_code(name):
    text = (SRC / f"{name}.py").read_text(encoding="utf-8")
    text = re.sub(r"^\s*from \.\w* import .*$\n", "", text, flags=re.M)
    text = text.replace("ROOT = Path(__file__).resolve().parent.parent", "ROOT = Path.cwd()")
    return text.strip() + "\n"


def split_sections(code):
    parts = re.split(r"^# -{20,}.*$\n", code, flags=re.M)
    return [p.strip() + "\n" for p in parts if p.strip()]


def md(s):
    return nbformat.v4.new_markdown_cell(s.strip())


def code(s):
    return nbformat.v4.new_code_cell(s.strip())


def spec(lb, run):
    r = lb[lb["run"] == run].iloc[0]
    return {"run": run, "family": r["family"], "note": r["note"], "rung": r["rung"], "init": r["init"],
            "k_extra": int(r["k_extra"]) if pd.notna(r["k_extra"]) and r["k_extra"] > 0 else None,
            "frozen_dim": int(r["frozen_dim"]) if "frozen_dim" in r and pd.notna(r["frozen_dim"]) else 300,
            "model_config": json.loads(r["model_config"]), "train_config": json.loads(r["train_config"]),
            "ladder_dev_intent": float(r["dev_intent"]), "ladder_dev_slot_f1": float(r["dev_slot_f1"])}


def build():
    lb = pd.read_csv(LB)
    systems = json.loads(SYSTEMS.read_text())
    members = list(dict.fromkeys(m for s in systems.values() for m in s["members"]))
    specs = {m: spec(lb, m) for m in members}
    s1 = lb[lb["run"] == "stage1_baseline"].iloc[0]

    rows = "\n".join(
        f"| `{k}` | {len(v['members'])} model(s) | {'needs lecturer OK' if v['flagged'] else 'plain reading of the rules'} |"
        for k, v in systems.items())
    cells = [md(f"""
# MASSIVE en-US: Joint Intent Classification and Slot Filling

**SC4002 Group Assignment, `main.ipynb`** · Group ID: _fill in_ · Members: _fill in_

One shared model predicts the intent (60 classes) and the BIO slot labels (55 slot types, 108 observed
BIO labels) of each utterance. Fixed word embedding for both stages: **NLPL Wikipedia Word2Vec 2017,
model 6** (300-d, 302,866 vectors). No pretrained language model is used; every system stays under
**20,000,000 parameters**, embedding table included. Train is used for training only, dev for all
selection, and test is evaluated once, after all selection, from saved checkpoints (see Final evaluation).

Stage 1 is the baseline. Stage 2 systems trained below, selected on dev from the full method ladder:

| System | Size | Status |
|---|---|---|
{rows}

> **Integrity note.** The assignment requires the Stage-1 baseline to be developed without generative-AI
> assistance. All code below, Stage 1 included, was produced with AI assistance (Claude). Replace the
> Stage-1 sections with the group's independently written baseline before submitting, and record the AI
> use in `ai_collaboration.md`.

> **Data note.** This notebook reads the verified CSV `data/massive_enUS_full.csv`. The assignment expects
> the official `1.0/data/en-US.jsonl`; swap the loader in section 2 before submitting.
"""),
             md("## 0. Environment\nRequires `torch`, `numpy`, `pandas`, `scikit-learn`, `sentencepiece`, `psutil`, "
                "`matplotlib`. The dataset is read from `data/`, and the Word2Vec archive is downloaded to "
                "`data/nlpl_model_6/` if missing. Neither is part of the submission."),
             code("""
import platform, torch, numpy, pandas, sklearn, sentencepiece, types
print("python", platform.python_version(), "| torch", torch.__version__, "| CUDA", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
"""),
             md("## 1. Configuration, reproducibility and machine safety"),
             code(module_code("config")),
             code(module_code("guard") + "\nguard = types.SimpleNamespace(check=check, peak_ram_mb=peak_ram_mb, ResourceAbort=ResourceAbort)"),
             md("## 2. Load and verify the official MASSIVE en-US split\n"
                "Tokenisation, bracket annotation to BIO conversion, train-only vocabularies, BPE (train only), "
                "character and n-gram tables."),
             code(module_code("data")),
             md("## 3. Fixed pretrained embedding: Wikipedia Word2Vec 2017 (NLPL model 6)"),
             code("""
import urllib.request, zipfile
W2V_URL = "https://vectors.nlpl.eu/repository/20/6.zip"
W2V_DIR.mkdir(parents=True, exist_ok=True)
if not any(W2V_DIR.glob("model.txt")) and not (CACHE_DIR / "w2v_casefold.npy").exists():
    zip_path = W2V_DIR / "6.zip"
    if not zip_path.exists():
        print("downloading", W2V_URL)
        urllib.request.urlretrieve(W2V_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(W2V_DIR)
"""),
             code(module_code("embeddings")),
             md("## 4. Shared evaluation functions\nExact-span slot micro-F1: a true positive needs the exact slot "
                "type and the exact token span. Intent accuracy over all utterances."),
             code(module_code("evaluate")),
             md("## 5. Model definitions\n`JointModel(ModelConfig())` is the Stage-1 model. Stage-2 switches add "
                "modules that start at zero, so a Stage-1-initialised Stage-2 model starts exactly at Stage 1.")]
    cells += [code(c) for c in split_sections(module_code("models"))]
    cells += [md("## 6. Training loop\nAdamW, gradient clipping 1.0, early stopping and best-checkpoint selection on "
                 "the dev joint score (mean of intent accuracy and slot micro-F1). Epoch-level train and dev losses "
                 "and elapsed time are logged."),
              code(module_code("train")),
              md("## 7. Experiment runner and data context\nLoads and verifies the data, builds the embedding table, "
                 "and runs one experiment with checkpointing, subgroup tables and consistency checks."),
              code(module_code("experiment").replace('STAGE1_NAME = "stage1_baseline"', 'STAGE1_NAME = "main_stage1"')),
              code("""
ctx = Context()
assets = ctx.assets
assert len(assets.word_vocab) == 5300 and len(assets.slot_labels) == 108
assert (ctx.emb_stats["pretrained_init"], ctx.emb_stats["random_init"]) == (4731, 567)
display(ctx.df.head())
for s in ("dev", "test"):
    print(s, ctx.oov[s]["train_oov_group"].value_counts().to_dict())
"""),
              md("""
# Stage 1: baseline

**Architecture.** word ids, 300-d embedding (Word2Vec init; the 567 train words without a vector get their own
random row; unseen dev or test words map to `<UNK>`), Linear 300 to 256, sinusoidal positions, 4-layer pre-norm
Transformer encoder (4 heads, FF 768, GELU, dropout 0.15), shared token states, then (a) masked mean pool and a
Linear layer to 60 intent logits and (b) a Linear layer per token to 108 BIO logits.

**Input and output.** Input: `word_ids [B,T]`, `attention_mask [B,T]`. Output: `intent_logits [B,60]`,
`slot_logits [B,T,108]`.

**Loss.** L = 1.0 x CE(intent) + 1.0 x CE(BIO tokens, padding ignored).
**Optimiser.** AdamW lr 3e-4, weight decay 1e-4, batch 64, at most 12 epochs, patience 3, clip 1.0, seed 42.
"""),
              code("""
seed_everything(SEED)
probe = build_model(ModelConfig(), assets, ctx.emb_matrix)
total, trainable = count_params(probe)
print(probe)
print(f"Stage 1 parameters: total {total:,} | trainable {trainable:,} | within the 20,000,000 cap: {total <= PARAM_CAP}")
assert total <= PARAM_CAP
del probe
"""),
              code("""
import shutil
from IPython.display import Image, display as show
FORCE_RETRAIN = True      # False: reuse checkpoints already saved by an earlier run of this notebook
if FORCE_RETRAIN:
    for p in list(RESULTS_DIR.glob("main_*")):
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    for p in CKPT_DIR.glob("main_*.pt"):
        p.unlink()
stage1_row = run(ctx, "main_stage1", ModelConfig(), TrainConfig(), rung="NB", note="notebook Stage 1")
ck = torch.load(CKPT_DIR / "main_stage1.pt", map_location="cpu")
display(pd.DataFrame(ck["history"]).round(4))
print("best epoch:", ck["best_epoch"], "| best dev joint:", round(ck["dev"]["joint_score"], 2),
      "| checkpoint:", CKPT_DIR / "main_stage1.pt", "| training time (s):", round(ck["elapsed_s"], 1))
show(Image(filename=str(RESULTS_DIR / "main_stage1" / "curve.png")))
"""),
              md("### Stage 1: reload the best checkpoint and evaluate (no further training)"),
              code(module_code("posthoc")),
              code("""
def own_decoding(run_name, split):
    model, ck = load_member(ctx, run_name)
    data = ctx.datasets(model.cfg, ck["meta"].get("k_extra"), splits=(split,))[split]
    loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=False, collate_fn=make_collate(assets))
    r = predict(model, loader, assets)
    del model
    return r, ck

retrained = {}
r, ck = own_decoding("main_stage1", "dev")
table, _ = oov_table(r["pred_frame"], ctx.oov["dev"])
retrained["Stage1"] = {"intent_accuracy": r["intent_accuracy"], "slot_micro_f1": r["slot_micro_f1"]}
print(f"Stage 1 dev (retrained here): intent accuracy {r['intent_accuracy']:.2f} | slot micro-F1 {r['slot_micro_f1']:.2f}")
display(table.round(2))
"""),
              md("""
# Stage 2: diagnosis and improvement

**Diagnosed Stage-1 weaknesses (dev).** Slot F1 collapses on utterances whose slot contains a word never seen in
training (Slot-Train-OOV), and the gap survives matching on length and slot count. Intent accuracy drops sharply
when an unseen word occurs outside the slots (NonSlot-Train-OOV). The Stage-1 curve was still improving at its last
epochs under a constant learning rate.

**Method ladder.** Every rung screens variants on one seed, combines helpful ones greedily, and confirms the top
candidates on three seeds, always on dev. Rungs: A training recipe; B open-vocabulary word representations; C slot
and intent heads and task interaction; D encoder family and size; R regularisers; H random search; E decoding
options and ensembles. All runs are listed below, including the unsuccessful ones.
"""),
              code("""
lb_path = RESULTS_DIR / "leaderboard.csv"
if lb_path.exists():
    lb = pd.read_csv(lb_path)
    lb = lb[lb["rung"] != "NB"]
    cols = ["run", "rung", "note", "flagged", "params_total", "best_epoch", "dev_intent", "dev_slot_f1", "dev_joint",
            "f1_no_oov", "f1_slot_oov", "gain_slot_oov", "targeted_adv"]
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_colwidth", 90)
    display(lb[[c for c in cols if c in lb.columns]].sort_values(["rung", "dev_joint"]))
"""),
              md("## Stage 2: train every member of the selected systems"),
              code(f"""
MEMBER_SPECS = {pprint.pformat(specs, width=110)}
SYSTEM_SPECS = {pprint.pformat(systems, width=110)}

for name, spec in MEMBER_SPECS.items():
    print("=" * 90)
    print(name, "|", spec["note"])
    mcfg = ModelConfig.from_dict(spec["model_config"])
    tcfg = TrainConfig.from_dict(spec["train_config"])
    row = run(ctx, "main_" + name, mcfg, tcfg, init=None if spec["init"] == "fresh" else spec["init"],
              k_extra=spec["k_extra"], frozen_dim=spec["frozen_dim"], rung="NB", note="notebook retrain of " + name)
    assert row["params_total"] <= PARAM_CAP
    ck = torch.load(CKPT_DIR / f"main_{{name}}.pt", map_location="cpu")
    print(f"parameters: total {{ck['params_total']:,}} | trainable {{ck['params_trainable']:,}} | best epoch {{ck['best_epoch']}} of "
          f"{{ck['epochs_run']}} | best dev joint {{ck['dev']['joint_score']:.2f}} | checkpoint {{CKPT_DIR / f'main_{{name}}.pt'}} | "
          f"training time {{ck['elapsed_s']:.0f}} s")
    print(f"dev now: intent {{row['dev_intent']}} slot {{row['dev_slot_f1']}} | ladder run: intent {{spec['ladder_dev_intent']}} slot {{spec['ladder_dev_slot_f1']}}")
    display(pd.DataFrame(ck["history"]).round(4).tail(12))
    show(Image(filename=str(RESULTS_DIR / ("main_" + name) / "curve.png")))
"""),
              md("""
## Final evaluation: one-time test results and reproduction check

Test is evaluated **once**, by `run_final_test.py`, on the saved checkpoints of the selected runs, after every
selection was made on dev. This notebook never evaluates test. It reads those results and checks that the systems match
the ones tested. It also retrains the same systems and compares their dev numbers with the logged ones, asserting a
largest gap of at most one point. GPU kernels are not bit-deterministic, so a retrain can differ slightly and can stop
at a different epoch. The reported test numbers belong to the saved checkpoints, not to these retrained models.
"""),
              code("""
import json
lb_now = rebuild_leaderboard()
decoder = Decoder(assets)
golds = {"dev": gold_frame(ctx, "dev")}
cache = {}
FINAL = RESULTS_DIR / "final"
assert (FINAL / "final_comparison.csv").exists(), "run run_final_test.py first: it holds the one-time test evaluation"
lock = json.loads((FINAL / "TEST_EVALUATED.json").read_text())
print("test was evaluated once, by run_final_test.py, at", lock["time"])
# The systems that were tested must be the systems this notebook retrains, with the same members and decode.
assert set(lock["systems"]) == set(SYSTEM_SPECS), "the notebook and the one-time test disagree on the systems"
for label, s in SYSTEM_SPECS.items():
    locked = lock["systems"][label]
    assert locked["members"] == s["members"] and locked["decode"] == s["decode"] and locked["flagged"] == s["flagged"], label
logged = pd.read_csv(FINAL / "final_comparison.csv")
logged = logged[logged["split"] == "dev"].set_index("system")
for label, s in SYSTEM_SPECS.items():
    system = System(label, ["main_" + m for m in s["members"]], Decode(**s["decode"]), s["flagged"], s.get("mc", 0))
    m, frame = score_system(ctx, system, "dev", golds["dev"], decoder, cache)
    table, _ = oov_table(frame, ctx.oov["dev"])
    retrained[label] = {"intent_accuracy": m["intent_accuracy"], "slot_micro_f1": m["slot_micro_f1"]}
    print(label, "dev (retrained here)")
    display(table.round(2))
check = pd.DataFrame([{"system": k, "intent retrained": round(v["intent_accuracy"], 2), "intent logged": logged.loc[k, "intent_accuracy"],
                       "slot F1 retrained": round(v["slot_micro_f1"], 2), "slot F1 logged": logged.loc[k, "slot_micro_f1"]}
                      for k, v in retrained.items()])
check["intent diff"] = (check["intent retrained"] - check["intent logged"]).round(2)
check["slot F1 diff"] = (check["slot F1 retrained"] - check["slot F1 logged"]).round(2)
display(check)
REPRO_TOL = 1.0    # points; retrained numbers may differ slightly because GPU kernels are not bit-deterministic
worst = check[["intent diff", "slot F1 diff"]].abs().max().max()
print(f"largest gap between retrained and logged dev numbers: {worst:.2f} points (tolerance {REPRO_TOL})")
assert worst <= REPRO_TOL, "retrained models do not reproduce the logged dev numbers"
display(pd.read_csv(FINAL / "final_comparison.csv").drop(columns=["members"]))
# The error analysis describes the strict-track system that is best on dev, not the first key alphabetically.
strict_names = [k for k in SYSTEM_SPECS if not SYSTEM_SPECS[k]["flagged"] and not SYSTEM_SPECS[k].get("mc", 0)]
PRIMARY = max(strict_names, key=lambda k: (logged.loc[k, "intent_accuracy"] + logged.loc[k, "slot_micro_f1"]) / 2)
print("primary system (strict track, best on dev):", PRIMARY)
"""),
              md("### Paired bootstrap, difficulty-matched OOV gap and subgroup tables (test, one-time evaluation)"),
              code("""
boot = pd.read_csv(FINAL / "bootstrap_vs_stage1.csv")
display(boot[boot["split"] == "test"])
display(pd.read_csv(FINAL / "matched_control_test.csv"))
oov = pd.read_csv(FINAL / "final_oov_tables.csv")
display(oov[oov["split"] == "test"])
"""),
              md("### Error cases of the improved model (test)"),
              code("""
errs = pd.read_csv(FINAL / f"error_cases_{PRIMARY}.csv")
print(len(errs), "test utterances with an intent or slot error for", PRIMARY)
pd.set_option("display.max_colwidth", 120)
display(errs.groupby("error").head(3))
"""),
              md("""
## Notes for the report
* Dev numbers are recomputed here from retrained checkpoints and checked against the ladder log.
* Test numbers are the one-time results of `run_final_test.py` on the saved ladder checkpoints.
* Error analysis, limitations and future work belong in `report.pdf`.
""")]
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    out = ROOT / "main.ipynb"
    nbformat.write(nb, out)
    print("wrote", out, "| cells:", len(cells), "| members:", members)
    return out


if __name__ == "__main__":
    build()
