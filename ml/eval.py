"""
Etap 4 — Ewaluacja wytrenowanego modelu (Real vs AI-generated).

Co robi:
  1. Wczytuje checkpoint (domyslnie models/efficientnet_b0_best.pt) i manifesty testowe.
  2. Liczy predykcje na test.csv (in-distribution) ORAZ test_heldout.csv (generalizacja
     na generator nigdy nie widziany w treningu — Imagen).
  3. Dla kazdego zbioru: accuracy, precision, recall, F1, AUC + macierz pomylek (PNG)
     + krzywa ROC (PNG) + metryki per-generator (ktore generatory sa "latwe", a ktore nie).
  4. Zapisuje wszystko do models/eval{tag}/ na Drive:
     - metrics_<zbior>.json        (metryki zbiorcze)
     - per_generator_<zbior>.csv   (tabela do aneksu pracy)
     - predictions_<zbior>.csv     (path, label, prob — do dalszych analiz, np. progu)
     - confusion_<zbior>.png, roc_<zbior>.png (wykresy do rozdzialu wynikow)

Uzycie (Colab, po treningu):
    !python ml/eval.py --cache-local                 # domyslny checkpoint (best)
    !python ml/eval.py --checkpoint models/efficientnet_b0_best_noaug.pt --tag _noaug
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import build_transforms, ArtifactDataset
from train import build_model, maybe_cache_local


def parse_args():
    p = argparse.ArgumentParser(description="Ewaluacja modelu (test + test_heldout)")
    p.add_argument("--data-root", default="/content/drive/MyDrive/ai-image-detector")
    p.add_argument("--checkpoint", default="",
                   help="Sciezka do checkpointu .pt (domyslnie: models/efficientnet_b0_best<tag>.pt)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128, help="Bez gradientow miesci sie wiecej niz w treningu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--cache-local", action="store_true", help="Kopia obrazow na dysk lokalny (jak w treningu)")
    p.add_argument("--tag", default="", help="Sufiks biegu (spojny z --tag z treningu)")
    return p.parse_args()


@torch.no_grad()
def predict(model, manifest, transform, args, device):
    """Zwraca DataFrame z manifestu + kolumny prob (P(fake)) i pred (0/1).
    Kolejnosc wierszy odpowiada manifestowi (shuffle=False), wiec mozemy je skleic 1:1."""
    df = pd.read_csv(manifest)
    df = maybe_cache_local(df, args.data_root, args.cache_local)
    tmp = manifest + ".eval.csv"
    df.to_csv(tmp, index=False)
    dl = DataLoader(ArtifactDataset(tmp, transform), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model.eval()
    probs = []
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        out = model(x)                                  # logity [B, 2]
        p = torch.softmax(out, dim=1)[:, 1]             # P(klasa 1 = fake) — nasz confidence score
        probs.append(p.float().cpu().numpy())
    df["prob"] = np.concatenate(probs)
    df["pred"] = (df["prob"] >= 0.5).astype(int)        # domyslny prog 0.5
    return df


def compute_metrics(df):
    """Metryki zbiorcze. Konwencja: klasa pozytywna = fake (1) — 'wykrycie' oznacza wskazanie AI."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, confusion_matrix)
    y, yhat, prob = df["label"], df["pred"], df["prob"]
    return {
        "n": len(df),
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat),          # ile z "AI" to naprawde AI (falszywe alarmy)
        "recall": recall_score(y, yhat),                # ile fake'ow wykryto (przeoczenia)
        "f1": f1_score(y, yhat),
        "auc": roc_auc_score(y, prob),                  # jakosc rankingu, niezalezna od progu 0.5
        "confusion_matrix": confusion_matrix(y, yhat).tolist(),  # [[TN, FP], [FN, TP]]
    }


def per_generator_table(df):
    """Skutecznosc per generator: dla fake'ow = recall (wykrywalnosc), dla real = specyficznosc.
    Gotowa tabela do aneksu — pokazuje, ktore generatory model 'lapie', a ktore go myla."""
    rows = []
    for (gen, label), g in df.groupby(["generator", "label"]):
        rows.append({"generator": gen, "label": int(label), "n": len(g),
                     "accuracy": float((g["pred"] == g["label"]).mean()),
                     "mean_prob_fake": float(g["prob"].mean())})
    return pd.DataFrame(rows).sort_values(["label", "accuracy"]).reset_index(drop=True)


def plot_confusion(cm, title, out_png):
    import matplotlib
    matplotlib.use("Agg")                               # bez okna — tylko zapis do pliku
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Real", "AI"]); ax.set_yticks([0, 1], ["Real", "AI"])
    ax.set_xlabel("Predykcja"); ax.set_ylabel("Etykieta"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def plot_roc(df, title, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(df["label"], df["prob"])
    auc = roc_auc_score(df["label"], df["prob"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="losowy")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = args.checkpoint or os.path.join(args.data_root, "models",
                                                f"efficientnet_b0_best{args.tag}.pt")
    out_dir = os.path.join(args.data_root, "models", f"eval{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[info] device={device} | checkpoint={ckpt_path}\n[info] wyniki -> {out_dir}")

    # model: ta sama architektura co w treningu, wagi z checkpointu (nie z ImageNet)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = build_model(device, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"[info] checkpoint z epoki {ckpt.get('epoch')} | val_acc={ckpt.get('val_acc'):.4f}")

    _, eval_tf = build_transforms(args.img_size)        # transform walidacyjny (bez augmentacji!)
    man = os.path.join(args.data_root, "manifests")

    for name in ["test", "test_heldout"]:
        manifest = os.path.join(man, f"{name}.csv")
        if not os.path.exists(manifest):
            print(f"[{name}] brak manifestu — pomijam"); continue
        print(f"\n=== {name} ===")
        df = predict(model, manifest, eval_tf, args, device)
        m = compute_metrics(df)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
                         indent=2, ensure_ascii=False))
        json.dump(m, open(os.path.join(out_dir, f"metrics_{name}.json"), "w"), indent=2)
        per_generator_table(df).to_csv(os.path.join(out_dir, f"per_generator_{name}.csv"), index=False)
        df[["path", "label", "generator", "source", "prob", "pred"]].to_csv(
            os.path.join(out_dir, f"predictions_{name}.csv"), index=False)
        plot_confusion(m["confusion_matrix"], f"Macierz pomylek — {name}",
                       os.path.join(out_dir, f"confusion_{name}.png"))
        plot_roc(df, f"Krzywa ROC — {name}", os.path.join(out_dir, f"roc_{name}.png"))

    print(f"\nGotowe. Wszystkie pliki w: {out_dir}")


if __name__ == "__main__":
    main()
