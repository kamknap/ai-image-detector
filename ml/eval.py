"""
Etap 4 — Ewaluacja wytrenowanego modelu (Real vs AI-generated).

Co robi:
  1. Wczytuje checkpoint (domyslnie models/efficientnet_b0_best.pt) i manifesty testowe.
  2. Liczy predykcje na test.csv (in-distribution) ORAZ test_heldout.csv (generalizacja
     na generator nigdy nie widziany w treningu — Imagen).
  3. Dla kazdego zbioru: accuracy, precision, recall, F1, AUC + macierz pomylek (PNG)
     + krzywa ROC (PNG) + metryki per-generator (ktore generatory sa "latwe", a ktore nie).
  4. Kalibracja (czy deklarowana pewnosc = faktyczna trafnosc): ECE, log-loss, Brier + diagram
     niezawodnosci; 95% przedzialy ufnosci (bootstrap) dla metryk; tabela per zrodlo danych.
  5. Zapisuje wszystko do models/eval{tag}/ na Drive:
     - metrics_<zbior>.json        (metryki zbiorcze + ci95)
     - summary.csv                 (jedna tabela: zbiory x metryki z CI — do rozdzialu wynikow)
     - per_generator_<zbior>.csv   (tabela do aneksu pracy)
     - per_source_<zbior>.csv      (COCOAI vs OpenFake: specyficznosc, recall, AUC)
     - predictions_<zbior>.csv     (path, label, prob — do dalszych analiz, np. progu)
     - confusion_<zbior>.png, roc_<zbior>.png, calibration_<zbior>.png (wykresy)

Uzycie (Colab, po treningu; dane rozpakowywane z aidata_images.tar jak w treningu):
    !python ml/eval.py                               # domyslny checkpoint (best)
    !python ml/eval.py --tag _noaug                  # checkpoint biegu ablacyjnego
    !python ml/eval.py --arch vit_small --tag _vits  # ViT (architektura/normalizacja z checkpointu)
    !python ml/eval.py --from-predictions            # tylko przelicz metryki/wykresy z zapisanych
                                                     # predictions_*.csv (bez modelu i GPU, sekundy)
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
from train import build_model, prepare_image_root, manifest_dir


def parse_args():
    p = argparse.ArgumentParser(description="Ewaluacja modelu (test + test_heldout)")
    p.add_argument("--data-root", default="/content/drive/MyDrive/ai-image-detector")
    p.add_argument("--checkpoint", default="",
                   help="Sciezka do checkpointu .pt (domyslnie: models/<arch>_best<tag>.pt)")
    p.add_argument("--arch", default="efficientnet_b0",
                   help="Tylko do domyslnej sciezki checkpointu; architektura i tak jest czytana z checkpointu")
    p.add_argument("--local-root", default="/content/aidata")
    p.add_argument("--img-size", type=int, default=0,
                   help="0 = wez z checkpointu (ta sama rozdzielczosc co w treningu)")
    p.add_argument("--batch-size", type=int, default=128, help="Bez gradientow miesci sie wiecej niz w treningu")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-cache", action="store_true", help="Czytaj obrazy prosto z Drive (WOLNE)")
    p.add_argument("--cache-local", action="store_true", help=argparse.SUPPRESS)  # zgodnosc wstecz
    p.add_argument("--tag", default="", help="Sufiks biegu (spojny z --tag z treningu)")
    p.add_argument("--out-tag", default=None,
                   help="Sufiks katalogu wynikow eval<out-tag> (domyslnie = --tag), np. _last dla last.pt")
    p.add_argument("--from-predictions", action="store_true",
                   help="Nie uruchamiaj modelu — przelicz metryki z istniejacych predictions_*.csv")
    p.add_argument("--n-boot", type=int, default=1000, help="Liczba prob bootstrap dla 95% CI")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def predict(model, manifest, transform, image_root, args, device):
    """Zwraca DataFrame z manifestu + kolumny prob (P(fake)) i pred (0/1).
    Kolejnosc wierszy odpowiada manifestowi (shuffle=False), wiec mozemy je skleic 1:1."""
    df = pd.read_csv(manifest)
    dl = DataLoader(ArtifactDataset(df, transform, root=image_root), batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    model.eval()
    probs = []
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            out = model(x)                              # logity [B, 2]
        p = torch.softmax(out, dim=1)[:, 1]             # P(klasa 1 = fake) — nasz confidence score
        probs.append(p.float().cpu().numpy())
    df["prob"] = np.concatenate(probs)
    df["pred"] = (df["prob"] >= 0.5).astype(int)        # domyslny prog 0.5
    return df


ECE_BINS = 15
CI_METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "auc", "ece"]


def ece_score(y, prob, n_bins=ECE_BINS):
    """Expected Calibration Error dla klasyfikacji binarnej.
    Pewnosc predykcji = max(p, 1-p) (w [0.5, 1]). Dzielimy na przedzialy i porownujemy srednia
    pewnosc ze srednia trafnoscia w kazdym; ECE = srednia wazona |trafnosc - pewnosc|.
    ECE=0.02 oznacza, ze model myli sie srednio o 2 pp w ocenie wlasnej pewnosci."""
    y, prob = np.asarray(y), np.asarray(prob)
    conf = np.maximum(prob, 1 - prob)
    correct = ((prob >= 0.5).astype(int) == y)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _point_metrics(y, prob, pred):
    """Metryki punktowe na tablicach numpy (uzywane tez w bootstrapie, wiec bez pandas)."""
    from sklearn.metrics import roc_auc_score
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(y),
        "precision": prec,                              # ile z "AI" to naprawde AI (falszywe alarmy)
        "recall": rec,                                  # ile fake'ow wykryto (przeoczenia)
        "specificity": tn / (tn + fp) if tn + fp else 0.0,   # ile realnych poprawnie uznano za realne
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "auc": roc_auc_score(y, prob) if len(np.unique(y)) == 2 else float("nan"),
        "ece": ece_score(y, prob),
        "confusion_matrix": [[tn, fp], [fn, tp]],       # [[TN, FP], [FN, TP]]
    }


def bootstrap_ci(y, prob, pred, n_boot, seed):
    """95% przedzialy ufnosci metoda bootstrap percentylowy: losujemy ze zwracaniem n obrazow
    ze zbioru testowego n_boot razy i liczymy metryke na kazdej probie. Pokazuje, na ile wynik
    zalezy od konkretnej probki testowej (wazne przy malym held-out i przy porownaniu modeli)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    samples = {k: [] for k in CI_METRICS}
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        m = _point_metrics(y[i], prob[i], pred[i])
        for k in CI_METRICS:
            samples[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in samples.items()}


def compute_metrics(df, n_boot=0, seed=42):
    """Metryki zbiorcze. Konwencja: klasa pozytywna = fake (1) — 'wykrycie' oznacza wskazanie AI."""
    from sklearn.metrics import log_loss, brier_score_loss
    y = df["label"].to_numpy(); pred = df["pred"].to_numpy()
    prob = df["prob"].to_numpy().clip(1e-7, 1 - 1e-7)
    m = {"n": int(len(df))}
    m.update(_point_metrics(y, prob, pred))
    m["log_loss"] = float(log_loss(y, prob, labels=[0, 1]))   # kara za pewne pomylki (to samo co val_loss)
    m["brier"] = float(brier_score_loss(y, prob))            # blad kwadratowy prawdopodobienstwa
    m["frac_conf_gt_099"] = float((np.maximum(prob, 1 - prob) > 0.99).mean())  # odsetek "bardzo pewnych"
    if n_boot:
        m["ci95"] = bootstrap_ci(y, prob, pred, n_boot, seed)
    return m


def per_source_table(df):
    """COCOAI vs OpenFake: czy trudnosc zalezy od zrodla danych (inne tresci, inne wersje generatorow)."""
    from sklearn.metrics import roc_auc_score
    rows = []
    for src, g in df.groupby("source"):
        real, fake = g[g.label == 0], g[g.label == 1]
        rows.append({"source": src, "n_real": len(real), "n_fake": len(fake),
                     "specificity": float((real.pred == 0).mean()) if len(real) else np.nan,
                     "recall": float((fake.pred == 1).mean()) if len(fake) else np.nan,
                     "accuracy": float((g.pred == g.label).mean()),
                     "auc": float(roc_auc_score(g.label, g.prob)) if g.label.nunique() == 2 else np.nan,
                     "ece": ece_score(g.label, g.prob)})
    return pd.DataFrame(rows)


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


def plot_calibration(df, title, out_png, n_bins=ECE_BINS):
    """Diagram niezawodnosci (reliability diagram): os X = deklarowana pewnosc, os Y = faktyczna
    trafnosc w danym przedziale. Idealna kalibracja = przekatna; slupki PONIZEJ przekatnej =
    model nadmiernie pewny siebie. Dolny panel: ile predykcji wpada do kazdego przedzialu."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prob = df["prob"].to_numpy(); y = df["label"].to_numpy()
    conf = np.maximum(prob, 1 - prob)
    correct = ((prob >= 0.5).astype(int) == y)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2
    acc = np.array([correct[idx == b].mean() if (idx == b).any() else np.nan for b in range(n_bins)])
    cnt = np.array([(idx == b).sum() for b in range(n_bins)])
    w = edges[1] - edges[0]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4.8, 5.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.bar(centers, acc, width=w * 0.95, edgecolor="black", alpha=0.8, label="trafnosc w przedziale")
    ax.plot([0.5, 1], [0.5, 1], "--", color="gray", label="idealna kalibracja")
    ax.set_ylabel("Trafnosc"); ax.set_ylim(0.4, 1.02)
    ax.set_title(f"{title}\nECE = {ece_score(y, prob):.4f}"); ax.legend(loc="upper left", fontsize=8)
    ax2.bar(centers, cnt, width=w * 0.95, color="gray")
    ax2.set_yscale("log"); ax2.set_ylabel("Liczba"); ax2.set_xlabel("Pewnosc predykcji  max(p, 1-p)")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def summary_row(name, m):
    row = {"zbior": name, "n": m["n"]}
    for k in CI_METRICS:
        row[k] = round(m[k], 4)
        if "ci95" in m:
            lo, hi = m["ci95"][k]
            row[f"{k}_ci95"] = f"[{lo:.4f}, {hi:.4f}]"
    row["log_loss"] = round(m["log_loss"], 4); row["brier"] = round(m["brier"], 4)
    return row


def main():
    args = parse_args()
    out_tag = args.tag if args.out_tag is None else args.out_tag
    out_dir = os.path.join(args.data_root, "models", f"eval{out_tag}")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.from_predictions:
        print(f"[info] tryb --from-predictions: czytam predictions_*.csv z {out_dir}")
        model = None
    else:
        ckpt_path = args.checkpoint or os.path.join(args.data_root, "models",
                                                    f"{args.arch}_best{args.tag}.pt")
        print(f"[info] device={device} | checkpoint={ckpt_path}\n[info] wyniki -> {out_dir}")
        # model: ta sama architektura co w treningu, wagi z checkpointu (nie z ImageNet)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        arch = ckpt.get("arch", "efficientnet_b0")      # starsze checkpointy nie mialy tego pola
        img_size = args.img_size or ckpt.get("img_size", 224)
        model, (mean, std) = build_model(device, pretrained=False, arch=arch, img_size=img_size)
        model.load_state_dict(ckpt["model_state"])
        mean, std = ckpt.get("norm_mean", mean), ckpt.get("norm_std", std)
        print(f"[info] checkpoint z epoki {ckpt.get('epoch')} | "
              f"val_acc={ckpt.get('val_acc', float('nan')):.4f} | arch={arch} | img_size={img_size}")
        _, eval_tf = build_transforms(img_size, mean, std)         # transform walidacyjny (bez augmentacji!)
        image_root = prepare_image_root(args.data_root, args.local_root, args.no_cache)
        man = manifest_dir(image_root, args.data_root)

    summary = []
    for name in ["test", "test_heldout"]:
        pred_csv = os.path.join(out_dir, f"predictions_{name}.csv")
        if model is None:
            if not os.path.exists(pred_csv):
                print(f"[{name}] brak {pred_csv} — pomijam"); continue
            df = pd.read_csv(pred_csv)
        else:
            manifest = os.path.join(man, f"{name}.csv")
            if not os.path.exists(manifest):
                print(f"[{name}] brak manifestu — pomijam"); continue
            df = predict(model, manifest, eval_tf, image_root, args, device)
            df[["path", "label", "generator", "source", "prob", "pred"]].to_csv(pred_csv, index=False)

        print(f"\n=== {name} ===")
        m = compute_metrics(df, args.n_boot, args.seed)
        json.dump(m, open(os.path.join(out_dir, f"metrics_{name}.json"), "w"), indent=2)
        row = summary_row(name, m); summary.append(row)
        for k in CI_METRICS + ["log_loss", "brier"]:
            ci = f"  95% CI {row[k + '_ci95']}" if f"{k}_ci95" in row else ""
            print(f"  {k:12s} {row[k]:.4f}{ci}")
        print(f"  macierz pomylek [[TN, FP], [FN, TP]] = {m['confusion_matrix']}")
        print(f"  odsetek predykcji z pewnoscia > 0.99: {m['frac_conf_gt_099']:.3f}")

        per_generator_table(df).to_csv(os.path.join(out_dir, f"per_generator_{name}.csv"), index=False)
        src = per_source_table(df)
        src.to_csv(os.path.join(out_dir, f"per_source_{name}.csv"), index=False)
        print("  per zrodlo:\n" + src.round(4).to_string(index=False).replace("\n", "\n    ").join(["    ", ""]))
        plot_confusion(m["confusion_matrix"], f"Macierz pomylek — {name}",
                       os.path.join(out_dir, f"confusion_{name}.png"))
        plot_roc(df, f"Krzywa ROC — {name}", os.path.join(out_dir, f"roc_{name}.png"))
        plot_calibration(df, f"Kalibracja — {name}", os.path.join(out_dir, f"calibration_{name}.png"))

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    print(f"\nGotowe. Wszystkie pliki w: {out_dir}")


if __name__ == "__main__":
    main()
