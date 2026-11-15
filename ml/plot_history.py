"""
Krzywe uczenia (loss i accuracy: train vs val) z models/history<tag>.json -> PNG do rozdzialu wynikow.
Zaznacza epoke najlepszego modelu (min val_loss) — to ten checkpoint trafia do ewaluacji.

Uzycie (Colab):  !python ml/plot_history.py            # lub --tag _noaug
"""
import os
import json
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/content/drive/MyDrive/ai-image-detector")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models_dir = os.path.join(args.data_root, "models")
    hist = json.load(open(os.path.join(models_dir, f"history{args.tag}.json")))
    ep = [h["epoch"] for h in hist]
    best = min(hist, key=lambda h: h["val_loss"])["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in [(axes[0], "loss", "Funkcja straty (cross-entropy)"),
                           (axes[1], "acc", "Dokladnosc (accuracy)")]:
        ax.plot(ep, [h[f"train_{key}"] for h in hist], "o-", label="train")
        ax.plot(ep, [h[f"val_{key}"] for h in hist], "o-", label="val")
        ax.axvline(best, color="gray", ls="--", lw=1, label=f"najlepszy model (epoka {best})")
        ax.set_xlabel("Epoka"); ax.set_title(title); ax.set_xticks(ep); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    out = os.path.join(models_dir, f"learning_curves{args.tag}.png")
    fig.savefig(out, dpi=150)
    print(f"Zapisano: {out} | najlepsza epoka (min val_loss): {best} | "
          f"laczny czas treningu: {sum(h.get('time_s', 0) for h in hist) / 60:.1f} min")


if __name__ == "__main__":
    main()
