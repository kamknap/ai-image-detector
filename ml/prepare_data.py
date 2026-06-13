"""
Etap 2 — Przygotowanie danych (ArtiFact) jako pojedynczy skrypt.

Robi: pobranie ArtiFact z Hugging Face -> indeks -> cap per generator ->
balans 50/50 -> stratyfikowany split -> zapis manifestow train/val/test.csv.

Uzycie:
    # 1) zaleznosci (raz):
    pip install -U "huggingface_hub>=0.23" pandas scikit-learn pillow

    # 2a) w Google Colab — najpierw zamontuj Drive w KOMORCE notebooka:
    #     from google.colab import drive; drive.mount('/content/drive')
    #     a potem:  !python ml/prepare_data.py
    #
    # 2b) lokalnie — wskaz wlasny katalog:
    #     python ml/prepare_data.py --data-root /sciezka/do/ai-image-detector

Parametry (--help pokazuje wszystkie) maja domyslne wartosci uzgodnione w tabeli.
Funkcje build_transforms() i ArtifactDataset sa importowalne w treningu (Etap 3):
    from ml.prepare_data import build_transforms, ArtifactDataset
"""
import os
import glob
import random
import argparse

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# 1. Konfiguracja / argumenty
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Przygotowanie danych ArtiFact -> manifesty CSV")
    p.add_argument("--data-root", default="/content/drive/MyDrive/ai-image-detector",
                   help="Glowny katalog projektu (na Drive lub lokalnie)")
    p.add_argument("--img-size", type=int, default=224, help="Wejscie modelu (ViT/EffNet=224)")
    p.add_argument("--per-gen-cap", type=int, default=20000, help="Max obrazow na generator")
    p.add_argument("--n-train", type=int, default=80000, help="Obrazy treningowe na klase")
    p.add_argument("--n-val",   type=int, default=10000, help="Obrazy walidacyjne na klase")
    p.add_argument("--n-test",  type=int, default=10000, help="Obrazy testowe na klase")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-download", action="store_true",
                   help="Pomin pobieranie (gdy ArtiFact juz jest na dysku)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 3. Pobranie ArtiFact z Hugging Face
# --------------------------------------------------------------------------- #
def download_artifact(artifact_dir):
    from huggingface_hub import snapshot_download
    print(f"[pobieranie] bitmind/ArtiFact -> {artifact_dir} (~31.7 GB, wznawialne)")
    path = snapshot_download(
        repo_id="bitmind/ArtiFact",
        repo_type="dataset",
        local_dir=artifact_dir,
    )
    print(f"[pobieranie] gotowe: {path}")
    return path


# --------------------------------------------------------------------------- #
# 4. Budowa jednolitego indeksu z metadata.csv
# --------------------------------------------------------------------------- #
def build_index(artifact_dir):
    metas = glob.glob(os.path.join(artifact_dir, "**", "metadata.csv"), recursive=True)
    if not metas:
        raise FileNotFoundError(
            f"Brak metadata.csv w {artifact_dir}. Sprawdz strukture pobranych plikow "
            "(foldery najwyzszego poziomu) i dostosuj parser.")

    rows = []
    for m in metas:
        folder = os.path.dirname(m)
        generator = os.path.basename(folder)          # nazwa zrodla/generatora = nazwa folderu
        df = pd.read_csv(m)
        cols = {c.lower(): c for c in df.columns}      # standaryzacja nazw kolumn
        path_col  = cols.get("image_path") or cols.get("path") or list(df.columns)[0]
        label_col = cols.get("target") or cols.get("label")
        for _, r in df.iterrows():
            rows.append({
                "path": os.path.join(folder, str(r[path_col])),
                "label": int(r[label_col]),            # 0 = real, 1 = fake
                "generator": generator,
            })

    index = pd.DataFrame(rows)
    print(f"[indeks] obrazow: {len(index)} | klasy: {index.label.value_counts().to_dict()}")
    print(f"[indeks] generatory: {index.generator.nunique()}")
    return index


# --------------------------------------------------------------------------- #
# 5. Cap per generator + balans 50/50
# --------------------------------------------------------------------------- #
def cap_and_balance(index, per_gen_cap, budget_per_class, seed):
    capped = (index.groupby("generator", group_keys=False)
                   .apply(lambda g: g.sample(min(len(g), per_gen_cap), random_state=seed)))
    real = capped[capped.label == 0]
    fake = capped[capped.label == 1]
    n = min(budget_per_class, len(real), len(fake))
    if n < budget_per_class:
        print(f"[balans] UWAGA: dostepne {n}/klasa < budzet {budget_per_class}. "
              "Zwieksz --per-gen-cap lub zmniejsz N_*.")
    real = real.sample(n, random_state=seed)
    fake = fake.sample(n, random_state=seed)
    balanced = pd.concat([real, fake]).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"[balans] zbior: {len(balanced)} | {balanced.label.value_counts().to_dict()}")
    return balanced


# --------------------------------------------------------------------------- #
# 6. Stratyfikowany split + zapis manifestow
# --------------------------------------------------------------------------- #
def split_and_save(balanced, manifest_dir, n_train, n_val, n_test, seed):
    from sklearn.model_selection import train_test_split
    os.makedirs(manifest_dir, exist_ok=True)

    balanced["strat"] = balanced.label.astype(str) + "_" + balanced.generator.astype(str)
    total = n_train + n_val + n_test
    test_frac = n_test / total
    val_frac  = n_val / (n_train + n_val)

    trainval, test = train_test_split(
        balanced, test_size=test_frac, stratify=balanced["strat"], random_state=seed)
    train, val = train_test_split(
        trainval, test_size=val_frac, stratify=trainval["strat"], random_state=seed)

    for name, part in [("train", train), ("val", val), ("test", test)]:
        out = os.path.join(manifest_dir, f"{name}.csv")
        part[["path", "label", "generator"]].to_csv(out, index=False)
        print(f"[split] {name}: {len(part):>6} | {part.label.value_counts().to_dict()} -> {out}")


# --------------------------------------------------------------------------- #
# 7. Preprocessing — importowalne w treningu (Etap 3)
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(img_size=224):
    """Zwraca (train_tf, eval_tf). Augmentacja umiarkowana — nie niszczy artefaktow generatora."""
    from torchvision import transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        # transforms.RandomApply([transforms.GaussianBlur(3)], p=0.1),  # opcjonalnie: robustnosc web
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class ArtifactDataset:
    """Dataset PyTorch czytajacy obrazy z manifestu CSV (path, label, generator)."""
    def __init__(self, manifest_csv, transform):
        from PIL import Image
        self._Image = Image
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = self._Image.open(row["path"]).convert("RGB")
        return self.transform(img), int(row["label"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    artifact_dir = os.path.join(args.data_root, "artifact_raw")
    manifest_dir = os.path.join(args.data_root, "manifests")
    os.makedirs(args.data_root, exist_ok=True)

    if not args.skip_download:
        download_artifact(artifact_dir)
    else:
        print("[pobieranie] pominiete (--skip-download)")

    index = build_index(artifact_dir)
    budget = args.n_train + args.n_val + args.n_test
    balanced = cap_and_balance(index, args.per_gen_cap, budget, args.seed)
    split_and_save(balanced, manifest_dir, args.n_train, args.n_val, args.n_test, args.seed)
    print("\nGotowe. Manifesty w:", manifest_dir)


if __name__ == "__main__":
    main()
