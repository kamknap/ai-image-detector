"""
Etap 2 (wersja wielozrodlowa) — Przygotowanie danych: ArtiFact + COCOAI + OpenFake.

Cel: zbudowac zbalansowany (50/50 Real vs AI), zroznicowany generatorowo zbior z TRZECH zrodel,
ujednolicic do rozdzielczosci 200 px i zapisac manifesty train/val/test + osobny test_heldout
(generalizacja na generator NIE widziany w treningu).

Zrodla i ich rola:
  - ArtiFact (bitmind/ArtiFact)      : generatory 2023 (GAN + wczesne diffusion). Pliki na dysku.
  - COCOAI   (Defactify_Image_Dataset): SD2.1/SDXL/SD3/DALLE3/MJ6, realne z COCO. Parquet (96k, ~7.5 GB).
  - OpenFake (ComplexDataLab/OpenFake): NAJNOWSZE (FLUX, MJ6/7, GPT-Image-1, SD3.5, Imagen...). Parquet.
                                        UWAGA: calosc = 1.06 TB -> STRUMIENIUJEMY podzbior, nie pobieramy calosci.

Wspolny manifest ma kolumny: path, label (0=real/1=fake), generator, source.

Uzycie (w Colab, po zamontowaniu Drive):
    pip install -U "huggingface_hub>=0.23" datasets "pandas==2.2.2" "pillow<12.0" scikit-learn
    python ml/prepare_data.py
Najwazniejsze parametry: --holdout (generator wykluczony z treningu), --modern-per-gen-cap, --skip-*,
--data-root (gdzie zapisac dane na Drive), --openfake-max-scan (limit czasu/transferu OpenFake).

Funkcje build_transforms() i ArtifactDataset (na dole) sa importowalne w treningu (Etap 3).
"""
import os
import glob
import random
import argparse

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Konfiguracja
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Przygotowanie danych (ArtiFact + COCOAI + OpenFake)")
    p.add_argument("--data-root", default="/content/drive/MyDrive/ai-image-detector",
                   help="Glowny katalog projektu na Drive. Wszystkie dane laduja w jego podfolderach.")
    p.add_argument("--save-size", type=int, default=200,
                   help="Rozdzielczosc zapisu (ujednolicenie domeny; ArtiFact ma natywnie 200)")
    p.add_argument("--img-size", type=int, default=224, help="Wejscie modelu (uzywane w transforms)")
    # capy na generator (roznorodnosc)
    p.add_argument("--artifact-cap", type=int, default=15000, help="Max obrazow na generator z ArtiFact")
    p.add_argument("--modern-per-gen-cap", type=int, default=8000,
                   help="Max obrazow na rodzine generatora z OpenFake/COCOAI")
    p.add_argument("--openfake-real-cap", type=int, default=60000, help="Max realnych pobranych z OpenFake")
    p.add_argument("--openfake-max-scan", type=int, default=120000,
                   help="Limit wierszy strumienia OpenFake do przejrzenia (ogranicza czas i transfer)")
    # budzety splitow (na klase)
    p.add_argument("--n-train", type=int, default=80000)
    p.add_argument("--n-val",   type=int, default=10000)
    p.add_argument("--n-test",  type=int, default=10000)
    # held-out: generator(y) WYKLUCZONE z treningu -> osobny test generalizacji
    p.add_argument("--holdout", nargs="+", default=["imagen"],
                   help="Rodziny generatorow trzymane wylacznie w test_heldout (np. imagen gpt-image)")
    p.add_argument("--seed", type=int, default=42)
    # mozliwosc pominiecia ciezkich krokow przy ponownym uruchomieniu
    p.add_argument("--skip-artifact", action="store_true")
    p.add_argument("--skip-cocoai", action="store_true")
    p.add_argument("--skip-openfake", action="store_true")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Narzedzia wspolne
# --------------------------------------------------------------------------- #
def save_resized(pil_img, out_path, size):
    """Zapisuje obraz przeskalowany do size x size jako JPEG (ujednolicenie domeny + oszczednosc miejsca)."""
    img = pil_img.convert("RGB").resize((size, size))
    img.save(out_path, "JPEG", quality=95)


def normalize_generator(model_name):
    """Sprowadza dziesiatki wariantow (loRA/finetune) do RODZINY generatora — sensowny cap i held-out.
    Np. 'sdxl-epic-realism' -> 'sdxl', 'flux.1-dev' -> 'flux', 'midjourney-6' -> 'midjourney'."""
    m = str(model_name).lower()
    families = [
        ("flux", "flux"), ("midjourney", "midjourney"), ("dall", "dalle"), ("gpt-image", "gpt-image"),
        ("imagen", "imagen"), ("ideogram", "ideogram"), ("grok", "grok"), ("hidream", "hidream"),
        ("recraft", "recraft"), ("chroma", "chroma"), ("sd-3", "sd-3.5"), ("sd3", "sd-3.5"),
        ("sdxl", "sdxl"), ("sd-2", "sd-2.1"), ("sd-1", "sd-1.5"), ("stable", "sdxl"),
    ]
    for key, fam in families:
        if key in m:
            return fam
    return m  # fallback: oryginalna nazwa


# --------------------------------------------------------------------------- #
# 1. ArtiFact -> indeks (pliki juz na dysku)
# --------------------------------------------------------------------------- #
def prepare_artifact(data_root, skip):
    artifact_dir = os.path.join(data_root, "artifact_raw")
    if not skip:
        from huggingface_hub import snapshot_download
        print(f"[artifact] pobieranie bitmind/ArtiFact -> {artifact_dir} (~31.7 GB, wznawialne)")
        snapshot_download(repo_id="bitmind/ArtiFact", repo_type="dataset", local_dir=artifact_dir)

    metas = glob.glob(os.path.join(artifact_dir, "**", "metadata.csv"), recursive=True)
    if not metas:
        print("[artifact] UWAGA: brak metadata.csv — pomijam ArtiFact (sprawdz strukture).")
        return pd.DataFrame(columns=["path", "label", "generator", "source"])

    rows = []
    for mfile in metas:
        folder = os.path.dirname(mfile)
        generator = os.path.basename(folder)
        df = pd.read_csv(mfile)
        cols = {c.lower(): c for c in df.columns}
        pcol = cols.get("image_path") or cols.get("path") or list(df.columns)[0]
        lcol = cols.get("target") or cols.get("label")
        for _, r in df.iterrows():
            rows.append({"path": os.path.join(folder, str(r[pcol])),
                         "label": int(r[lcol]), "generator": generator, "source": "artifact"})
    out = pd.DataFrame(rows)
    print(f"[artifact] obrazow: {len(out)} | klasy: {out.label.value_counts().to_dict()}")
    return out


# --------------------------------------------------------------------------- #
# 2. COCOAI -> wypakowanie do plikow + indeks
# --------------------------------------------------------------------------- #
COCOAI_LABEL_B = {0: "real", 1: "sd-2.1", 2: "sdxl", 3: "sd-3.5", 4: "dalle", 5: "midjourney"}

def prepare_cocoai(data_root, save_size, per_gen_cap, skip, seed):
    out_dir = os.path.join(data_root, "cocoai_extracted")
    manifest = os.path.join(out_dir, "_index.csv")
    if skip and os.path.exists(manifest):
        print("[cocoai] pomijam (uzywam istniejacego _index.csv)")
        return pd.read_csv(manifest)

    from datasets import load_dataset, concatenate_datasets
    os.makedirs(out_dir, exist_ok=True)
    print("[cocoai] load_dataset (96k, ~7.5 GB)...")
    ds = load_dataset("Rajarshi-Roy-research/Defactify_Image_Dataset")
    pool = concatenate_datasets([ds[s] for s in ds.keys()])  # laczymy wszystkie splity, podzielimy sami

    counts, rows = {}, []
    order = list(range(len(pool)))
    random.Random(seed).shuffle(order)
    for i in order:
        ex = pool[i]
        label = int(ex["Label_A"])                         # 0 real / 1 fake
        gen = COCOAI_LABEL_B.get(int(ex["Label_B"]), "unknown")
        key = "real" if label == 0 else gen
        if counts.get(key, 0) >= per_gen_cap:              # cap per generator (i osobno realne)
            continue
        counts[key] = counts.get(key, 0) + 1
        path = os.path.join(out_dir, f"cocoai_{i}.jpg")
        if not os.path.exists(path):
            save_resized(ex["Image"], path, save_size)
        rows.append({"path": path, "label": label, "generator": gen, "source": "cocoai"})
    out = pd.DataFrame(rows)
    out.to_csv(manifest, index=False)
    print(f"[cocoai] zapisano: {len(out)} | per-bucket: {counts}")
    return out


# --------------------------------------------------------------------------- #
# 3. OpenFake -> STRUMIENIOWO pobierany podzbior (calosc to 1.06 TB!)
# --------------------------------------------------------------------------- #
def prepare_openfake(data_root, save_size, per_gen_cap, real_cap, max_scan, skip, seed):
    out_dir = os.path.join(data_root, "openfake_extracted")
    manifest = os.path.join(out_dir, "_index.csv")
    if skip and os.path.exists(manifest):
        print("[openfake] pomijam (uzywam istniejacego _index.csv)")
        return pd.read_csv(manifest)

    import io
    from PIL import Image, ImageFile
    from datasets import load_dataset, Image as HFImage
    ImageFile.LOAD_TRUNCATED_IMAGES = True   # toleruj lekko uciete pliki
    os.makedirs(out_dir, exist_ok=True)
    print(f"[openfake] streaming 'core' — czytam sekwencyjnie max {max_scan} wierszy "
          f"(dane sa juz wymieszane). Postep co 2000:")
    # OpenFake wymaga nazwy konfiguracji: 'core' = glowny wyselekcjonowany zbior (ten chcemy),
    # 'reddit' = dodatkowe realne zdjecia. Bez tego load_dataset rzuca "Config name is missing".
    # BEZ .shuffle() — duzy bufor blokowalby start na kilka GB; 'core' jest juz wymieszany.
    stream = load_dataset("ComplexDataLab/OpenFake", "core", split="train", streaming=True)
    # decode=False -> 'datasets' NIE dekoduje obrazu automatycznie (omija bug PIL.getexif na
    # uszkodzonym EXIF, ktory wywalal caly proces). Surowe bajty otwieramy sami, z obsluga bledow.
    stream = stream.cast_column("image", HFImage(decode=False))

    counts, rows, seen, bad = {}, [], 0, 0
    for ex in stream:
        seen += 1
        is_real = (str(ex["label"]).lower() == "real")
        if is_real:
            key, label, gen, cap = "real", 0, "real", real_cap
        else:
            gen = normalize_generator(ex["model"])
            key, label, cap = gen, 1, per_gen_cap
        if counts.get(key, 0) < cap:
            path = os.path.join(out_dir, f"openfake_{seen}.jpg")
            try:
                if not os.path.exists(path):
                    img = Image.open(io.BytesIO(ex["image"]["bytes"]))
                    save_resized(img, path, save_size)
                counts[key] = counts.get(key, 0) + 1     # licz dopiero po udanym zapisie
                rows.append({"path": path, "label": label, "generator": gen, "source": "openfake"})
            except Exception:
                bad += 1                                 # uszkodzony obraz -> pomijamy, nie przerywamy
        if seen % 2000 == 0:
            print(f"   ...przejrzano {seen}, zebrano {len(rows)}, pominieto {bad} | rodzin: {len(counts)}")
        if seen >= max_scan:   # twardy limit -> krok zawsze sie konczy
            break
    out = pd.DataFrame(rows)
    out.to_csv(manifest, index=False)
    print(f"[openfake] zapisano: {len(out)} (przejrzano {seen}, pominieto {bad}) | per-bucket: {counts}")
    return out


# --------------------------------------------------------------------------- #
# 4. Laczenie, cap, balans, held-out, split
# --------------------------------------------------------------------------- #
def cap_per_generator(df, cap, seed):
    # losowe podprobkowanie do `cap` na generator (bez deprecated apply-on-grouping-columns)
    parts = [g.sample(min(len(g), cap), random_state=seed) for _, g in df.groupby("generator")]
    return pd.concat(parts).reset_index(drop=True) if parts else df


def build_splits(index, args):
    from sklearn.model_selection import train_test_split
    manifest_dir = os.path.join(args.data_root, "manifests")
    os.makedirs(manifest_dir, exist_ok=True)

    holdout = set(args.holdout)
    # 1) HELD-OUT: fake z wykluczonych generatorow -> tylko test generalizacji (nigdy w treningu)
    heldout_fake = index[(index.label == 1) & (index.generator.isin(holdout))]
    pool = index[~((index.label == 1) & (index.generator.isin(holdout)))].copy()

    # 2) cap per generator TYLKO dla fake (zeby zaden generator nie dominowal);
    #    realnych NIE capujemy per-generator — wszystkie maja generator="real"/zrodlo,
    #    wiec cap zlepialby je w jeden kubelek. Balans 50/50 i tak nastepuje nizej.
    real = pool[pool.label == 0]
    fake = pool[pool.label == 1]
    if args.modern_per_gen_cap:
        fake = cap_per_generator(fake, args.modern_per_gen_cap, args.seed)
    budget = args.n_train + args.n_val + args.n_test
    n = min(budget, len(real), len(fake))
    if n < budget:
        print(f"[split] UWAGA: dostepne {n}/klasa < budzet {budget}. Zmniejsz N_* lub zwieksz capy.")
    real = real.sample(n, random_state=args.seed)
    fake = fake.sample(n, random_state=args.seed)
    balanced = pd.concat([real, fake]).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # 3) stratyfikowany split po (klasa x generator x source)
    balanced["strat"] = (balanced.label.astype(str) + "_" +
                         balanced.generator.astype(str) + "_" + balanced.source.astype(str))
    # rzadkie kombinacje psuja stratyfikacje -> sklejamy do "_rare"
    vc = balanced["strat"].value_counts()
    balanced.loc[balanced.strat.isin(vc[vc < 3].index), "strat"] = "_rare"

    test_frac = args.n_test / budget
    val_frac = args.n_val / (args.n_train + args.n_val)
    trainval, test = train_test_split(balanced, test_size=test_frac,
                                      stratify=balanced["strat"], random_state=args.seed)
    train, val = train_test_split(trainval, test_size=val_frac,
                                  stratify=trainval["strat"], random_state=args.seed)

    # 4) test_heldout: balans z realnymi z puli testowej (zeby byl 50/50)
    n_ho = min(len(heldout_fake), len(test[test.label == 0]))
    heldout = pd.concat([
        heldout_fake.sample(n_ho, random_state=args.seed),
        test[test.label == 0].sample(n_ho, random_state=args.seed),
    ]).sample(frac=1, random_state=args.seed)

    cols = ["path", "label", "generator", "source"]
    for name, part in [("train", train), ("val", val), ("test", test), ("test_heldout", heldout)]:
        out = os.path.join(manifest_dir, f"{name}.csv")
        part[cols].to_csv(out, index=False)
        print(f"[split] {name:13s}: {len(part):>6} | klasy={part.label.value_counts().to_dict()}"
              f" | generatory={part.generator.nunique()} -> {out}")
    print(f"[split] held-out generatory (poza treningiem): {sorted(holdout)}")


# --------------------------------------------------------------------------- #
# 5. Preprocessing — importowalne w treningu (Etap 3)
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(img_size=224):
    from torchvision import transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
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
    """Dataset PyTorch czytajacy obrazy z manifestu CSV (path, label, generator, source).

    Odczyt z Google Drive (montowany przez FUSE) bywa zawodny przy wielu malych plikach:
    zdarza sie PRZEJSCIOWY 'OSError: [Errno 5] Input/output error', mimo ze plik jest OK.
    Dlatego kazdy odczyt ma kilka prob z rosnaca pauza; dopiero potem zglaszamy blad
    (z pelna sciezka, zeby dalo sie zdiagnozowac naprawde uszkodzony plik)."""
    RETRIES = 4        # liczba prob odczytu
    RETRY_WAIT = 1.5   # bazowa pauza [s]; rosnie liniowo: 1.5s, 3s, 4.5s...

    def __init__(self, manifest_csv, transform):
        from PIL import Image
        self._Image = Image
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _open_with_retry(self, path):
        import time
        last_err = None
        for attempt in range(self.RETRIES):
            try:
                return self._Image.open(path).convert("RGB")
            except OSError as e:
                last_err = e
                time.sleep(self.RETRY_WAIT * (attempt + 1))
        raise OSError(f"Nie udalo sie odczytac '{path}' po {self.RETRIES} probach "
                      f"(ostatni blad: {last_err})") from last_err

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = self._open_with_retry(row["path"])
        return self.transform(img), int(row["label"])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.data_root, exist_ok=True)
    print(f"[info] Wszystkie dane zapisuje w: {args.data_root}")
    print(f"[info] Podfoldery: artifact_raw/ cocoai_extracted/ openfake_extracted/ manifests/\n")

    parts = []
    parts.append(prepare_artifact(args.data_root, args.skip_artifact))
    parts.append(prepare_cocoai(args.data_root, args.save_size, args.modern_per_gen_cap,
                                args.skip_cocoai, args.seed))
    parts.append(prepare_openfake(args.data_root, args.save_size, args.modern_per_gen_cap,
                                  args.openfake_real_cap, args.openfake_max_scan,
                                  args.skip_openfake, args.seed))

    index = pd.concat(parts, ignore_index=True)
    print(f"\n[indeks] LACZNIE: {len(index)} | klasy={index.label.value_counts().to_dict()}")
    print(index.groupby("source").size().to_dict())

    build_splits(index, args)
    print("\nGotowe. Manifesty w:", os.path.join(args.data_root, "manifests"))


if __name__ == "__main__":
    main()
