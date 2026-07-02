"""
Etap 3 — Trening EfficientNet-B0 (Real vs AI-generated).

Co robi:
  1. Wczytuje manifesty z Etapu 2 (train/val/test) i buduje DataLoadery.
  2. Laduje EfficientNet-B0 z wagami ImageNet (transfer learning) i podmienia glowice na 2-klasowa.
  3. Trenuje z walidacja, mixed precision (AMP), cosine LR, early stopping.
  4. Po KAZDEJ epoce zapisuje checkpoint na Drive (rozlaczenie Colaba nie marnuje postepu).
  5. Zapisuje najlepszy model + historie uczenia (do wykresow w Etapie 4) + szybki test na koncu.

Uzycie (Colab, GPU wlaczony: Runtime -> Change runtime type -> T4 GPU):
    from google.colab import drive; drive.mount('/content/drive')
    %cd /content/ai-image-detector
    !pip install -q "pandas==2.2.2" "pillow<12.0"
    !python ml/train.py                       # domyslne hiperparametry
    !python ml/train.py --epochs 10 --batch-size 96 --cache-local   # szybciej (kopia danych na dysk lokalny)

Wynik: models/efficientnet_b0_best.pt + models/history.json na Drive.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

# pozwala zaimportowac prepare_data niezaleznie od katalogu uruchomienia
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import build_transforms, ArtifactDataset


def parse_args():
    p = argparse.ArgumentParser(description="Trening EfficientNet-B0 (Real vs AI)")
    p.add_argument("--data-root", default="/content/drive/MyDrive/ai-image-detector")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=8,
                   help="Fine-tuning zbiega szybko; early stopping i tak przerwie wczesniej")
    p.add_argument("--batch-size", type=int, default=64, help="64 miesci sie na T4; na wiekszym GPU zwieksz")
    p.add_argument("--lr", type=float, default=3e-4, help="AdamW; dobry default do fine-tuningu CNN")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Regularyzacja L2")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=3, help="Early stopping: ile epok bez poprawy val-loss")
    p.add_argument("--cache-local", action="store_true",
                   help="Skopiuj obrazy z Drive na lokalny dysk Colaba (duzo szybsze epoki)")
    p.add_argument("--limit", type=int, default=0,
                   help="Szybki test: ogranicz liczbe obrazow na split (np. 4000). 0 = caly zbior")
    # --- przelaczniki ablacyjne (badanie wplywu czynnikow na jakosc) ---
    p.add_argument("--no-aug", action="store_true",
                   help="ABLACJA: wylacz augmentacje (trening na transform jak walidacja)")
    p.add_argument("--no-pretrained", action="store_true",
                   help="ABLACJA: trenuj od zera, bez wag ImageNet (bez uczenia transferowego)")
    p.add_argument("--tag", default="",
                   help="Sufiks nazw plikow wynikowych, by oddzielic biegi ablacyjne (np. _noaug)")
    p.add_argument("--out-subdir", default="models")
    return p.parse_args()


CACHE_SUBDIRS = ["cocoai_extracted", "openfake_extracted", "artifact_raw"]

def maybe_cache_local(df, data_root, enable):
    """Drive (FUSE) jest bardzo wolny przy odczycie dziesiatek tysiecy malych plikow.
    Strategia dwupoziomowa:
      - Jesli na Drive istnieje archiwum aidata_images.tar -> kopiujemy JEDEN duzy plik
        (szybkie, minuty) i rozpakowujemy lokalnie.
      - Jesli nie -> jednorazowo kopiujemy pliki pojedynczo (wolne, godziny), po czym
        budujemy tar z kopii lokalnej i odkladamy na Drive, zeby KazDA kolejna sesja
        (restart po rozlaczeniu, ablacje, eval, ViT) startowala w minuty.
    Przepisujemy sciezki w manifescie na lokalne."""
    if not enable:
        return df
    import shutil
    import subprocess
    local = "/content/aidata"
    archive = os.path.join(data_root, "aidata_images.tar")
    if not os.path.isdir(local):
        os.makedirs(local, exist_ok=True)
        if os.path.exists(archive):
            print("[cache] znaleziono aidata_images.tar — kopiuje archiwum z Drive (minuty)...")
            tmp = "/content/aidata_images.tar"
            shutil.copyfile(archive, tmp)
            print("[cache] rozpakowuje lokalnie...")
            subprocess.run(["tar", "-xf", tmp, "-C", local], check=True)
            os.remove(tmp)
        else:
            for sub in CACHE_SUBDIRS:
                src, dst = os.path.join(data_root, sub), os.path.join(local, sub)
                if os.path.isdir(src) and not os.path.isdir(dst):
                    print(f"[cache] kopiuje {sub} -> dysk lokalny (jednorazowo, WOLNE)...")
                    shutil.copytree(src, dst)
            present = [s for s in CACHE_SUBDIRS if os.path.isdir(os.path.join(local, s))]
            print("[cache] buduje aidata_images.tar z kopii lokalnej i odkladam na Drive "
                  "(jednorazowo; przyspieszy kazda kolejna sesje)...")
            tmp = "/content/aidata_images.tar"
            subprocess.run(["tar", "-cf", tmp, "-C", local] + present, check=True)
            shutil.copyfile(tmp, archive)
            os.remove(tmp)
            print(f"[cache] archiwum zapisane: {archive}")
    df = df.copy()
    df["path"] = df["path"].str.replace(data_root, local, regex=False)
    return df


def make_loader(manifest, transform, data_root, cache, limit, batch_size, workers, shuffle):
    df = pd.read_csv(manifest)
    df = maybe_cache_local(df, data_root, cache)       # ew. przepisanie sciezek na lokalne
    if limit and limit > 0:                            # szybki test na wycinku
        df = df.sample(min(limit, len(df)), random_state=42).reset_index(drop=True)
    tmp = manifest + ".run.csv"                         # ArtifactDataset czyta z CSV
    df.to_csv(tmp, index=False)
    ds = ArtifactDataset(tmp, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=True), len(ds)


def build_model(device, pretrained=True):
    # EfficientNet-B0; pretrained=True -> wagi ImageNet (uczenie transferowe), False -> od zera (ablacja)
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    # podmiana glowicy: oryginalnie 1000 klas ImageNet -> 2 klasy (Real=0, AI=1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model.to(device)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """Jedna epoka. Gdy optimizer=None -> tryb ewaluacji (bez gradientow)."""
    train = optimizer is not None
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad()
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            out = model(x)
            loss = criterion(out, y)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, correct / total


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device={device} | torch={torch.__version__}")
    if device == "cpu":
        print("[ostrzezenie] Brak GPU! Wlacz: Runtime -> Change runtime type -> T4 GPU.")

    man = os.path.join(args.data_root, "manifests")
    out_dir = os.path.join(args.data_root, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.img_size)
    if args.no_aug:                         # ablacja: trening bez augmentacji
        train_tf = eval_tf
    print(f"[konfig] augmentacja={'NIE' if args.no_aug else 'TAK'} | "
          f"pretrained(ImageNet)={'NIE' if args.no_pretrained else 'TAK'} | tag='{args.tag}'")
    train_dl, n_tr = make_loader(os.path.join(man, "train.csv"), train_tf, args.data_root,
                                 args.cache_local, args.limit, args.batch_size, args.num_workers, True)
    val_dl, n_va = make_loader(os.path.join(man, "val.csv"), eval_tf, args.data_root,
                               args.cache_local, args.limit, args.batch_size, args.num_workers, False)
    print(f"[dane] train={n_tr}  val={n_va}  | batch={args.batch_size}")

    model = build_model(device, pretrained=not args.no_pretrained)
    criterion = nn.CrossEntropyLoss()                 # 2 wyjscia + softmax; klasy 50/50 -> bez wag
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    history, best_val, no_improve = [], float("inf"), 0
    best_path = os.path.join(out_dir, f"efficientnet_b0_best{args.tag}.pt")
    last_path = os.path.join(out_dir, f"last{args.tag}.pt")
    hist_path = os.path.join(out_dir, f"history{args.tag}.json")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        dt = time.time() - t0
        print(f"[epoka {epoch}/{args.epochs}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} | {dt:.0f}s")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        json.dump(history, open(hist_path, "w"), indent=2)

        # checkpoint na Drive co epoke (zabezpieczenie przed rozlaczeniem Colaba)
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "val_acc": va_acc, "classes": {0: "real", 1: "ai"}}, last_path)

        # zapis najlepszego wg val-loss + early stopping
        if va_loss < best_val:
            best_val, no_improve = va_loss, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_acc": va_acc, "classes": {0: "real", 1: "ai"}}, best_path)
            print(f"   -> nowy najlepszy model (val_loss={va_loss:.4f}) zapisany: {best_path}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[early stopping] brak poprawy od {args.patience} epok — przerywam.")
                break

    print(f"\nGotowe. Najlepszy model: {best_path} | historia: {hist_path}")


if __name__ == "__main__":
    main()
