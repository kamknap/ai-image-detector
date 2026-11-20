"""
Etap 3 — Trening klasyfikatora Real vs AI-generated: EfficientNet-B0 (torchvision) lub ViT (timm).

Co robi:
  1. Rozpakowuje archiwum aidata_images.tar z Drive na dysk lokalny Colaba (jeden duzy plik
     zamiast ~100k malych — minuty zamiast godzin) i czyta manifesty z tej samej migawki.
  2. Laduje siec z wagami pre-trenowanymi (transfer learning) i podmienia glowice na 2-klasowa:
       --arch efficientnet_b0 (domyslnie; 4,0 M param., wagi ImageNet-1k)
       --arch vit_small       (ViT-S/16, 22 M param., timm augreg: ImageNet-21k -> 1k; wymaga: pip install timm)
       --arch vit_base        (ViT-B/16, 86 M param., timm augreg2)
  3. Trenuje z walidacja, mixed precision (AMP), cosine LR, early stopping.
  4. Po KAZDEJ epoce zapisuje pelny checkpoint na Drive (model + optymalizator + scheduler + historia),
     wiec po rozlaczeniu Colaba mozna wznowic trening flaga --resume.
  5. Zapisuje najlepszy model, historie uczenia i konfiguracje biegu (do pracy).

Reprodukowalnosc: --seed ustawia ziarna Pythona/NumPy/PyTorch, a kolejnosc danych i augmentacja
w kazdej epoce wynikaja z (seed + numer epoki) — wznowiony bieg widzi te same dane co nieprzerwany.
(cuDNN w trybie benchmark nie jest w 100% deterministyczny — roznice rzedu szumu numerycznego.)

Uzycie (Colab, GPU: Runtime -> Change runtime type -> T4 GPU):
    from google.colab import drive; drive.mount('/content/drive')
    %cd /content/ai-image-detector
    !pip install -q datasets   # bez pinow pandas/pillow (Colab ma Python 3.13)
    !python ml/train.py --limit 4000 --epochs 1 --tag _smoke   # szybki test
    !python ml/train.py                                         # pelny bieg
    !python ml/train.py --resume                                # wznowienie po rozlaczeniu
    !python ml/train.py --arch vit_small --img-size 224 --lr 1e-4 --weight-decay 0.05 --warmup-epochs 1 --tag _vits

Wynik (na Drive, models/): <arch>_best<tag>.pt, last<tag>.pt, history<tag>.json, config<tag>.json
(dla architektury innej niz efficientnet_b0 bez --tag ustawiany jest automatycznie tag _<arch>,
 zeby nie nadpisac plikow EfficientNetu)
"""
import os
import sys
import json
import time
import random
import argparse
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

# pozwala zaimportowac prepare_data niezaleznie od katalogu uruchomienia
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_data import build_transforms, ArtifactDataset, ARCHIVE_NAME, IMAGENET_MEAN, IMAGENET_STD

# Rejestr architektur: nazwa w CLI -> identyfikator wag w timm (None = torchvision)
ARCHS = {
    "efficientnet_b0": None,
    "vit_small": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
}


def parse_args():
    p = argparse.ArgumentParser(description="Trening EfficientNet-B0 (Real vs AI)")
    p.add_argument("--data-root", default="/content/drive/MyDrive/ai-image-detector")
    p.add_argument("--arch", default="efficientnet_b0", choices=list(ARCHS))
    p.add_argument("--drop-path", type=float, default=0.1,
                   help="ViT: stochastic depth (regularyzacja; EfficientNet ma wlasne 0.2 wbudowane)")
    p.add_argument("--warmup-epochs", type=int, default=0,
                   help="Liniowy rozbieg LR przez N epok (zalecane 1 dla ViT; 0 = jak w biegu EfficientNet)")
    p.add_argument("--local-root", default="/content/aidata",
                   help="Gdzie rozpakowac archiwum z obrazami (szybki dysk lokalny Colaba)")
    p.add_argument("--no-cache", action="store_true",
                   help="Czytaj obrazy bezposrednio z --data-root zamiast z archiwum (WOLNE na Drive)")
    p.add_argument("--cache-local", action="store_true", help=argparse.SUPPRESS)  # zgodnosc wstecz (teraz domyslne)
    p.add_argument("--img-size", type=int, default=224,
                   help="Rozdzielczosc wejscia sieci. Obrazy w zbiorze maja 200 px; 200 = bez interpolacji")
    p.add_argument("--epochs", type=int, default=8,
                   help="Fine-tuning zbiega szybko; early stopping i tak przerwie wczesniej")
    p.add_argument("--batch-size", type=int, default=64, help="64 miesci sie na T4; na wiekszym GPU zwieksz")
    p.add_argument("--lr", type=float, default=3e-4, help="AdamW; dobry default do fine-tuningu CNN")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="Regularyzacja L2")
    p.add_argument("--num-workers", type=int, default=2, help="Colab free ma 2 vCPU")
    p.add_argument("--patience", type=int, default=3, help="Early stopping: ile epok bez poprawy val-loss")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true",
                   help="Wznow trening z models/last<tag>.pt (po rozlaczeniu Colaba)")
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


# --------------------------------------------------------------------------- #
# Dane: archiwum -> dysk lokalny
# --------------------------------------------------------------------------- #
def prepare_image_root(data_root, local_root, no_cache=False):
    """Zwraca katalog, wzgledem ktorego rozwiazujemy (wzgledne) sciezki z manifestow.

    Drive (FUSE) jest bardzo wolny przy odczycie dziesiatek tysiecy malych plikow (~5 obr./s),
    ale szybki przy sekwencyjnym czytaniu JEDNEGO duzego pliku. Dlatego trzymamy na Drive archiwum
    .tar i rozpakowujemy je raz na sesje. Znacznik .complete chroni przed uzyciem czesciowo
    rozpakowanych danych (np. gdy poprzednia proba zostala przerwana)."""
    if no_cache:
        return data_root
    marker = os.path.join(local_root, ".complete")
    if os.path.exists(marker):
        print(f"[dane] uzywam danych juz obecnych w {local_root}")
        return local_root
    archive = os.path.join(data_root, ARCHIVE_NAME)
    if not os.path.exists(archive):
        raise FileNotFoundError(
            f"Brak {archive}. Najpierw zbuduj zbior: "
            f"python ml/prepare_data.py --skip-artifact --work-dir {local_root}")
    os.makedirs(local_root, exist_ok=True)
    t0 = time.time()
    print(f"[dane] rozpakowuje {archive} ({os.path.getsize(archive) / 1e9:.2f} GB) -> {local_root} ...")
    subprocess.run(["tar", "-xf", archive, "-C", local_root], check=True)
    open(marker, "w").write("extracted\n")
    print(f"[dane] gotowe w {time.time() - t0:.0f}s")
    return local_root


def manifest_dir(image_root, data_root):
    """Manifesty z TEJ SAMEJ migawki co obrazy (z archiwum); w trybie --no-cache — z Drive."""
    local = os.path.join(image_root, "manifests")
    return local if os.path.isdir(local) else os.path.join(data_root, "manifests")


# --------------------------------------------------------------------------- #
# Reprodukowalnosc
# --------------------------------------------------------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True   # szybciej; kosztem pelnego determinizmu (patrz docstring)


def _seed_worker(worker_id):
    # kazdy worker DataLoadera dostaje ziarno pochodne od ziarna glownego -> powtarzalna augmentacja
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(manifest, transform, image_root, limit, batch_size, workers, shuffle, seed):
    df = pd.read_csv(manifest)
    if limit and limit > 0:                            # szybki test na wycinku
        df = df.sample(min(limit, len(df)), random_state=seed).reset_index(drop=True)
    ds = ArtifactDataset(df, transform, root=image_root)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=_seed_worker,
                      generator=g), len(ds)


# --------------------------------------------------------------------------- #
# Model i petla treningowa
# --------------------------------------------------------------------------- #
def build_model(device, pretrained=True, arch="efficientnet_b0", img_size=224, drop_path=0.1):
    """Zwraca (model, (mean, std)) — normalizacja wejscia zgodna z wagami pre-trenowanymi.
    pretrained=False -> losowa inicjalizacja (ablacja / ladowanie wlasnego checkpointu)."""
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        # podmiana glowicy: oryginalnie 1000 klas ImageNet -> 2 klasy (Real=0, AI=1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
        norm = (IMAGENET_MEAN, IMAGENET_STD)
    else:
        import timm
        if img_size % 16:
            raise ValueError(f"ViT dzieli obraz na fragmenty 16x16 px — img_size musi byc wielokrotnoscia 16 "
                             f"(np. 192, 208, 224), podano {img_size}")
        # num_classes=2 -> timm sam podmienia glowice; img_size != 224 -> interpolacja embeddingow pozycyjnych
        model = timm.create_model(ARCHS[arch], pretrained=pretrained, num_classes=2,
                                  img_size=img_size, drop_path_rate=drop_path)
        cfg = model.pretrained_cfg
        norm = (list(cfg["mean"]), list(cfg["std"]))     # augreg ViT: mean=std=0.5 (nie ImageNet!)
    return model.to(device), norm


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    """Jedna epoka. Gdy optimizer=None -> tryb ewaluacji (bez gradientow)."""
    train = optimizer is not None
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
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
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "-"
    print(f"[info] device={device} ({gpu}) | torch={torch.__version__} | seed={args.seed}")
    if device == "cpu":
        print("[ostrzezenie] Brak GPU! Wlacz: Runtime -> Change runtime type -> T4 GPU.")

    if args.arch != "efficientnet_b0" and not args.tag:
        args.tag = f"_{args.arch}"          # ochrona przed nadpisaniem plikow EfficientNetu
        print(f"[info] brak --tag -> ustawiam tag '{args.tag}'")
    out_dir = os.path.join(args.data_root, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"{args.arch}_best{args.tag}.pt")
    last_path = os.path.join(out_dir, f"last{args.tag}.pt")
    hist_path = os.path.join(out_dir, f"history{args.tag}.json")
    cfg_path = os.path.join(out_dir, f"config{args.tag}.json")

    image_root = prepare_image_root(args.data_root, args.local_root, args.no_cache)
    man = manifest_dir(image_root, args.data_root)

    model, (mean, std) = build_model(device, pretrained=not args.no_pretrained, arch=args.arch,
                                     img_size=args.img_size, drop_path=args.drop_path)
    n_params = sum(p.numel() for p in model.parameters())
    train_tf, eval_tf = build_transforms(args.img_size, mean, std)
    if args.no_aug:                         # ablacja: trening bez augmentacji
        train_tf = eval_tf
    print(f"[konfig] arch={args.arch} | augmentacja={'NIE' if args.no_aug else 'TAK'} | "
          f"pretrained(ImageNet)={'NIE' if args.no_pretrained else 'TAK'} | "
          f"img_size={args.img_size} | tag='{args.tag}'")
    train_dl, n_tr = make_loader(os.path.join(man, "train.csv"), train_tf, image_root, args.limit,
                                 args.batch_size, args.num_workers, True, args.seed)
    val_dl, n_va = make_loader(os.path.join(man, "val.csv"), eval_tf, image_root, args.limit,
                               args.batch_size, args.num_workers, False, args.seed)
    print(f"[dane] train={n_tr}  val={n_va}  | batch={args.batch_size} | manifesty: {man}")

    criterion = nn.CrossEntropyLoss()                 # 2 wyjscia + softmax; klasy 50/50 -> bez wag
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup_epochs > 0:
        # epoki 1..W: LR rosnie liniowo od 10% do 100%; potem cosinus do ~0.
        # Rozbieg stabilizuje fine-tuning transformerow (duze, losowe gradienty nowej glowicy na starcie).
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)],
            milestones=[args.warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    history, best_val, no_improve, start_epoch = [], float("inf"), 0, 1
    if args.resume:
        if not os.path.exists(last_path):
            raise FileNotFoundError(f"--resume: brak {last_path}")
        ck = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optimizer_state"])
        scheduler.load_state_dict(ck["scheduler_state"])
        scaler.load_state_dict(ck["scaler_state"])
        history, best_val, no_improve = ck["history"], ck["best_val"], ck["no_improve"]
        start_epoch = ck["epoch"] + 1
        print(f"[resume] wznawiam od epoki {start_epoch} (najlepszy val_loss={best_val:.4f})")
        if no_improve >= args.patience or start_epoch > args.epochs:
            print("[resume] trening byl juz zakonczony — nic do zrobienia.")
            return

    config = {"args": vars(args), "model": args.arch, "weights": ARCHS[args.arch] or "torchvision IMAGENET1K_V1",
              "norm_mean": mean, "norm_std": std, "n_params": n_params,
              "device": gpu, "torch": torch.__version__, "n_train": n_tr, "n_val": n_va,
              "optimizer": "AdamW", "loss": "CrossEntropyLoss",
              "scheduler": f"warmup {args.warmup_epochs} ep + CosineAnnealingLR" if args.warmup_epochs
                           else "CosineAnnealingLR",
              "amp": device == "cuda", "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(config, open(cfg_path, "w"), indent=2)
    print(f"[model] {args.arch} | parametry: {n_params / 1e6:.2f} M | normalizacja mean={mean} std={std} "
          f"| konfiguracja: {cfg_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        # ziarno zalezne od epoki -> kolejnosc danych i augmentacja identyczne po --resume
        torch.manual_seed(args.seed + epoch)
        train_dl.generator.manual_seed(args.seed + epoch)
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        scheduler.step()
        dt = time.time() - t0
        print(f"[epoka {epoch}/{args.epochs}] train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} | lr={lr:.2e} | {dt:.0f}s")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "lr": lr, "time_s": dt})
        json.dump(history, open(hist_path, "w"), indent=2)

        # zapis najlepszego wg val-loss (lekki: tylko wagi + metadane — do ewaluacji i wdrozenia)
        improved = va_loss < best_val
        if improved:
            best_val, no_improve = va_loss, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": va_acc,
                        "val_loss": va_loss, "img_size": args.img_size, "arch": args.arch,
                        "norm_mean": mean, "norm_std": std,
                        "classes": {0: "real", 1: "ai"}}, best_path)
            print(f"   -> nowy najlepszy model (val_loss={va_loss:.4f}) zapisany: {best_path}")
        else:
            no_improve += 1

        # pelny checkpoint co epoke (do --resume po rozlaczeniu Colaba)
        torch.save({"epoch": epoch, "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(), "history": history,
                    "best_val": best_val, "no_improve": no_improve, "img_size": args.img_size,
                    "arch": args.arch, "norm_mean": mean, "norm_std": std}, last_path)

        if not improved and no_improve >= args.patience:
            print(f"[early stopping] brak poprawy od {args.patience} epok — przerywam.")
            break

    print(f"\nGotowe. Najlepszy model: {best_path} | historia: {hist_path}")


if __name__ == "__main__":
    main()
