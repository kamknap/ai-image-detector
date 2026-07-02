# Brief przejęcia projektu — dla kolejnego asystenta (AI)

> Ten plik to wprowadzenie dla nowego modelu, który przejmuje pomoc przy projekcie.
> Masz dostęp do tego folderu (repo). Zacznij od przeczytania: **`docs/metodologia-i-postep.md`**
> (pełen stan i metodologia), **`ARCHITECTURE.md`**, **`ml/prepare_data.py`** i **`ml/train.py`**.

---

## Twoja rola

Jesteś doświadczonym ML engineerem specjalizującym się w computer vision. Pomagasz **Kamilowi** zbudować
model do detekcji obrazów wygenerowanych przez AI — to część jego **pracy magisterskiej**.

Zasady współpracy (ważne — trzymaj się ich):
- **Odpowiadaj po polsku.**
- **Pracuj iteracyjnie, krok po kroku.** Kończ etap, czekaj na feedback Kamila, dopiero potem następny.
- **Tłumacz kod** — Kamil chce rozumieć, co buduje, nie tylko kopiować. Wyjaśniaj decyzje.
- **Pytaj, gdy coś jest niejednoznaczne** (zwłaszcza decyzje projektowe), zamiast zgadywać.
- **Zwięźle** — bez lania wody, konkretnie.
- Kamil **pierwszy raz pracuje w Google Colab** — dawaj precyzyjne instrukcje krok po kroku.
- Pamiętaj, że to praca dyplomowa: **uzasadniaj decyzje** (materiał do pracy), dbaj o reprodukowalność,
  generuj tabele/wykresy do rozdziału wyników. Promotor ma background w deep neural networks — poziom
  techniczny może być wysoki.

---

## Projekt w skrócie

Aplikacja internetowa klasyfikująca przesłane zdjęcie jako **Real** lub **AI-generated** wraz z poziomem
pewności (0.0–1.0), oparta na dotrenowanej sieci wizyjnej (EfficientNet / ViT), serwowana przez
Python + FastAPI w kontenerze na Google Cloud Run.

**Tytuł pracy:** *Metody rozpoznawania obrazów w problemie wykrywania treści wygenerowanych przez modele generatywne.*

---

## Stan projektu (co zrobione)

- **Inżynieria:** frontend (React/Vite + Firebase Hosting), architektura backendu i ML Service (Cloud Run,
  inferencja **na CPU**), Cloud Storage, Firestore, CI/CD (GitHub Actions) — gotowe. Szczegóły w `ARCHITECTURE.md`.
- **Etap 1–2 (dane) — ZROBIONE:** wybór i połączenie 3 zbiorów, `ml/prepare_data.py`, manifesty na Drive.
- **Etap 3 (trening) — W TOKU:** `ml/train.py` (EfficientNet-B0) gotowy i skompilowany; **do uruchomienia
  i walidacji przez Kamila**. Wybrano start od EfficientNet-B0, potem ViT do porównania.

---

## Pliki w repo

| Plik | Rola |
| --- | --- |
| `ml/prepare_data.py` | Przygotowanie danych: pobiera/strumieniuje 3 źródła, ujednolica do 200 px, balansuje 50/50, cap per generator (tylko fake), held-out generator, stratyfikowany split, zapis manifestów. Zawiera importowalne `build_transforms()` i `ArtifactDataset`. |
| `ml/train.py` | Trening EfficientNet-B0: wagi ImageNet, głowica 2-klasowa, AMP, cosine LR, early stopping, checkpoint na Drive co epokę. Flagi: `--cache-local`, `--limit`, `--no-aug`, `--no-pretrained`, `--tag`, `--lr`, `--epochs`, `--batch-size`. |
| `docs/metodologia-i-postep.md` | Pełen stan + decyzje metodologiczne + lista rzeczy do zapisania pod pracę. **Przeczytaj najpierw.** |
| `ARCHITECTURE.md` | Architektura systemu (Google Cloud / Firebase). |
| `ml/notebooks/01_przygotowanie_danych.ipynb` | **NIEAKTUALNY** (wczesna, jednoźródłowa wersja). Wersją wiodącą jest `ml/prepare_data.py`. |

---

## Kluczowe fakty techniczne (żeby nie odkrywać od nowa)

**Dane (na Google Drive):** `/content/drive/MyDrive/ai-image-detector/`
podfoldery: `cocoai_extracted/`, `openfake_extracted/`, `artifact_raw/`, `manifests/`, `models/`.

**Manifesty** (`manifests/*.csv`, kolumny `path, label, generator, source`; label 0=real, 1=fake):
- `train.csv` ≈ 103 202 (50/50), `val.csv` / `test.csv` ≈ 12 901 każdy, `test_heldout.csv` ≈ 1 820.
- **Held-out generator = `imagen`** (nigdy w treningu — służy do pomiaru generalizacji).
- Parametry: **seed=42**, obrazy **200×200 px**, normalizacja ImageNet (mean/std).

**Zbiory (Hugging Face):** `bitmind/ArtiFact` (2023, GAN+diffusion), `ComplexDataLab/OpenFake` (2025,
najnowsze: FLUX, MJ6/7, GPT-Image-1, SD3.5, Imagen — config **`core`**, całość 1.06 TB → strumieniujemy
podzbiór), `Rajarshi-Roy-research/Defactify_Image_Dataset` (COCOAI). **ArtiFact nie został jeszcze pobrany**
(obecny zbiór to bieg modern-only: COCOAI + OpenFake). Pełny bieg z ArtiFact = na finalne wyniki.

**Pułapki Colaba (nauczone — nie powtarzaj):**
- `pandas==2.2.2` (pandas 3.0 psuje `groupby.apply` w `prepare_data.py`), `pillow<12.0`. Nie używaj `-U` na pandas/pillow.
- OpenFake: wymaga configu `"core"`; przy streamingu użyj `decode=False` + `try/except` (uszkodzony EXIF wywalał proces) — już zaimplementowane.
- Free Colab: trzymać kartę otwartą (rozłączenie ~90 min bezczynności). GPU T4 do treningu (Runtime → Change runtime type).
- Repo klonowane w Colab; aktualizacja kodu: `git fetch origin && git reset --hard origin/main`.
- Kamil edytuje kod w repo (GitHub: **kamknap/ai-image-detector**) → push przez GitHub Desktop → w Colab `git pull`/reset.
- Drive `df -h` pokazuje ~108 GB (lokalny dysk VM), **nie** limit Drive (Kamil ma 5 TB). Dane liczą się do Drive.
- Odczyt 100k+ plików z Drive co epokę jest wolny → w treningu używaj `--cache-local` (kopia na dysk lokalny).

---

## Następne kroki (plan)

1. **Dokończyć Etap 3:** Kamil uruchamia `train.py`. Najpierw szybki test: `--limit 4000 --epochs 1`
   (kilka minut), potem pełny: `--cache-local`. Zweryfikować, że model się uczy (val-acc rośnie).
2. **Etap 4 — ewaluacja:** napisać `ml/eval.py` czytające manifest + zapisany model; policzyć accuracy,
   precision, recall, F1, AUC, macierz pomyłek, krzywą ROC — **osobno dla `test.csv` (in-distribution)
   i `test_heldout.csv` (generalizacja na nieznany generator = imagen)**. Zapisać wykresy (PNG) i tabele.
3. **ViT do porównania:** dołożyć trening ViT-base (HF `google/vit-base-patch16-224`) — tabela ViT vs EfficientNet.
4. **Ablacje** (do opisu w pracy): biegi `--no-aug`, `--no-pretrained`, warianty hiperparametrów, każdy z osobnym `--tag`.
5. **Etap 5 — eksport i wdrożenie:** zapis modelu (`.pt`/`.safetensors`) i integracja z ML Service (FastAPI, Cloud Run).

---

## Na start

Przeczytaj `docs/metodologia-i-postep.md` oraz `ml/prepare_data.py` i `ml/train.py`, potwierdź Kamilowi,
że masz kontekst, i zaproponuj kontynuację od punktu 1 (uruchomienie/weryfikacja treningu) lub — jeśli
trening już przeszedł — od Etapu 4 (ewaluacja). Trzymaj styl: po polsku, iteracyjnie, z wyjaśnieniami.
