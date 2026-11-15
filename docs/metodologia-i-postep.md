# Detektor obrazów generowanych przez AI — metodologia i postęp prac

**Autor:** Kamil Knapik
**Stan na:** listopad 2026
**Tytuł pracy:** *Metody rozpoznawania obrazów w problemie wykrywania treści wygenerowanych przez modele generatywne*

Dokument podsumowuje stan realizacji i decyzje metodologiczne. Sekcja 6 to lista rzeczy, które warto zapisywać na bieżąco, aby ułatwić pisanie pracy (reprodukowalność + materiał na rozdziały o danych, metodzie i wynikach).

---

## 1. Cel i zakres

Celem jest binarna klasyfikacja obrazów: **Real (0)** vs **AI-generated (1)**, wraz z miarą pewności (confidence score 0.0–1.0). Rozwiązanie ma działać jako aplikacja webowa: użytkownik przesyła zdjęcie, otrzymuje werdykt i poziom pewności.

Zakres techniczny obejmuje: zbudowanie i ujednolicenie zbioru danych z wielu źródeł, dotrenowanie sieci wizyjnej (EfficientNet i/lub Vision Transformer), ewaluację (w tym generalizację na nieznane generatory) oraz wdrożenie modelu jako usługi.

---

## 2. Architektura systemu (zrealizowane inżynieryjnie)

System oparto o ekosystem Google Cloud / Firebase:

| Warstwa | Technologia | Status |
| --- | --- | --- |
| Frontend | React (Vite) + Firebase Hosting | zaimplementowane |
| Backend API | Node.js/Express na Cloud Run (kontener) | architektura gotowa |
| ML Service | Python + FastAPI na Cloud Run (kontener, **inferencja na CPU**) | model w toku (Etap 3) |
| Przechowywanie obrazów | Cloud Storage | gotowe |
| Logi predykcji | Firestore (timestamp, wynik, confidence, metadane) | gotowe |
| CI/CD | GitHub Actions + Artifact Registry | gotowe |

Istotny dla metody jest wybór **inferencji na CPU** ze scale-to-zero — to wpływa na dobór architektury modelu (rozmiar, czas cold startu).

---

## 3. Dane — metodologia (zrealizowane)

### 3.1 Wybór i charakterystyka zbiorów

Aby pokryć zarówno starsze, jak i obecnie dominujące generatory, połączono trzy publiczne, uzupełniające się zbiory:

| Zbiór | Rozmiar | Generatory | Realne (źródło) | Rok | Licencja |
| --- | --- | --- | --- | --- | --- |
| **ArtiFact** | ~2,5 mln | 25 (13 GAN, 7 diffusion, 5 inne) | ImageNet, COCO, FFHQ, CelebA-HQ, LSUN, AFHQ, MetFaces | 2023 | — |
| **OpenFake** | ~1,93 mln | FLUX, Midjourney 6/7, DALL·E 3, GPT-Image-1, Imagen 3/4, SD 3.5, Grok-2, Ideogram 3, HiDream, Recraft, Chroma | LAION-400M (treści newsowe/polityczne) | 2025 | CC-BY-SA-4.0 (subsety zamknięte: non-commercial) |
| **COCOAI / Defactify** | 96 tys. | SD 2.1, SDXL, SD 3, DALL·E 3, Midjourney 6 | COCO | 2025 | — |

Uzasadnienie doboru: ArtiFact daje różnorodność architektur starszej generacji (GAN-y, wczesne diffusion), OpenFake i COCOAI pokrywają najnowsze modele dominujące w realnym ruchu (FLUX, GPT-Image-1, SD3.5 itd.). Dzięki temu detektor ma szansę działać na obrazach spotykanych „w codzienności", a nie tylko na danych z jednej epoki.

### 3.2 Ujednolicenie domeny

Wszystkie obrazy sprowadzane są do **200×200 px** (rozdzielczość natywna ArtiFact, który dodatkowo stosuje kompresję JPEG i przeskalowania symulujące realne warunki — standard IEEE VIP Cup 2022). Wspólny format zapobiega temu, by model uczył się **rozdzielczości lub kompresji zamiast realności** — to częsta pułapka w detekcji AI.

Z każdego źródła budowany jest wspólny **manifest** o kolumnach: `path, label (0/1), generator, source`.

### 3.3 Balans klas i różnorodność generatorów

Przyjęto cztery zasady (do opisania w rozdziale metodologicznym):

1. **Balans klas 50/50** — eliminuje bias klasowy.
2. **Limit (cap) na generator — tylko dla klasy fake** (domyślnie 8000/rodzinę). Zapobiega dominacji pojedynczego generatora i wymusza różnorodność. Klasy real **nie** ograniczamy capem (wszystkie realne mają wspólną etykietę źródła, więc cap zlepiłby je w jeden kubełek — to był błąd wczesnej wersji, naprawiony).
3. **Normalizacja nazw generatorów do rodzin** (np. `sdxl-epic-realism` → `sdxl`, `flux.1-dev` → `flux`) — sensowne capowanie mimo dziesiątek wariantów LoRA/finetune.
4. **Held-out generator** — jeden generator (**Imagen**) całkowicie wykluczony z treningu, użyty wyłącznie w osobnym zbiorze testowym do pomiaru **generalizacji na nieznany generator**.

### 3.4 Podział danych

Stratyfikowany podział `train/val/test` (80/10/10) po kombinacji `(klasa × generator × źródło)`, co zapewnia proporcjonalną reprezentację w każdej części. Manifesty zapisywane są jako pliki CSV (reprodukowalność, niezależność treningu od kodu przygotowania danych). Ziarno losowości: **seed = 42**.

### 3.5 Aktualny zbiór (bieg COCOAI + OpenFake, bez ArtiFact)

| Część | Liczność | Balans (real/fake) | Generatory |
| --- | --- | --- | --- |
| train | 102 744 | 51 372 / 51 372 | 50 |
| val | 12 844 | 6 422 / 6 422 | 50 |
| test (in-distribution) | 12 844 | 6 422 / 6 422 | 50 |
| **test_heldout (Imagen)** | 1 826 | 913 / 913 | 1 (Imagen) + real |

Pula przed balansem: 159 774 obrazów (68 000 real, 91 774 fake; COCOAI 48 000 + OpenFake 111 774 ze skanu 120 000 wierszy). Po zasadach z 3.3 limiterem była klasa fake (64 216 po capie), stąd ~64,2 tys./klasę.

Budowa zbioru: 14.11.2026, Colab (Python 3.13, datasets 4.8.5, pandas 2.2.3, Pillow 11.3), seed=42. Rewizja OpenFake (commit HF): `00b2f0ae3f5f7684c4b00f794f4c500076a9355d` — wcześniejszy bieg (czerwiec, rewizja `89dc06b2…`) dawał 103 202 / 12 901 / 12 901 / 1 820; różnice wynikają z aktualizacji zbioru na HF. Pełna migawka (obrazy + manifesty + `dataset_info.json` + `per_generator_counts.csv`): `aidata_images.tar` (3,25 GB) na Drive.

> Uwaga: ArtiFact nie został jeszcze włączony (bieg modern-only). Pełny zbiór z ArtiFact + większym podzbiorem OpenFake planowany jest na finalne wyniki.

---

## 3a. Wyniki — EfficientNet-B0 (bieg 14–15.11.2026)

### Konfiguracja treningu

EfficientNet-B0 (4,01 mln parametrów, wagi ImageNet `IMAGENET1K_V1`), głowica 2-klasowa, wejście **200×200** (natywna
rozdzielczość zbioru — bez interpolacji), AdamW (lr 3e-4, weight decay 1e-4), CosineAnnealingLR (T_max = 8), batch 64,
AMP, CrossEntropyLoss, augmentacja: RandomResizedCrop (0,85–1,0), flip poziomy, lekki ColorJitter (0,1). Early stopping
(patience 3) i **wybór checkpointu wg minimalnej straty walidacyjnej** (kryterium ustalone przed ewaluacją na teście).
Sprzęt: Google Colab, Tesla T4, 2 vCPU; czas: 8 epok, **72,3 min** (~9 min/epokę). Seed = 42.

### Przebieg uczenia (`models/learning_curves.png`)

Najlepsza epoka wg val_loss: **5** (val_loss 0,1997, val_acc 92,65%). Od epoki 5 strata treningowa dalej spada
(0,107 → 0,044), a walidacyjna lekko rośnie (→ 0,213), przy wciąż rosnącej val_acc (→ 93,69% w epoce 8) — model
poprawia decyzje, ale staje się nadmiernie pewny siebie na błędach (pogorszenie kalibracji). Luka train/val
(98% vs 94%) wskazuje na przeuczenie — materiał do dyskusji o regularyzacji / ablacjach.

### Metryki (klasa pozytywna = AI, próg 0,5)

| Checkpoint | Zbiór | Acc | Precision | Recall | F1 | AUC | ECE | Log-loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **epoka 5 (wybrany)** | test (in-distribution, n=12 844) | 92,71% | 92,77% | 92,65% | 92,71% | 0,979 | 0,024 | 0,203 |
| **epoka 5 (wybrany)** | test_heldout — Imagen (n=1 826) | 89,05% | 91,50% | 86,09% | 88,71% | 0,958 | 0,050 | 0,318 |
| epoka 8 (ostatni) | test | 93,80% | 94,10% | 93,46% | 93,78% | 0,982 | 0,034 | 0,224 |
| epoka 8 (ostatni) | test_heldout — Imagen | 89,92% | 93,44% | 85,87% | 89,50% | 0,964 | 0,064 | 0,363 |

ECE (expected calibration error, 15 przedziałów), log-loss i 95% przedziały ufności (bootstrap, 1000 prób) liczy `ml/eval.py` (`summary.csv`, `calibration_*.png`). 95% CI dla epoki 5: test acc [92,25; 93,16]%, AUC [0,977; 0,981]; held-out acc [87,68; 90,36]%, recall [83,85; 88,21]%, AUC [0,949; 0,965]. Przedziały testu i held-out się nie nakładają — spadek na nieznanym generatorze jest istotny. Diagram kalibracji: słupki systematycznie poniżej przekątnej w przedziale pewności 0,6–0,95 → model jest nadmiernie pewny siebie (kandydat: temperature scaling na zbiorze walidacyjnym).
Epoka 8 ma ~1 pp wyższą trafność, ale gorszą kalibrację (ECE, log-loss; 80% vs 66% predykcji z pewnością > 0,99).
Ponieważ aplikacja zwraca **poziom pewności**, kalibracja jest istotna — co uzasadnia kryterium val_loss.
Wynik epoki 8 raportować jako analizę uzupełniającą (nie wybierać checkpointu na podstawie testu).

### Generalizacja na nieznany generator (Imagen)

- Spadek wykrywalności fake: recall **92,65% → 86,09%** (−6,6 pp), AUC 0,979 → 0,958. Skuteczność na realnych w
  held-out (~92%) jest jak w teście — spadek dotyczy wyłącznie wykrywania obrazów z nieznanego generatora.
- Rozkład P(AI) dla Imagen jest dwumodalny: mediana 0,994 (większość wykrywana pewnie), ~14% przeoczeń.
- **Zastrzeżenie:** Imagen pochodzi z tego samego zbioru (OpenFake: te same prompty, realne z LAION, ten sam
  pipeline) — to test generalizacji **między generatorami**, nie między zbiorami/domenami. Test cross-dataset
  (inne źródło realnych, kompresja z social mediów) prawdopodobnie da większy spadek.

### Analiza per źródło i per generator (test in-distribution, epoka 5)

| Źródło | Real (specyficzność) | Fake (recall) | AUC |
| --- | --- | --- | --- |
| COCOAI | 94,6% (n=755) | 96,4% (n=2 505) | 0,992 |
| OpenFake | 92,5% (n=5 667) | 90,2% (n=3 917) | 0,971 |

Najtrudniejsze generatory (n ≥ 100): OpenFake FLUX 86,0%, HiDream 86,4%, Midjourney 86,7%, Ideogram 89,5%,
SDXL 90,0%, GPT-Image 90,1%. Najłatwiejsze: COCOAI DALL·E 99,1%, SDXL 98,0%, SD3.5 97,6%. Ten sam generator jest
trudniejszy w OpenFake niż w COCOAI (np. Midjourney 86,7% vs 91,9%) — OpenFake ma bardziej zróżnicowane treści
i nowsze wersje modeli. Pełna tabela: `models/eval/per_generator_test.csv`.

## 4. Decyzje metodologiczne — argumentacja do pracy

Poniższe wybory warto wprost uzasadnić w rozdziale o metodzie, bo bezpośrednio wpływają na wiarygodność wyników:

- **Ujednolicenie rozdzielczości i kompresji** — by mierzyć detekcję *treści generowanej*, nie artefaktów rozdzielczości.
- **Cap tylko na generatory fake** — różnorodność generatorów bez dławienia klasy real.
- **Held-out generator** — uczciwy pomiar generalizacji; pokazuje, czy model uogólnia, czy tylko zapamiętuje „odcisk" znanych generatorów.
- **Parowanie real/fake w obrębie źródła** — w każdym zbiorze realne i syntetyczne mają zbliżoną treść (COCO↔COCOAI, LAION↔OpenFake), co ogranicza ryzyko, że model uczy się „stylu źródła" zamiast realności. **To do zweryfikowania w ewaluacji (Etap 4).**
- **Umiarkowana augmentacja** (lekki crop, flip, delikatny color jitter) — agresywna augmentacja (silny blur/kompresja) potrafi zatrzeć wysokoczęstotliwościowe artefakty generatorów, na których opiera się detekcja.

---

## 5. Ograniczenia (do uczciwego opisania)

- Generatory ArtiFact pochodzą z 2023 r. — bez najnowszych modeli; dlatego dołożono OpenFake/COCOAI.
- Realne obrazy pochodzą głównie z COCO i LAION — ograniczona różnorodność „dziedziny realnej".
- Held-out (Imagen) jest niewielki (913 fake) przy obecnym podzbiorze OpenFake — przy pełnym biegu będzie większy.
- OpenFake: część obrazów z modeli zamkniętych objęta licencją non-commercial (dopuszczalne dla pracy dyplomowej; do odnotowania).

---

## 6. Co zapisywać teraz — pod pisanie pracy

### 6.1 Reprodukowalność (zapisać i wersjonować)

- **Nazwy i wersje zbiorów** z Hugging Face: `bitmind/ArtiFact`, `ComplexDataLab/OpenFake` (konfiguracja `core`), `Rajarshi-Roy-research/Defactify_Image_Dataset` — wraz z **datą pobrania** (zbiory bywają aktualizowane).
- **Manifesty CSV** (`train/val/test/test_heldout.csv`) — to dokładny zapis, które obrazy trafiły gdzie. Zarchiwizować.
- **Parametry przygotowania danych:** seed=42, rozmiar 200 px, cap 8000/generator (fake), held-out=Imagen, normalizacja ImageNet (mean/std).
- **Rozkład per-generator** (liczność każdego z 51 generatorów) — gotowy materiał na tabelę w aneksie.
- **Skrypt `ml/prepare_data.py`** w repo (z historią commitów) — pełny, odtwarzalny pipeline.

### 6.2 Materiał na rozdziały pracy

- **Rozdział „Dane":** tabele z sekcji 3.1 i 3.5, opis ujednolicenia i balansu (3.2–3.4).
- **Rozdział „Metoda":** argumentacja z sekcji 4 + wybór architektury (Etap 3).
- **Rozdział „Wyniki":** (powstanie w Etapie 4) accuracy, precision, recall, F1, macierz pomyłek, krzywa ROC/AUC — osobno dla testu in-distribution i held-out.
- **Rozdział „Ograniczenia/dyskusja":** sekcja 5 + obserwacje z ewaluacji.

### 6.3 Do zapisania w kolejnych etapach (przypomnienie)

- Architektura modelu i liczba parametrów, źródło wag pre-trenowanych (HF).
- Hiperparametry: learning rate, batch size, liczba epok, optymalizator, scheduler, early stopping.
- Krzywe uczenia (loss/accuracy train vs val) — zrzuty wykresów.
- Metryki końcowe + wykresy (macierz pomyłek, ROC) — jako pliki PNG/PDF do wstawienia.
- Wagi wytrenowanego modelu (`.pt`/`.safetensors`) — zarchiwizować.
- Czas treningu i sprzęt (Colab GPU) — do sekcji o środowisku.

---

## 7. Następne kroki

1. **Etap 3 — model i trening:** start od **EfficientNet-B0** (lekki, szybki na CPU — zgodny z wdrożeniem na Cloud Run), dotrenowanie z wag ImageNet, podmiana głowicy na binarną, pętla treningowa + walidacja. Następnie **ViT-base** do porównania (tabela ViT vs EfficientNet — mocny rozdział wyników).
2. **Etap 4 — ewaluacja:** metryki + macierz pomyłek + ROC, osobno in-distribution i held-out (generalizacja).
3. **Etap 5 — eksport:** zapis modelu do `.pt`/`.safetensors` i integracja z ML Service (FastAPI na Cloud Run).
