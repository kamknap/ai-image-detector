# Detektor obrazów generowanych przez AI — metodologia i postęp prac

**Autor:** Kamil Knapik
**Stan na:** czerwiec 2026
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
| train | 103 202 | 51 600 / 51 602 | 51 |
| val | 12 901 | 6 451 / 6 450 | 51 |
| test (in-distribution) | 12 901 | 6 451 / 6 450 | 51 |
| **test_heldout (Imagen)** | 1 820 | 910 / 910 | 1 (Imagen) + real |

Pula przed balansem: 159 995 obrazów (68 000 real, 91 995 fake). Po zasadach z 3.3 limiterem była klasa fake (64 502 po capie), stąd ~64,5 tys./klasę.

> Uwaga: ArtiFact nie został jeszcze włączony (bieg modern-only). Pełny zbiór z ArtiFact + większym podzbiorem OpenFake planowany jest na finalne wyniki.

---

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
- Held-out (Imagen) jest niewielki (910 fake) przy obecnym podzbiorze OpenFake — przy pełnym biegu będzie większy.
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
