# ARCHITEKTURA — AI Image Detector (Google Cloud / Firebase)

> Wersja oparta o ekosystem Google. Powód migracji z Azure: subskrypcja
> **Azure for Students** ma systemową politykę regionów, która blokuje tworzenie
> Static Web Apps (`RequestDisallowedByAzure`). Cały stack przenosi się na
> Google Cloud / Firebase niemal 1:1 — częściowo prościej i taniej.

## Mapowanie usług Azure → Google Cloud

| Azure (plan pierwotny) | Google Cloud / Firebase | Rola |
| --- | --- | --- |
| Static Web Apps | Firebase Hosting | hosting frontendu |
| Container Apps | Cloud Run | backend API (kontener) |
| Container Apps | Cloud Run | ML Service (kontener) |
| Blob Storage | Cloud Storage | przechowywanie obrazów |
| Cosmos DB | Firestore | logi predykcji |
| Container Registry | Artifact Registry | rejestr obrazów Docker |
| GitHub Actions | GitHub Actions | CI/CD (bez zmian) |

---

## 1. Frontend — Firebase Hosting

- React (Vite) — **zaimplementowane**
- Formularz uploadu obrazu (drag & drop + klik), podgląd, wynik
  (Real / AI-generated + confidence score + pasek procentowy)
- Hosting: Firebase Hosting — globalny CDN, darmowy SSL, SPA-routing
- Deploy: `npm run build && firebase deploy` (lub auto-deploy z GitHub Actions)
- **Koszt: $0** (plan Spark, bez karty)

## 2. Backend API — Cloud Run

- Node.js / Express w kontenerze Docker
- Zadania: przyjmuje obraz od frontendu (`POST /analyze`, multipart/form-data),
  zapisuje do Cloud Storage, wywołuje ML Service, loguje wynik do Firestore,
  zwraca odpowiedź `{ "result": "...", "confidence": 0.0–1.0 }`
- **Scale-to-zero** przy braku ruchu (min-instances=0)
- Uwierzytelnianie do Storage/Firestore przez konto serwisowe (Application
  Default Credentials — automatycznie wewnątrz Cloud Run)
- **Koszt: ~$0** (darmowy tier: 2 mln żądań/mies., scale-to-zero)

## 3. ML Service — Cloud Run

- Python + FastAPI w osobnym kontenerze Docker
- Model ładowany **raz do pamięci przy starcie** kontenera (nie per-request)
- Zwraca: predykcja (Real / AI) + confidence score
- Model: TBD — CNN / ViT / fine-tuned ResNet, PyTorch lub TensorFlow
- Inferencja na CPU w zupełności wystarcza do demo/pracy; Cloud Run daje do
  32 GB RAM na instancję (mieści typowe modele wizyjne) **Cold start:** przy scale-to-zero pierwszy request po przerwie ładuje
  model (kilka sekund). Opcje:
  - zostawić scale-to-zero (najtaniej; akceptowalne do demo), lub
  - `min-instances=1` (zero cold-startów, ale serwis działa stale — drobny
    koszt CPU/RAM)
- **Koszt: ~$0** przy scale-to-zero

## 4. Warstwa danych

- **Cloud Storage** — uploadowane obrazy.
  Darmowy tier (~5 GB) z zapasem. **Koszt: ~$0**
- **Firestore** — logi predykcji: timestamp, wynik, confidence, metadane pliku
  (nazwa, rozmiar, typ MIME, ścieżka w Storage).
  Darmowy tier na zawsze (1 GB, 50 tys. odczytów/dzień, 20 tys. zapisów/dzień).
  **Materiał do analizy statystycznej w pracy dyplomowej.** **Koszt: $0**

## 5. CI/CD — GitHub Actions + Artifact Registry

Osobny pipeline dla każdego serwisu backendowego:

```
git push
  → GitHub Actions: build & test
  → docker build
  → push image → Artifact Registry
  → gcloud run deploy → Cloud Run
```

Frontend ma osobny, prostszy workflow — deploy bezpośrednio do Firebase Hosting
(`firebase deploy`), bez budowania obrazu Docker.

- GitHub Actions: **$0**
- Artifact Registry: 0,5 GB gratis, dalej ~$0,10/GB/mies. → realnie **~$0**
  (zamiast ~$5/mies. za Azure Container Registry Basic)

---

## Podsumowanie kosztów

| Usługa | Koszt miesięczny |
| --- | --- |
| Firebase Hosting | $0 |
| Cloud Run (backend) | ~$0 (scale-to-zero) |
| Cloud Run (ML) | ~$0 (scale-to-zero) |
| Cloud Storage | ~$0 |
| Firestore | $0 |
| Artifact Registry | ~$0 |
| GitHub Actions | $0 |
| **Razem** | **≈ $0 / mies.** |

