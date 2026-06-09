# AI Image Detector — Frontend

Web frontend for detecting AI-generated images. The user uploads an image and
receives a verdict (**Real** / **AI-generated**) with a confidence score.

Built with **React + Vite**, deployed to **Azure Static Web Apps** via GitHub Actions.

## Features

- Drag & drop or click-to-browse image upload
- Live preview before sending
- `POST /analyze` request (`multipart/form-data`, field `image`)
- Result label + confidence score + progress bar
- Client-side validation (format, size) and network error / timeout handling
- Built-in mock backend so the app runs with no server

## Requirements

- Node.js 18+ and npm

## Run locally

```bash
npm install
npm run dev      # http://localhost:5173
```

The app starts in **mock mode** by default and returns a sample response.

## Connect the real backend

Everything is controlled from `src/config.js` via two environment variables.
Copy `.env.example` to `.env` and set:

```bash
VITE_API_URL=https://your-backend.example.com
VITE_USE_MOCK=false
```

That is the only change needed — `src/api.js` automatically calls
`POST {VITE_API_URL}/analyze` instead of the mock.

**Expected backend response:**

```json
{ "result": "AI-generated", "confidence": 0.943 }
```

## Build

```bash
npm run build    # outputs to /dist
npm run preview  # preview the production build
```

## Deployment (Azure Static Web Apps)

Pushing to `main` triggers `.github/workflows/deploy-frontend.yml`, which builds
the app and deploys `/dist`.

### Required GitHub secret

| Secret | Where to get it |
| --- | --- |
| `AZURE_STATIC_WEB_APPS_API_TOKEN` | Azure Portal → your Static Web App → **Manage deployment token** |

Add it under **GitHub repo → Settings → Secrets and variables → Actions → New repository secret**.

> `GITHUB_TOKEN` is provided automatically by GitHub Actions — you do not add it.

### Pointing the deployed app at the real backend

In `.github/workflows/deploy-frontend.yml`, under the build step's `env:`, set
`VITE_API_URL` and change `VITE_USE_MOCK` to `"false"`. These are baked in at
build time.

## Project structure

```
.
├── .github/workflows/deploy-frontend.yml   # CI/CD: build + deploy to Azure
├── src/
│   ├── components/
│   │   ├── UploadForm.jsx                   # Drag & drop + validation
│   │   └── ResultCard.jsx                   # Label + confidence bar
│   ├── api.js                               # analyzeImage(): mock or real
│   ├── config.js                            # Single source of config / env
│   ├── App.jsx                              # App state machine
│   ├── main.jsx                             # React entry point
│   └── index.css                            # Styling (plain CSS)
├── staticwebapp.config.json                # Azure SWA routing + headers
├── index.html
├── vite.config.js
└── package.json
```
