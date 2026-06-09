# AI Image Detector — Frontend

Web frontend for detecting AI-generated images. The user uploads an image and
receives a verdict (**Real** / **AI-generated**) with a confidence score.

Built with **React + Vite**, deployed to **Firebase Hosting**.

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

## Deployment (Firebase Hosting)

One-time setup:

```bash
npm install -g firebase-tools   # install the Firebase CLI
firebase login                  # log in with your Google account
```

Create a project at https://console.firebase.google.com, then point this repo
at it (replaces the placeholder in `.firebaserc`):

```bash
firebase use --add              # pick your project, alias it "default"
```

Deploy (build first, then upload `/dist`):

```bash
npm run build
firebase deploy
```

The app goes live at `https://<your-project-id>.web.app`.

Config lives in `firebase.json`: it serves the `dist/` folder and rewrites all
routes to `index.html` (SPA routing).

### Pointing the deployed app at the real backend

Set the env vars before building, then deploy:

```bash
VITE_API_URL=https://your-backend.example.com VITE_USE_MOCK=false npm run build
firebase deploy
```

(or put them in a local `.env` file — see `.env.example`).

## Project structure

```
.
├── src/
│   ├── components/
│   │   ├── UploadForm.jsx     # Drag & drop + validation
│   │   └── ResultCard.jsx     # Label + confidence bar
│   ├── api.js                 # analyzeImage(): mock or real
│   ├── config.js              # Single source of config / env
│   ├── App.jsx                # App state machine
│   ├── main.jsx               # React entry point
│   └── index.css             # Styling (plain CSS)
├── firebase.json              # Firebase Hosting config (serves dist/, SPA routing)
├── .firebaserc                # Firebase project alias
├── index.html
├── vite.config.js
└── package.json
```
