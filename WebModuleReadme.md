# Web Module Notes

This file contains the instructions on how to run and update the web module.

---

# Project Structure

```text
project-root/
│
├── data/                 # Raw datasets (CSV, Excel, etc.)
├── experiments/          # Notebooks / experiments / testing
├── models/               # Saved trained models
├── pipeline/             # Training + preprocessing pipelines
├── scripts/              # Setup / utility scripts
├── src/                  # Core ML / NLP logic
│   ├── models/
│   ├── preprocessing/
│   └── utils/
│
├── backend/              # Flask API (backend layer)
│   ├── app/
│   │   ├── routes/       # API endpoints (Blueprints)
│   │   │   ├── health.py
│   │   │   ├── upload.py
│   │   │   ├── predict.py
│   │   │   ├── train.py
│   │   │   └── feedback.py
│   │   │
│   │   └── __init__.py   # Flask app factory + blueprint registration
│   │
│   └── run.py            # Backend entry point
│
├── frontend/             # React + Vite frontend
│   ├── src/
│   │   ├── pages/        # Full screens (routes)
│   │   ├── components/   # Reusable UI components
│   │   ├── services/     # API calls (backend communication)
│   │   │   └── api.js
│   │   │
│   │   ├── App.jsx       # Routing shell
│   │   └── main.jsx
│   │
│   └── package.json
│
└── README.md
```

---

# Running the App

## Backend

From project root:

```bash
python backend/run.py
```

Runs on:

```text
http://localhost:5000
```

---

## Frontend

From `frontend/`:

```bash
npm install
npm run dev
```

Runs on:

```text
http://localhost:5173
```

---

# Backend Rules (Flask)

* Use **Blueprints per route file**
* Keep ML logic in `/src`
* Backend only handles API + calls ML functions

Example routes:

```text
/api/health
/api/upload
/api/predict
/api/train
/api/feedback
```

---

# Frontend Rules (React)

* UI only (no ML logic)
* API calls go through:

```text
frontend/src/services/api.js
```

* Components in `/components`
* Pages in `/pages`

---

# Core Flow

```text
React UI
   ↓
Flask API
   ↓
src/ ML code
   ↓
Predictions returned
   ↓
React displays results
```

---

# Key Development Rules

## DO

* Keep ML code in `/src`
* Keep Flask routes small
* Use API service layer in React
* Register new Flask Blueprints

## DO NOT

* Put ML logic inside Flask routes
* Duplicate pipeline code
* Break existing folder structure

---

# Basic Commands

## Backend

```bash
python backend/run.py
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

---

# Notes

* Ensure Node.js (LTS) is installed for frontend (Node.js version v24.15.0 (LTS)): https://nodejs.org/en/download
* Ensure Python venv is active for backend
* CORS must be enabled for React ↔ Flask communication
