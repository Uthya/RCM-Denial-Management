# RCM Claim Denial Management System

A full-stack application for managing healthcare claim denials, tracking appeals, and predicting denial risk using machine learning.

## Tech Stack

- **Frontend:** React (Vite) + Tailwind CSS
- **Backend:** Python + FastAPI
- **Database:** PostgreSQL
- **ML Model:** XGBoost (denial prediction)

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.12+
- PostgreSQL 16+
- Docker & Docker Compose (optional)

### Backend

```bash
cd backend
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

### Docker

```bash
docker-compose up
```

## API Documentation

Once the backend is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
