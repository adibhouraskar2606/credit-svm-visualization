# Credit SVM Visualization App

Full-stack app that trains an SVM (RBF) to predict credit risk and visualizes decision regions in 2D PCA space.

## Architecture
- **Backend:** FastAPI + scikit-learn pipeline  
  - preprocess (numeric scaling + categorical OHE) → PCA(2) → SVC(RBF)
  - endpoints for raw data, PCA points, decision grid, prediction
- **Frontend:** React (Vite) + Plotly  
  - scatter of PCA points + heatmap of decision scores + decision boundary contour
  - form to submit a new applicant and plot it
  - paginated table to inspect raw dataset

## Setup

### Backend
```bash
cd backend
python -m venv .venv
# windows: .\.venv\Scripts\Activate.ps1
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
