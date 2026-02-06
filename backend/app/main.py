from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

app = FastAPI(title="Credit SVM Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = Path(__file__).parent / "data" / "credit.csv"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "credit_pipeline.joblib"

# Globals cached in memory for speed (fine for take-home)
_df: Optional[pd.DataFrame] = None
_pipeline: Optional[Pipeline] = None
_viz_df: Optional[pd.DataFrame] = None

CATEGORICAL_COLS = [
    "status",
    "credit_history",
    "purpose",
    "savings",
    "employment_duration",
    "personal_status",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

NUMERIC_COLS = [
    "duration",
    "amount",
    "installment_rate",
    "present_residence",
    "age",
    "number_credits",
    "people_liable",
]

# label mapping for consistent API output
# (e.g. dataset might be 1/2, or "good"/"bad", etc.)
_label_map: Optional[Dict[Any, str]] = None


# ---------- Request schema for /predict ----------
# Based on the columns you listed (excluding credit_risk).
# Types: keep categoricals as str; numerics as int.

IntLike = Union[int, str]

class Applicant(BaseModel):
    status: IntLike
    duration: int = Field(..., ge=0)
    credit_history: IntLike
    purpose: IntLike
    amount: int = Field(..., ge=0)
    savings: IntLike
    employment_duration: IntLike
    installment_rate: int = Field(..., ge=0)
    personal_status: IntLike
    other_debtors: IntLike
    present_residence: int = Field(..., ge=0)
    property: IntLike
    age: int = Field(..., ge=0)
    other_installment_plans: IntLike
    housing: IntLike
    number_credits: int = Field(..., ge=0)
    job: IntLike
    people_liable: int = Field(..., ge=0)
    telephone: IntLike
    foreign_worker: IntLike


# ---------- Data / ML helpers ----------
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"credit.csv not found at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    if "credit_risk" not in df.columns:
        raise ValueError("Expected column 'credit_risk' not found in dataset.")

    return df


def infer_label_mapping(y_series: pd.Series) -> Dict[Any, str]:
    """
    Returns mapping from raw dataset labels to {"good","bad"} strings.
    Handles common cases: {1,2} or strings containing good/bad.
    """
    uniques = list(pd.unique(y_series))
    # German credit dataset commonly uses 1=good, 2=bad
    if set(uniques) == {1, 2}:
        return {1: "good", 2: "bad"}

    # If strings
    if all(isinstance(v, str) for v in uniques):
        lower = [v.strip().lower() for v in uniques]
        # map any label containing "good" to good, "bad" to bad
        mapping: Dict[Any, str] = {}
        for orig, l in zip(uniques, lower):
            if "good" in l or l in {"g", "pos", "positive"}:
                mapping[orig] = "good"
            elif "bad" in l or l in {"b", "neg", "negative"}:
                mapping[orig] = "bad"
        if len(mapping) == len(uniques):
            return mapping

    # Fallback: first label -> good, second -> bad (document this in README)
    if len(uniques) == 2:
        return {uniques[0]: "good", uniques[1]: "bad"}

    raise ValueError(f"Unsupported label values for credit_risk: {uniques}")


def build_full_pipeline(df: pd.DataFrame, C: float = 10.0, gamma: float = 0.1) -> Tuple[Pipeline, Dict[Any, str]]:
    """
    Full pipeline:
      preprocess (numeric scaling + categorical OHE)
      -> PCA(2)  [for visualization + boundary computation]
      -> SVC(RBF)
    """
    X = df.drop(columns=["credit_risk"])
    y_raw = df["credit_risk"]

    label_map = infer_label_mapping(y_raw)
    y = y_raw.map(label_map)  # becomes "good"/"bad"

    numeric_cols = NUMERIC_COLS
    categorical_cols = CATEGORICAL_COLS


    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    svc = SVC(kernel="rbf", C=C, gamma=gamma)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("pca", PCA(n_components=2, random_state=42)),
            ("svc", svc),
        ]
    )

    pipeline.fit(X, y)
    return pipeline, label_map


def ensure_model_loaded():
    """
    Loads model from disk if available, otherwise trains and saves it.
    Also prepares cached PCA-projected points for visualization.
    """
    global _df, _pipeline, _viz_df, _label_map

    if _df is None:
        _df = load_dataset()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if _pipeline is None:
        if MODEL_PATH.exists():
            payload = joblib.load(MODEL_PATH)
            _pipeline = payload["pipeline"]
            _label_map = payload["label_map"]
        else:
            # Hyperparameters: start with reasonable defaults.
            # (You said not to worry about justification for now.)
            _pipeline, _label_map = build_full_pipeline(_df, C=10.0, gamma=0.1)
            joblib.dump({"pipeline": _pipeline, "label_map": _label_map}, MODEL_PATH)

    # Cache PCA points for scatter
    X = _df.drop(columns=["credit_risk"])
    y_raw = _df["credit_risk"]
    y = y_raw.map(_label_map) if _label_map else y_raw.astype(str)

    coords = _pipeline.named_steps["pca"].transform(_pipeline.named_steps["preprocess"].transform(X))
    _viz_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "label": y.astype(str)})


@app.on_event("startup")
def on_startup():
    try:
        ensure_model_loaded()
    except Exception as e:
        # Let API start; endpoints will show clear errors
        print(f"[startup] Failed to initialize: {e}")


# ---------- Endpoints ----------
@app.get("/schema")
def get_schema(max_categories: int = Query(50, ge=5, le=500)):
    ensure_model_loaded()
    if _df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")

    X = _df.drop(columns=["credit_risk"])

    features = []

    # Numeric ranges
    for col in NUMERIC_COLS:
        s = pd.to_numeric(X[col], errors="coerce")
        features.append(
            {
                "name": col,
                "type": "numeric",
                "min": float(np.nanmin(s.values)),
                "max": float(np.nanmax(s.values)),
            }
        )

    # Categorical values (top by frequency)
    for col in CATEGORICAL_COLS:
        vc = X[col].astype(int).value_counts(dropna=False)
        values = [int(v) for v in vc.index.tolist()[:max_categories]]
        features.append({"name": col, "type": "categorical", "values": values})

    return {"features": features, "target_values": ["good", "bad"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/data")
def get_data(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    """
    Returns raw dataset rows for the table (paginated).
    """
    ensure_model_loaded()
    if _df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")
    total = len(_df)
    rows = _df.iloc[offset : offset + limit].to_dict(orient="records")
    return {"columns": _df.columns.tolist(), "total": total, "offset": offset, "limit": limit, "rows": rows}


@app.get("/viz/pca")
def get_pca_points(
    sample: int = Query(2000, ge=10, le=20000),
    seed: int = Query(42, ge=0),
):
    """
    Returns PCA-reduced (x,y) points + label for scatter plot.
    """
    ensure_model_loaded()
    if _viz_df is None:
        raise HTTPException(status_code=500, detail="Visualization data not ready.")

    df = _viz_df
    if len(df) > sample:
        df = df.sample(n=sample, random_state=seed)

    return {"count": len(df), "points": df.to_dict(orient="records")}


@app.get("/viz/grid")
def get_decision_grid(
    resolution: int = Query(200, ge=50, le=500),
    padding: float = Query(0.5, ge=0.0, le=5.0),
):
    """
    Returns a PCA-space grid + SVM decision scores for heatmap/contours.
    Frontend can render heatmap and decision boundary (score=0 contour).
    """
    ensure_model_loaded()
    if _pipeline is None or _viz_df is None:
        raise HTTPException(status_code=500, detail="Model not ready.")

    x_min, x_max = float(_viz_df["x"].min()), float(_viz_df["x"].max())
    y_min, y_max = float(_viz_df["y"].min()), float(_viz_df["y"].max())

    x_pad = (x_max - x_min) * padding
    y_pad = (y_max - y_min) * padding

    x0, x1 = x_min - x_pad, x_max + x_pad
    y0, y1 = y_min - y_pad, y_max + y_pad

    xs = np.linspace(x0, x1, resolution)
    ys = np.linspace(y0, y1, resolution)
    xx, yy = np.meshgrid(xs, ys)

    grid_points = np.c_[xx.ravel(), yy.ravel()]  # shape: (res^2, 2)

    svc: SVC = _pipeline.named_steps["svc"]
    # decision_function expects same feature space as training input to svc,
    # which is PCA space (2D), because svc comes after PCA in the pipeline.
    scores = svc.decision_function(grid_points)

    return {
        "resolution": resolution,
        "x_min": x0,
        "x_max": x1,
        "y_min": y0,
        "y_max": y1,
        "scores": scores.reshape(resolution, resolution).tolist(),  # 2D array
    }


@app.post("/predict")
def predict(applicant: Applicant):
    ensure_model_loaded()
    if _pipeline is None:
        raise HTTPException(status_code=500, detail="Model not ready.")

    payload = applicant.model_dump()

    # Coerce categorical codes to int (your dataset uses integer-coded categories)
    for col in CATEGORICAL_COLS:
        payload[col] = int(payload[col])

    X_new = pd.DataFrame([payload])

    pred = _pipeline.predict(X_new)[0]

    X_pre = _pipeline.named_steps["preprocess"].transform(X_new)
    coords = _pipeline.named_steps["pca"].transform(X_pre)[0]

    svc: SVC = _pipeline.named_steps["svc"]
    score = float(svc.decision_function(coords.reshape(1, -1))[0])

    return {
        "prediction": str(pred),
        "pca": {"x": float(coords[0]), "y": float(coords[1])},
        "decision_score": score,
    }
