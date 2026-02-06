import { apiGet, apiPost } from "./client";
import type { GridResponse, PcaResponse, PredictResponse, SchemaResponse } from "./types";

export const getPca = (sample = 2000) => apiGet<PcaResponse>(`/viz/pca?sample=${sample}`);
export const getGrid = (resolution = 200, padding = 0.5) =>
  apiGet<GridResponse>(`/viz/grid?resolution=${resolution}&padding=${padding}`);

export const getSchema = () => apiGet<SchemaResponse>("/schema");

export const predict = (payload: unknown) => apiPost<PredictResponse>("/predict", payload);
