export type PcaPoint = { x: number; y: number; label: string };

export type PcaResponse = {
  count: number;
  points: PcaPoint[];
};

export type GridResponse = {
  resolution: number;
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
  scores: number[][];
};

export type PredictResponse = {
  prediction: string;
  pca: { x: number; y: number };
  decision_score: number;
};

export type FeatureSchema =
  | { name: string; type: "numeric"; min: number; max: number }
  | { name: string; type: "categorical"; values: number[] };

export type SchemaResponse = {
  features: FeatureSchema[];
  target_values: string[];
};
