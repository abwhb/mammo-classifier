export type Prediction = {
  label: "malignant" | "benign";
  confidence: number;
  malignant_probability: number;
  threshold: number;
  model_version: string;
  latency_ms: number;
  request_id: string;
  disclaimer: string;
};

export type ApiError = { error: string; status?: number };

export function isPrediction(x: unknown): x is Prediction {
  return (
    typeof x === "object" &&
    x !== null &&
    "label" in x &&
    "confidence" in x &&
    "model_version" in x
  );
}
