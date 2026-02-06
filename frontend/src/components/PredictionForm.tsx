import { useEffect, useMemo, useState } from "react";
import { getSchema, predict } from "../api/endpoints";
import type { FeatureSchema, SchemaResponse } from "../api/types";

type Props = {
  onPredicted: (p: { x: number; y: number; prediction: string }) => void;
};

export default function PredictionForm({ onPredicted }: Props) {
  const [schema, setSchema] = useState<SchemaResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string>("");
  const [result, setResult] = useState<{ prediction: string; decision_score: number } | null>(null);

  // Form state as a simple key-value map
  const [form, setForm] = useState<Record<string, string>>({});

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        setError("");
        const s = await getSchema();
        if (cancelled) return;
        setSchema(s);

        // Initialize defaults: first categorical option; midpoint numeric
        const defaults: Record<string, string> = {};
        for (const f of s.features) {
          if (f.type === "categorical") {
            defaults[f.name] = String(f.values[0] ?? "");
          } else {
            const mid = Math.round((f.min + f.max) / 2);
            defaults[f.name] = String(mid);
          }
        }
        setForm(defaults);
      } catch (e: any) {
        if (cancelled) return;
        setError(String(e?.message ?? e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const features = useMemo(() => schema?.features ?? [], [schema]);

  function updateField(name: string, value: string) {
    setForm((prev) => ({ ...prev, [name]: value }));
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!schema) return;

    setSubmitting(true);
    setError("");
    setResult(null);

    try {
      // Build payload: convert all fields to numbers because your dataset uses int-coded values
      const payload: Record<string, number> = {};
      for (const f of features) {
        const raw = form[f.name];
        if (raw === undefined || raw === "") {
          throw new Error(`Missing value for ${f.name}`);
        }
        const num = Number(raw);
        if (!Number.isFinite(num)) {
          throw new Error(`Invalid number for ${f.name}: "${raw}"`);
        }
        payload[f.name] = num;
      }

      const res = await predict(payload);

      setResult({ prediction: res.prediction, decision_score: res.decision_score });
      onPredicted({ x: res.pca.x, y: res.pca.y, prediction: res.prediction });
    } catch (e: any) {
      setError(String(e?.message ?? e));
    } finally {
      setSubmitting(false);
    }
  }

  if (loading) return <Panel title="Prediction">Loading schema…</Panel>;
  if (error && !schema) return <Panel title="Prediction">Error: {error}</Panel>;
  if (!schema) return <Panel title="Prediction">No schema loaded.</Panel>;

  return (
    <Panel title="Prediction">
      <form onSubmit={onSubmit} style={{ display: "grid", gap: 10 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          {features.map((f) => (
            <Field
              key={f.name}
              feature={f}
              value={form[f.name] ?? ""}
              onChange={(v) => updateField(f.name, v)}
            />
          ))}
        </div>

        <button
          type="submit"
          disabled={submitting}
          style={{
            padding: "10px 12px",
            borderRadius: 10,
            border: "1px solid #ddd",
            cursor: submitting ? "not-allowed" : "pointer",
            fontWeight: 600,
          }}
        >
          {submitting ? "Predicting…" : "Predict Credit Risk"}
        </button>

        {error ? <div style={{ color: "crimson" }}>{error}</div> : null}

        {result ? (
          <div style={{ padding: 10, border: "1px solid #eee", borderRadius: 10 }}>
            <div>
              <strong>Prediction:</strong> {result.prediction}
            </div>
            <div style={{ color: "#666" }}>
              Decision score: {result.decision_score.toFixed(3)} (boundary at 0)
            </div>
          </div>
        ) : null}
      </form>
    </Panel>
  );
}

function Field({
  feature,
  value,
  onChange,
}: {
  feature: FeatureSchema;
  value: string;
  onChange: (v: string) => void;
}) {
  const labelStyle: React.CSSProperties = { fontSize: 12, color: "#444", marginBottom: 4 };
  const inputStyle: React.CSSProperties = {
    width: "100%",
    padding: "8px 10px",
    borderRadius: 10,
    border: "1px solid #ddd",
  };

  if (feature.type === "categorical") {
    return (
      <div>
        <div style={labelStyle}>{feature.name}</div>
        <select value={value} onChange={(e) => onChange(e.target.value)} style={inputStyle}>
          {feature.values.map((v) => (
            <option key={v} value={String(v)}>
              {v}
            </option>
          ))}
        </select>
      </div>
    );
  }

  return (
    <div>
      <div style={labelStyle}>
        {feature.name} <span style={{ color: "#999" }}>({feature.min}–{feature.max})</span>
      </div>
      <input
        type="number"
        value={value}
        min={feature.min}
        max={feature.max}
        onChange={(e) => onChange(e.target.value)}
        style={inputStyle}
      />
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 12 }}>
      <h2 style={{ margin: "0 0 8px" }}>{title}</h2>
      {children}
    </div>
  );
}
