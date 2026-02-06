import { useState } from "react";
import VisualizationPanel from "./components/VisualizationPanel";
import PredictionForm from "./components/PredictionForm";
import DataTablePanel from "./components/DataTablePanel";

export default function App() {
  const [applicantPoint, setApplicantPoint] = useState<{ x: number; y: number; prediction?: string }>();
  const isNarrow = window.matchMedia?.("(max-width: 1100px)").matches;
  return (
    <div style={{ padding: 16, fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ marginTop: 0 }}>Credit SVM UI</h1>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: isNarrow ? "1fr" : "1.2fr 0.8fr",
          gap: 16,
          alignItems: "start",
        }}
      >
        <VisualizationPanel applicantPoint={applicantPoint} />
        <PredictionForm onPredicted={(p) => setApplicantPoint(p)} />
      </div>

      <div style={{ marginTop: 16 }}>
        <DataTablePanel />
      </div>
    </div>
  );
}
