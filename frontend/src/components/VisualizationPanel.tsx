import { useEffect, useMemo, useState } from "react";
import type { GridResponse, PcaPoint } from "../api/types";
import { getGrid, getPca } from "../api/endpoints";
import type * as Plotly from "plotly.js";
import Plot from "react-plotly.js";


type Props = {
    applicantPoint?: { x: number; y: number; prediction?: string };
};

export default function VisualizationPanel({ applicantPoint }: Props) {
    const [points, setPoints] = useState<PcaPoint[]>([]);
    const [grid, setGrid] = useState<GridResponse | null>(null);
    const [error, setError] = useState<string>("");

    useEffect(() => {
        let cancelled = false;

        async function load() {
            try {
                setError("");
                const [pcaRes, gridRes] = await Promise.all([getPca(2000), getGrid(200, 0.5)]);
                if (cancelled) return;
                setPoints(pcaRes.points);
                setGrid(gridRes);
            } catch (e: any) {
                if (cancelled) return;
                setError(String(e?.message ?? e));
            }
        }

        load();
        return () => {
            cancelled = true;
        };
    }, []);

    const traces = useMemo(() => {
        const t: Partial<Plotly.PlotData>[] = [];


        if (grid) {
            // Heatmap of decision scores
            t.push({
                type: "heatmap",
                z: grid.scores,
                x: linspace(grid.x_min, grid.x_max, grid.resolution),
                y: linspace(grid.y_min, grid.y_max, grid.resolution),
                opacity: 0.55,
                showscale: true,
                hovertemplate: "score=%{z:.3f}<extra></extra>",
            });

            // Decision boundary as a contour line at score=0
            t.push({
                type: "contour",
                z: grid.scores,
                name: "Decision boundary",
                x: linspace(grid.x_min, grid.x_max, grid.resolution),
                y: linspace(grid.y_min, grid.y_max, grid.resolution),
                contours: {
                    coloring: "none",
                    showlines: true,
                    start: 0,
                    end: 0,
                    size: 1,
                },
                line: { width: 2 },
                showscale: false,
                hoverinfo: "skip",
            });
        }

        if (points.length) {
            const good = points.filter((p) => p.label === "good" || p.label === "1");
            const bad = points.filter((p) => p.label === "bad" || p.label === "2");

            t.push({
                type: "scattergl",
                mode: "markers",
                name: "Good",
                x: good.map((p) => p.x),
                y: good.map((p) => p.y),
                marker: { size: 5 },
                hovertext: good.map(() => "good"),
                hovertemplate: "good<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
            });

            t.push({
                type: "scattergl",
                mode: "markers",
                name: "Bad",
                x: bad.map((p) => p.x),
                y: bad.map((p) => p.y),
                marker: { size: 5 },
                hovertext: bad.map(() => "bad"),
                hovertemplate: "bad<br>x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
            });
        }

        if (applicantPoint) {
            t.push({
                type: "scatter",
                mode: "markers",
                name: "Applicant",
                x: [applicantPoint.x],
                y: [applicantPoint.y],
                marker: { size: 12, symbol: "x" },
                hovertemplate:
                    `${applicantPoint.prediction ?? "applicant"}<br>` +
                    "x=%{x:.2f}, y=%{y:.2f}<extra></extra>",
            });
        }

        return t;
    }, [grid, points, applicantPoint]);

    return (
        <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 12 }}>
            <h2 style={{ margin: "0 0 8px" }}>Visualization</h2>
            {error ? (
                <div style={{ color: "crimson" }}>{error}</div>
            ) : (
                <Plot
                    data={traces}
                    layout={{
                        autosize: true,
                        height: 560, // bump this if you want it taller
                        margin: { l: 55, r: 15, t: 10, b: 55 },
                        xaxis: { title: { text: "PC1" }, zeroline: false },
                        yaxis: { title: { text: "PC2" }, zeroline: false },
                        legend: { orientation: "h" },
                    }}
                    useResizeHandler={true}
                    style={{ width: "100%", height: "100%" }}
                    config={{ displayModeBar: false, responsive: true }}
                />

            )}
        </div>
    );
}

function linspace(min: number, max: number, n: number): number[] {
    if (n <= 1) return [min];
    const step = (max - min) / (n - 1);
    const arr = new Array(n);
    for (let i = 0; i < n; i++) arr[i] = min + step * i;
    return arr;
}
