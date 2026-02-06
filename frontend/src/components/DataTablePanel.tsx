import { useEffect, useState } from "react";
import { apiGet } from "../api/client";

type DataResponse = {
  columns: string[];
  total: number;
  offset: number;
  limit: number;
  rows: Record<string, any>[];
};

export default function DataTablePanel() {
  const [data, setData] = useState<DataResponse | null>(null);
  const [error, setError] = useState("");
  const [offset, setOffset] = useState(0);
  const limit = 25;

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setError("");
        const res = await apiGet<DataResponse>(`/data?offset=${offset}&limit=${limit}`);
        if (cancelled) return;
        setData(res);
      } catch (e: any) {
        if (cancelled) return;
        setError(String(e?.message ?? e));
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [offset]);

  if (error) return <Panel title="Data Table">Error: {error}</Panel>;
  if (!data) return <Panel title="Data Table">Loading…</Panel>;

  const canPrev = offset > 0;
  const canNext = offset + limit < data.total;

  return (
    <Panel title="Data Table">
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
        <button disabled={!canPrev} onClick={() => setOffset(Math.max(0, offset - limit))}>
          Prev
        </button>
        <button disabled={!canNext} onClick={() => setOffset(offset + limit)}>
          Next
        </button>
        <span style={{ color: "#666" }}>
          Showing {offset + 1}–{Math.min(offset + limit, data.total)} of {data.total}
        </span>
      </div>

      <div style={{ overflow: "auto", maxHeight: 420, border: "1px solid #eee", borderRadius: 10 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr>
              {data.columns.map((c) => (
                <th key={c} style={thStyle}>
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.rows.map((row, idx) => (
              <tr key={idx}>
                {data.columns.map((c) => (
                  <td key={c} style={tdStyle}>
                    {String(row[c] ?? "")}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

const thStyle: React.CSSProperties = {
  position: "sticky",
  top: 0,
  background: "white",
  textAlign: "left",
  padding: "8px 10px",
  borderBottom: "1px solid #eee",
};

const tdStyle: React.CSSProperties = {
  padding: "6px 10px",
  borderBottom: "1px solid #f3f3f3",
  whiteSpace: "nowrap",
};

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ border: "1px solid #ddd", borderRadius: 12, padding: 12 }}>
      <h2 style={{ margin: "0 0 8px" }}>{title}</h2>
      {children}
    </div>
  );
}
