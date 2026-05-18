// Editable view of prediction results. Low-confidence rows are flagged.

import { useMemo, useState } from "react";

function pct(v) {
  return `${Math.round((v ?? 0) * 100)}%`;
}

function PredictionTable({ predictions, onEdit }) {
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState("row_id");
  const [sortDir, setSortDir] = useState("asc");
  const [showChangedOnly, setShowChangedOnly] = useState(false);
  const [showLowOnly, setShowLowOnly] = useState(false);
  const [pageSize, setPageSize] = useState(50);
  const [page, setPage] = useState(1);
  const safeRows = predictions ?? [];

  const labelOptions = useMemo(() => {
    const labels = new Set();
    safeRows.forEach((p) => {
      if (p.predicted_condition) labels.add(p.predicted_condition);
      if (p.final_condition) labels.add(p.final_condition);
    });
    return Array.from(labels).sort((a, b) => a.localeCompare(b));
  }, [safeRows]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const rows = safeRows.filter((p) => {
      if (showChangedOnly && p.final_condition === p.predicted_condition) {
        return false;
      }
      if (showLowOnly && !p.low_confidence) return false;
      if (!q) return true;
      const hay = [
        String(p.row_id ?? ""),
        p.discrepancy,
        p.predicted_condition,
        p.final_condition,
        p.xai?.keyword,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return hay.includes(q);
    });

    const dir = sortDir === "asc" ? 1 : -1;
    const sorted = [...rows].sort((a, b) => {
      const getVal = (row) => {
        if (sortKey === "confidence") return row.confidence ?? 0;
        if (sortKey === "row_id") return Number(row.row_id ?? 0);
        return String(row[sortKey] ?? "").toLowerCase();
      };
      const av = getVal(a);
      const bv = getVal(b);
      if (typeof av === "number" && typeof bv === "number") {
        return (av - bv) * dir;
      }
      return String(av).localeCompare(String(bv)) * dir;
    });
    return sorted;
  }, [safeRows, query, showChangedOnly, showLowOnly, sortKey, sortDir]);

  const total = filtered.length;
  const maxPage = pageSize === 0 ? 1 : Math.max(1, Math.ceil(total / pageSize));
  const safePage = Math.min(page, maxPage);
  const start = pageSize === 0 ? 0 : (safePage - 1) * pageSize;
  const end = pageSize === 0 ? total : start + pageSize;
  const paged = filtered.slice(start, end);

  if (!safeRows.length) return null;

  function toggleSort(nextKey) {
    if (sortKey === nextKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(nextKey);
      setSortDir("asc");
    }
  }

  return (
    <div className="pred-table-wrap">
      <div className="table-toolbar">
        <div className="table-controls">
          <input
            className="input"
            type="search"
            placeholder="Filter rows..."
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(1);
            }}
          />
          <label className="pill">
            <input
              type="checkbox"
              checked={showChangedOnly}
              onChange={(e) => {
                setShowChangedOnly(e.target.checked);
                setPage(1);
              }}
            />
            Edited only
          </label>
          <label className="pill">
            <input
              type="checkbox"
              checked={showLowOnly}
              onChange={(e) => {
                setShowLowOnly(e.target.checked);
                setPage(1);
              }}
            />
            Low confidence
          </label>
        </div>
        <div className="table-controls">
          <div className="sort-group">
            <button
              className="sort-btn"
              type="button"
              onClick={() => toggleSort("row_id")}
            >
              ID {sortKey === "row_id" ? (sortDir === "asc" ? "▲" : "▼") : ""}
            </button>
            <button
              className="sort-btn"
              type="button"
              onClick={() => toggleSort("predicted_condition")}
            >
              Predicted {sortKey === "predicted_condition" ? (sortDir === "asc" ? "▲" : "▼") : ""}
            </button>
            <button
              className="sort-btn"
              type="button"
              onClick={() => toggleSort("final_condition")}
            >
              Final {sortKey === "final_condition" ? (sortDir === "asc" ? "▲" : "▼") : ""}
            </button>
            <button
              className="sort-btn"
              type="button"
              onClick={() => toggleSort("confidence")}
            >
              Confidence {sortKey === "confidence" ? (sortDir === "asc" ? "▲" : "▼") : ""}
            </button>
          </div>
          <select
            className="select"
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setPage(1);
            }}
          >
            <option value={25}>25 rows</option>
            <option value={50}>50 rows</option>
            <option value={100}>100 rows</option>
            <option value={0}>All rows</option>
          </select>
        </div>
      </div>
      <div className="table-meta">
        Showing {paged.length} of {total} rows
      </div>
      <datalist id="cond-options">
        {labelOptions.map((label) => (
          <option key={label} value={label} />
        ))}
      </datalist>
      <table className="pred-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Discrepancy</th>
            <th>Predicted Condition</th>
            <th>Final Condition</th>
            <th>Confidence</th>
            <th>XAI Keyword</th>
            <th>Explanation</th>
          </tr>
        </thead>
        <tbody>
          {paged.map((p) => (
            <tr
              key={p.row_id}
              className={p.low_confidence ? "row-low-conf" : ""}
            >
              <td>{p.row_id}</td>
              <td className="cell-text">{p.discrepancy}</td>
              <td>
                <span className="cond-badge">{p.predicted_condition}</span>
              </td>
              <td>
                <input
                  className="input"
                  type="text"
                  value={p.final_condition ?? p.predicted_condition ?? ""}
                  onChange={(e) => onEdit?.(p.row_id, e.target.value)}
                  placeholder="Set final condition"
                  list="cond-options"
                />
                {p.final_condition &&
                  p.predicted_condition &&
                  p.final_condition !== p.predicted_condition && (
                    <span className="edit-flag">edited</span>
                  )}
              </td>
              <td>
                <span
                  className={
                    "conf" + (p.low_confidence ? " conf-low" : " conf-ok")
                  }
                >
                  {pct(p.confidence)}
                  {p.low_confidence && " ⚠"}
                </span>
              </td>
              <td>
                <code>{p.xai?.keyword}</code>
              </td>
              <td className="cell-text">{p.xai?.explanation}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {pageSize !== 0 && maxPage > 1 && (
        <div className="table-footer">
          <button
            className="btn-ghost"
            type="button"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={safePage === 1}
          >
            Previous
          </button>
          <span>
            Page {safePage} of {maxPage}
          </span>
          <button
            className="btn-ghost"
            type="button"
            onClick={() => setPage((p) => Math.min(maxPage, p + 1))}
            disabled={safePage === maxPage}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

export default PredictionTable;
