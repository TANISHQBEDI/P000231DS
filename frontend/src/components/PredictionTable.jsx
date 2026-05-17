// Read-only view of prediction results. Low-confidence rows are flagged.

function pct(v) {
  return `${Math.round((v ?? 0) * 100)}%`;
}

function PredictionTable({ predictions }) {
  if (!predictions?.length) return null;
  return (
    <div className="pred-table-wrap">
      <table className="pred-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Discrepancy</th>
            <th>Predicted Condition</th>
            <th>Confidence</th>
            <th>XAI Keyword</th>
            <th>Explanation</th>
          </tr>
        </thead>
        <tbody>
          {predictions.map((p) => (
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
    </div>
  );
}

export default PredictionTable;
