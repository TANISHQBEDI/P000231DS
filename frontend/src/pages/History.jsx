import { useEffect, useState } from "react";

import { getHistory, getHistoryItem } from "../services/api";

function History() {
  const [items, setItems] = useState([]);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHistory()
      .then((res) => setItems(res.items || []))
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  async function openItem(id) {
    setError("");
    try {
      setSelected(await getHistoryItem(id));
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <div className="page">
      <header className="page-head">
        <h1>Archived History</h1>
        <p className="page-sub">
          Versioned audit trail of saved predictions (timestamp + unique ID +
          model version).
        </p>
      </header>

      <section className="card">
        {loading && <p>Loading…</p>}
        {error && <p className="form-error">{error}</p>}
        {!loading && !items.length && <p>No archived records yet.</p>}
        {items.length > 0 && (
          <table className="pred-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>ID</th>
                <th>Model</th>
                <th>User</th>
                <th>Rows</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {items.map((it) => (
                <tr key={it.id}>
                  <td>{new Date(it.timestamp).toLocaleString()}</td>
                  <td>
                    <code>{it.id.slice(0, 8)}</code>
                  </td>
                  <td>{it.model_version}</td>
                  <td>{it.user}</td>
                  <td>{it.num_records}</td>
                  <td>
                    <button
                      className="btn-link"
                      onClick={() => openItem(it.id)}
                    >
                      View
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      {selected && (
        <section className="card">
          <h2>Record {selected.id.slice(0, 8)}</h2>
          <p className="page-sub">
            {new Date(selected.timestamp).toLocaleString()} ·{" "}
            {selected.model_version} · {selected.num_records} rows
          </p>
          <div className="diff-grid">
            <div>
              <h3>Before (pre-edit)</h3>
              <pre className="json-block">
                {JSON.stringify(selected.before, null, 2)}
              </pre>
            </div>
            <div>
              <h3>After (saved)</h3>
              <pre className="json-block">
                {JSON.stringify(selected.after, null, 2)}
              </pre>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

export default History;
