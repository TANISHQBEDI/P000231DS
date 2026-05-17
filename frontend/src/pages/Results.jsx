import { useState } from "react";
import { Link } from "react-router-dom";

import { useApp } from "../context/AppContext";
import { submitFeedback } from "../services/api";
import PredictionTable from "../components/PredictionTable";

function Results() {
  const { prediction } = useApp();
  const [saveMsg, setSaveMsg] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  if (!prediction) {
    return (
      <div className="page">
        <header className="page-head">
          <h1>Prediction Results</h1>
        </header>
        <div className="card empty">
          <p>No predictions yet.</p>
          <Link className="btn-primary" to="/">
            Go to Current Upload
          </Link>
        </div>
      </div>
    );
  }

  const { predictions, model_version, generated_at, source } = prediction;
  const lowCount = predictions.filter((p) => p.low_confidence).length;

  async function handleSave() {
    setError("");
    setBusy(true);
    try {
      const res = await submitFeedback(predictions, source || []);
      setSaveMsg(
        `Saved as record ${res.id} (${res.num_records} rows) — see Archived History.`,
      );
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header className="page-head">
        <h1>Prediction Results</h1>
        <p className="page-sub">
          Model <code>{model_version}</code> ·{" "}
          {new Date(generated_at).toLocaleString()} · {predictions.length} rows ·{" "}
          {lowCount > 0 ? (
            <span className="conf-low">{lowCount} low-confidence ⚠</span>
          ) : (
            <span className="conf-ok">all confident</span>
          )}
        </p>
      </header>

      <section className="card">
        <PredictionTable predictions={predictions} />
        <div className="actions">
          <button
            className="btn-primary"
            onClick={handleSave}
            disabled={busy}
          >
            {busy ? "Saving…" : "Save / Submit Feedback"}
          </button>
        </div>
        {saveMsg && <p className="form-ok">{saveMsg}</p>}
        {error && <p className="form-error">{error}</p>}
      </section>
    </div>
  );
}

export default Results;
