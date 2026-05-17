import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { useApp } from "../context/AppContext";
import { uploadCsv, runPrediction } from "../services/api";
import FileUpload from "../components/FileUpload";
import EditableGrid from "../components/EditableGrid";

function Dashboard() {
  const navigate = useNavigate();
  const {
    uploadedRows,
    setUploadedRows,
    columns,
    setColumns,
    editedRows,
    setEditedRows,
    setPrediction,
  } = useApp();

  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleLoaded(rows, cols, file) {
    setError("");
    setStatus("Validating with backend…");
    setUploadedRows(rows);
    setColumns(cols);
    setEditedRows(rows);
    try {
      const res = await uploadCsv(file);
      setStatus(
        `Backend accepted ${res.rows.length} row(s). Edit below, then run prediction.`,
      );
    } catch (e) {
      // Client parse already succeeded; surface backend note but keep grid.
      setStatus("");
      setError(`Backend validation: ${e.message}`);
    }
  }

  async function handlePredict() {
    setError("");
    setBusy(true);
    try {
      const result = await runPrediction(editedRows);
      setPrediction({ ...result, source: editedRows });
      navigate("/results");
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <header className="page-head">
        <h1>Current Upload</h1>
        <p className="page-sub">
          Upload a discrepancy CSV, review &amp; edit the records, then run the
          condition prediction.
        </p>
      </header>

      <section className="card">
        <h2>1 · Upload CSV</h2>
        <FileUpload onLoaded={handleLoaded} />
        {status && <p className="form-ok">{status}</p>}
        {error && <p className="form-error">{error}</p>}
      </section>

      {uploadedRows.length > 0 && (
        <section className="card">
          <h2>2 · Review &amp; Edit ({editedRows.length} rows)</h2>
          <p className="page-sub">
            Edit any cell (e.g. clean up a description) before prediction.
          </p>
          <EditableGrid
            columns={columns}
            rows={editedRows}
            onRowsChange={setEditedRows}
          />
          <div className="actions">
            <button
              className="btn-primary"
              onClick={handlePredict}
              disabled={busy}
            >
              {busy ? "Running…" : "Run Prediction →"}
            </button>
          </div>
        </section>
      )}
    </div>
  );
}

export default Dashboard;
