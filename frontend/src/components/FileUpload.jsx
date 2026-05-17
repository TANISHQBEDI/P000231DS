import { useRef, useState } from "react";
import Papa from "papaparse";

const REQUIRED_COLUMN = "Discrepancy";

// Client-side validation BEFORE hitting the backend (per spec).
function validateAndParse(file) {
  return new Promise((resolve) => {
    if (!file) {
      resolve({ ok: false, error: "No file selected." });
      return;
    }
    if (!file.name.toLowerCase().endsWith(".csv")) {
      resolve({ ok: false, error: "File must have a .csv extension." });
      return;
    }
    if (file.size === 0) {
      resolve({ ok: false, error: "File is empty." });
      return;
    }
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (res) => {
        const cols = res.meta.fields || [];
        if (!cols.includes(REQUIRED_COLUMN)) {
          resolve({
            ok: false,
            error: `Missing required column "${REQUIRED_COLUMN}". Found: ${
              cols.join(", ") || "none"
            }`,
          });
          return;
        }
        if (!res.data.length) {
          resolve({ ok: false, error: "CSV has headers but no data rows." });
          return;
        }
        const rows = res.data.map((r, i) => ({ ...r, id: i }));
        resolve({ ok: true, rows, columns: cols });
      },
      error: (err) =>
        resolve({ ok: false, error: `Could not parse CSV: ${err.message}` }),
    });
  });
}

function FileUpload({ onLoaded }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("");

  async function handleFile(file) {
    setError("");
    setFileName(file ? file.name : "");
    const result = await validateAndParse(file);
    if (!result.ok) {
      setError(result.error);
      return;
    }
    onLoaded(result.rows, result.columns, file);
  }

  return (
    <div>
      <div
        className={"dropzone" + (dragging ? " dragging" : "")}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFile(e.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click();
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          hidden
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
        <p className="dropzone-title">
          Drag &amp; drop a <code>.csv</code> file here
        </p>
        <p className="dropzone-sub">or click to browse</p>
        {fileName && <p className="dropzone-file">Selected: {fileName}</p>}
      </div>
      {error && <p className="form-error">{error}</p>}
    </div>
  );
}

export default FileUpload;
