// Health.jsx - backend connectivity check.

import { useEffect, useState } from "react";
import { checkBackend } from "../services/api";

function Health() {
  const [message, setMessage] = useState("Checking…");
  const [ok, setOk] = useState(null);

  useEffect(() => {
    checkBackend()
      .then((data) => {
        setMessage(data.status);
        setOk(true);
      })
      .catch(() => {
        setMessage("Backend connection failed");
        setOk(false);
      });
  }, []);

  return (
    <div className="page">
      <header className="page-head">
        <h1>System Health</h1>
      </header>
      <section className="card">
        <p>
          Backend status:{" "}
          <span className={ok ? "conf-ok" : ok === false ? "conf-low" : ""}>
            {message}
          </span>
        </p>
      </section>
    </div>
  );
}

export default Health;
