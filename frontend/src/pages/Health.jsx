// Health.jsx
// Routes to the health check page

import { useEffect, useState } from "react";
import { checkBackend } from "../services/api";

function Health() {

    const [message, setMessage] = useState("");

    useEffect(() => {

        async function fetchData() {

            try {

                const data = await checkBackend();

                setMessage(data.status);

            } catch (error) {

                setMessage("Backend connection failed");
            }
        }

        fetchData();

    }, []);

    return (
        <div style={{ padding: "2rem" }}>

            <h1>Health Check</h1>

            <p>{message}</p>

        </div>
    );
}

export default Health;