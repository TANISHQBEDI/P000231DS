import { useEffect, useState } from "react";
import { checkBackend } from "./services/api";

function App() {

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

            <h1>Maintenance NLP App</h1>

            <p>{message}</p>

        </div>
    );
}

export default App;