const API_BASE = "http://localhost:5000";


// Health check function to call the backend health endpoint
export async function checkBackend() {

    const response = await fetch(
        `${API_BASE}/api/health`
    );

    return response.json();
}