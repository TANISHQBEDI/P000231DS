const API_BASE = "http://localhost:5000";

export async function checkBackend() {

    const response = await fetch(
        `${API_BASE}/api/health`
    );

    return response.json();
}