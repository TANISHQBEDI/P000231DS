import { BrowserRouter, Routes, Route, Link } from "react-router-dom";

// Import page components
import Home from "./pages/Home";
import Health from "./pages/Health";

function App() {
    return (
        <BrowserRouter>

            {/* Simple nav shell */}
            <nav style={{ padding: "1rem", borderBottom: "1px solid #ccc" }}>
                <Link to="/" style={{ marginRight: "1rem" }}>Home</Link>
                <Link to="/health">Health</Link>
            </nav>

            {/* Page routing */}
            <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/health" element={<Health />} />
            </Routes>

        </BrowserRouter>
    );
}

export default App;