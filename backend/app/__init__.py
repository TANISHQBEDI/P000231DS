import sys
from pathlib import Path

from flask import Flask
from flask_cors import CORS

# Make the ML package importable from src/ without requiring an editable
# install (keeps `python backend/run.py` working out of the box).
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def create_app():
    app = Flask(__name__)
    CORS(
        app,
        resources={r"/api/*": {"origins": "http://localhost:5173"}},
        supports_credentials=True,
    )

    from app.routes.root import root_bp
    from app.routes.health import health_bp
    from app.routes.upload import upload_bp
    from app.routes.predict import predict_bp
    from app.routes.train import train_bp
    from app.routes.feedback import feedback_bp
    from app.routes.history import history_bp

    for bp in (
        root_bp,
        health_bp,
        upload_bp,
        predict_bp,
        train_bp,
        feedback_bp,
        history_bp,
    ):
        app.register_blueprint(bp)

    return app
