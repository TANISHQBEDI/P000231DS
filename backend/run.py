# This is the entry point for the Flask App. It initializes the app and defines the routes.

from app import create_app
from flask_cors import CORS

app = create_app()

if __name__ == "__main__":
    # Run the Flask app in debug mode.
    app.run(debug=True)
