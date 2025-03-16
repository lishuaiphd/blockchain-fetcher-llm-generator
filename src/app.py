import argparse
import os

from flask import Flask, jsonify
from prometheus_flask_instrumentator import Instrumentator
from opentelemetry.instrumentation.flask import FlaskInstrumentor

from api.v1.generate import generate
from api.v1.optimize import prune, quantize
from api.v1.train import train

app = Flask(__name__)
Instrumentator().instrument(app).expose(app)
FlaskInstrumentor.instrument_app(app)

@app.route("/")
def home():
    return jsonify({"message": "API for JSON challenge response to Java signature verifier"})

app.register_blueprint(train, url_prefix="/train")
app.register_blueprint(generate, url_prefix="/generate")
app.register_blueprint(prune, url_prefix="/prune")
app.register_blueprint(quantize, url_prefix="/quantize")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API for JSON challenge response to Java signature verifier.")
    parser.add_argument("--port", type=int, help="Port number to run the server on")

    args = parser.parse_args()
    port = args.port or int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port, debug=True)