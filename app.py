import os
from flask import Flask, request, jsonify, render_template
from engine import engine

try:
    from flask_cors import CORS
except ImportError:
    CORS = None

API_KEY = os.getenv("API_KEY")
ALLOWED_ORIGINS = [
    "https://www.ikonyek.my.id",
    "https://www.ikonyek.vercel.app",
    "https://ikonyek.vercel.app",
    "http://localhost:5173",
]

app = Flask(__name__)
if CORS:
    CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    ok = engine.load()
    return jsonify({"status": "ok" if ok else "error", "keras_loaded": engine.keras_model is not None}), (200 if ok else 500)


@app.route("/chat", methods=["POST"])
def chat():
    if request.headers.get("X-Secret-Token") != API_KEY:
        return jsonify({"response": "Error: Kunci API tidak valid.", "type": "error"}), 401

    if not engine.load():
        return jsonify({"response": "Server sedang sibuk (Gagal load model).", "type": "error"}), 500

    try:
        data = request.get_json(silent=True) or {}
        message = str(data.get("message", "")).strip()
        if not message:
            return jsonify({"response": "Error: Pesan kosong.", "type": "error"}), 400
        if len(message) > 500:
            return jsonify({"response": "Pesan terlalu panjang. Coba lebih singkat ya!", "type": "error"}), 400

        session_id = data.get("session_id") or request.remote_addr
        result = engine.predict(message, session_id=session_id)
        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "Error internal server.", "type": "error"}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
