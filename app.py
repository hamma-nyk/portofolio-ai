import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
# HAPUS IMPORT load_model DI ATAS (Kita pindahkan ke bawah)
# from tensorflow.keras.models import load_model 

# ... (Bagian Konfigurasi Path & NLTK Tetap Sama) ...
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ... (Kode NLTK download tetap sama) ...

app = Flask(__name__)
CORS(app)
lemmatizer = WordNetLemmatizer()

# --- BAGIAN INI YANG DIUBAH (LAZY LOADING) ---
# Kita siapkan variabel kosong dulu
intents = None
words = None
classes = None
model = None

# Fungsi ini hanya dipanggil saat ada yang chat
def load_resources_if_needed():
    global intents, words, classes, model
    
    # Cek apakah model sudah ada di memori?
    if model is None:
        print("⚡ Memuat Model AI ke Memori (Hanya sekali)...")
        from tensorflow.keras.models import load_model  # Import di sini biar hemat RAM awal
        
        try:
            intents = json.loads(open(os.path.join(BASE_DIR, 'intents.json')).read())
            words = pickle.load(open(os.path.join(BASE_DIR, 'words.pkl'), 'rb'))
            classes = pickle.load(open(os.path.join(BASE_DIR, 'classes.pkl'), 'rb'))
            model = load_model(os.path.join(BASE_DIR, 'chatbot_model.h5'))
            print("✅ Model Sukses Dimuat!")
        except Exception as e:
            print(f"❌ Gagal memuat model: {e}")
            return False
    return True

# ... (List Toxic Words & Fungsi clean_up_sentence, bow TETAP SAMA) ...

# ... (Fungsi predict_class TETAP SAMA, tapi hapus argumen model karena kita pakai global) ...
def predict_class(sentence): # Hapus parameter model
    global model
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    # ... (sisanya sama) ...
    return return_list

# ... (Fungsi get_response TETAP SAMA) ...

@app.route('/chat', methods=['POST'])
def chat():
    # 1. LOAD MODEL DULU SEBELUM JAWAB
    if not load_resources_if_needed():
        return jsonify({"response": "Server sedang sibuk (Gagal load model).", "type": "error"}), 500

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"response": "Error: Pesan kosong.", "type": "error"}), 400

        message = data['message']
        
        # ... (Logika Cek Toxic TETAP SAMA) ...

        # 2. Prediksi AI (Panggil tanpa parameter model)
        ints = predict_class(message) # Cukup passing message
        
        if len(ints) > 0:
            res = get_response(ints, intents)
            return jsonify({"response": res, "type": "success"})
        else:
            return jsonify({"response": "Maaf, saya tidak mengerti.", "type": "unknown"})
            
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "Error internal server.", "type": "error"}), 500

# ... (Main block tetap sama) ...