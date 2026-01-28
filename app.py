import os
from flask import Flask, request, jsonify, render_template 
from flask_cors import CORS
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
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Mendownload data NLTK...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Inisialisasi App    
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
toxic_words = [
    "anjing", "babi", "bangsat", "tolol", "goblok", 
    "bodoh", "tai", "jancok", "asu", "kampret", "bego", 
    "sialan", "fuck", "shit", "idiot"
]
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return(np.array(bag))
    
# ... (Fungsi predict_class TETAP SAMA, tapi hapus argumen model karena kita pakai global) ...
def predict_class(sentence):
    # 1. --- LOGIKA BARU: EXACT MATCH (Jalan Pintas) ---
    # Jika input user SAMA PERSIS dengan pattern di JSON, langsung ambil!
    # Ini mengatasi masalah kata pendek seperti "Hi", "P", "Tes" yang sering kena filter threshold
    sentence_lower = sentence.lower().strip()
    
    # Kita butuh akses ke intents untuk cek manual
    # Pastikan variable 'intents' sudah di-load (global intents)
    if intents: 
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                if pattern.lower() == sentence_lower:
                    # Langsung kembalikan hasil dengan keyakinan 100%
                    return [{"intent": intent['tag'], "probability": "1.0"}]

    # 2. --- LOGIKA LAMA: AI PREDICTION ---
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    
    # Turunkan Threshold sedikit biar lebih ramah (0.75 cukup aman)
    ERROR_THRESHOLD = 0.75 
    
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# ... (Fungsi get_response TETAP SAMA) ...
def get_response(ints, intents_json):
    if not ints:
        return "Maaf, saya tidak mengerti maksud Anda."
        
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result
    
@app.route('/', methods=['GET'])
def home():
    # Flask otomatis mencari file 'index.html' di dalam folder 'templates'
    return render_template('index.html')
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

# Untuk local testing saja. Saat deploy, Gunicorn yang handle.
if __name__ == '__main__':
    app.run(port=5000, debug=True)