import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# --- KONFIGURASI PATH (PENTING UNTUK DEPLOY) ---
# Ini memastikan file ditemukan dimanapun Gunicorn dijalankan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- SETUP NLTK ---
# Download data NLTK secara diam-diam (quiet=True) agar log bersih
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

# --- LOAD DATA DENGAN ABSOLUTE PATH ---
try:
    print("Loading model dan data...")
    intents = json.loads(open(os.path.join(BASE_DIR, 'intents.json')).read())
    words = pickle.load(open(os.path.join(BASE_DIR, 'words.pkl'), 'rb'))
    classes = pickle.load(open(os.path.join(BASE_DIR, 'classes.pkl'), 'rb'))
    model = load_model(os.path.join(BASE_DIR, 'chatbot_model.h5'))
    print("Model berhasil di-load!")
except Exception as e:
    print(f"CRITICAL ERROR: Gagal load file: {e}")
    # Biarkan error raise agar kita tahu di logs render jika file hilang

# List Kata Kasar
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

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    
    ERROR_THRESHOLD = 0.85
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

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

# --- API ENDPOINT ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"response": "Error: Pesan tidak boleh kosong.", "type": "error"}), 400

        message = data['message']
        
        # 1. Cek Toxic (LOGIKA DIPERBAIKI)
        # Tokenize dulu agar "lantai" tidak terdeteksi sebagai "tai"
        # Kita cek apakah ada kata kasar di dalam list kata-kata user
        message_words = nltk.word_tokenize(message.lower())
        
        is_toxic = False
        for word in message_words:
            if word in toxic_words:
                is_toxic = True
                break
                
        if is_toxic:
            return jsonify({
                "response": "Maaf, tolong gunakan bahasa yang sopan. Saya tidak akan merespon kata-kata kasar.",
                "type": "toxic"
            })

        # 2. Prediksi AI
        ints = predict_class(message, model)
        
        if len(ints) > 0:
            res = get_response(ints, intents)
            return jsonify({
                "response": res,
                "type": "success"
            })
        else:
            return jsonify({
                "response": "Maaf, pertanyaan Anda sepertinya di luar konteks Data Diri atau Skill saya.",
                "type": "unknown"
            })
            
    except Exception as e:
        print(f"Error pada endpoint /chat: {e}")
        return jsonify({
            "response": "Maaf, terjadi kesalahan internal pada server.",
            "type": "error"
        }), 500

# Untuk local testing saja. Saat deploy, Gunicorn yang handle.
if __name__ == '__main__':
    app.run(port=5000, debug=True)