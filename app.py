from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os

# Cek folder data NLTK standar
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Download hanya jika belum ada (Fallback)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Mendownload data NLTK...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Inisialisasi App
app = Flask(__name__)
CORS(app)  # Izinkan akses dari React

lemmatizer = WordNetLemmatizer()

# Load Data (Pastikan file ini ada di folder yang sama)
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

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
    data = request.get_json()
    message = data['message']
    
    # 1. Cek Toxic
    message_lower = message.lower()
    for word in toxic_words:
        if word in message_lower:
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

#if __name__ == '__main__':
#    app.run(port=5000, debug=True)