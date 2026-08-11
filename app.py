import os
import re
from difflib import SequenceMatcher
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

API_KEY = os.getenv("API_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://www.ikonyek.my.id", "https://www.ikonyek.vercel.app", "https://ikonyek.vercel.app", "http://localhost:5173"]}})
lemmatizer = WordNetLemmatizer()

intents = None
words = None
classes = None
model = None
pattern_index = None

toxic_words = [
    "anjing", "babi", "bangsat", "tolol", "goblok",
    "bodoh", "tai", "jancok", "asu", "kampret", "bego",
    "sialan", "fuck", "shit", "idiot", "kontol", "memek",
    "bajingan", "keparat", "setan", "damn", "bastard",
    "stupid", "dumb", "useless", "penipu", "scammer"
]

KEYWORD_MAP = {
    "skill_summary": ["skill", "keahlian", "kemampuan", "bisa apa", "tech stack", "expertise", "capable", "spesialisasi"],
    "tentang_iko": ["siapa iko", "tentang iko", "profil iko", "about iko", "who is iko", "iko itu", "background", "bio"],
    "status_kerja": ["open to work", "kerja", "rekrut", "hire", "available", "nganggur", "fulltime", "freelance", "lowongan"],
    "skill_network_strict": ["network", "jaringan", "mikrotik", "ubiquiti", "ruijie", "fiber optic", "vlan", "vpn", "firewall", "bandwidth", "cctv", "lan", "wan", "router", "switch"],
    "skill_backend_strict": ["backend", "server side", "nodejs", "node.js", "express", "flask", "laravel", "php", "rest api", "api", "microservice"],
    "skill_frontend_strict": ["frontend", "front end", "react", "vue", "tailwind", "html", "css", "bootstrap", "figma", "responsive", "ui", "ux"],
    "skill_database_strict": ["database", "sql", "postgresql", "mysql", "mongodb", "nosql", "mariadb", "sql server", "erd", "query"],
    "skill_it_support_strict": ["it support", "troubleshoot", "komputer rusak", "laptop", "maintenance", "windows", "printer", "scanner", "helpdesk", "hardware"],
    "skill_ai_strict": ["ai", "artificial intelligence", "machine learning", "chatbot", "nlp", "tensorflow", "kecerdasan buatan", "llm"],
    "skill_app_dev_general": ["aplikasi", "software", "app", "program", "development", "sistem informasi"],
    "skill_desktop_app": ["desktop", "electron", "tkinter", "exe", "gui", "aplikasi pc"],
    "development_workflow": ["workflow", "alur kerja", "sdlc", "proses", "cara kerja", "berapa lama"],
    "pendidikan_strict": ["pendidikan", "kuliah", "lulusan", "sarjana", "jurusan", "gelar", "universitas", "education", "degree", "university", "graduated"],
    "tahunlulus_strict": ["lulus tahun", "wisuda", "angkatan", "graduation year"],
    "sertifikasi": ["sertifikat", "sertifikasi", "certificate", "lisensi", "mtcna", "bnsp", "course"],
    "proyek_strict": ["proyek", "project", "portofolio", "portfolio", "karya", "hasil", "dibuat", "myctn", "whatsapp bot"],
    "jasa_strict": ["jasa", "freelance", "commission", "rate card", "hire", "terima", "service"],
    "lokasi_strict": ["lokasi", "domisili", "tinggal", "kota", "alamat", "onsite", "remote", "relokasi", "location", "city"],
    "kontak_strict": ["kontak", "contact", "whatsapp", "wa", "email", "linkedin", "sosmed", "hubungi", "telepon", "hp"],
    "salam_general": ["halo", "hai", "hi", "hello", "hey", "pagi", "siang", "sore", "malam", "permisi", "tes", "ping"],
    "salam_islam": ["assalamualaikum", "salam", "askum", "mikum", "samlekom"],
    "identitas_diri": ["bot", "robot", "siapa kamu", "mesin", "manusia", "admin asli", "asia siapa", "who are you", "are you"],
    "kata_kasar": toxic_words,
    "bye": ["terima kasih", "makasih", "thanks", "bye", "dadah", "sampai jumpa", "goodbye", "see you", "pamit"],
    "tutorial_strict": ["tutorial", "ajarin", "cara", "gimana", "hack", "spill", "bagi ilmu", "teach"],
}

SYNONYM_MAP = {
    "skill": ["keahlian", "kemampuan", "bisa", "jago", "mahir", "expertise", "capable", "ability"],
    "kerja": ["work", "job", "karir", "career", "posisi", "position"],
    "proyek": ["project", "portofolio", "portfolio", "karya"],
    "kontak": ["contact", "hubungi", "reach"],
    "network": ["jaringan", "networking", "net"],
    "coding": ["programming", "ngoding", "koding", "develop"],
    "bisa": ["mampu", "sanggup", "capable", "can", "able"],
}


def load_resources_if_needed():
    global intents, words, classes, model, pattern_index

    if model is None:
        print("⚡ Memuat Model AI ke Memori (Hanya sekali)...")
        from tensorflow.keras.models import load_model as tf_load_model

        try:
            intents_data = json.loads(open(os.path.join(BASE_DIR, 'intents.json')).read())
            intents = intents_data
            words = pickle.load(open(os.path.join(BASE_DIR, 'words.pkl'), 'rb'))
            classes = pickle.load(open(os.path.join(BASE_DIR, 'classes.pkl'), 'rb'))
            model = tf_load_model(os.path.join(BASE_DIR, 'chatbot_model.h5'))

            pattern_index = []
            for intent in intents['intents']:
                for pattern in intent['patterns']:
                    pattern_index.append((pattern.lower().strip(), intent['tag']))

            print("✅ Model Sukses Dimuat!")
        except Exception as e:
            print(f"❌ Gagal memuat model: {e}")
            return False
    return True


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def expand_synonyms(text):
    expanded = text
    for base, syns in SYNONYM_MAP.items():
        for syn in syns:
            if syn in text:
                expanded += " " + base
    return expanded


def check_toxic(message):
    msg_lower = normalize_text(message)
    msg_words = msg_lower.split()
    for word in msg_words:
        if word in toxic_words:
            return True
    for toxic in toxic_words:
        if toxic in msg_lower:
            return True
    return False


def exact_match(sentence_lower):
    if not intents:
        return None
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if pattern.lower().strip() == sentence_lower:
                return [{"intent": intent['tag'], "probability": "1.0"}]
    return None


def fuzzy_match(sentence_lower, threshold=0.75):
    best_score = 0
    best_tag = None
    for pattern, tag in pattern_index:
        score = SequenceMatcher(None, sentence_lower, pattern).ratio()
        if score > best_score:
            best_score = score
            best_tag = tag
    if best_score >= threshold:
        return [{"intent": best_tag, "probability": str(best_score)}]
    return None


def keyword_match(sentence_lower):
    expanded = expand_synonyms(sentence_lower)
    scores = {}
    for tag, keywords in KEYWORD_MAP.items():
        score = 0
        for kw in keywords:
            if kw in expanded:
                score += len(kw)
        if score > 0:
            scores[tag] = score

    if not scores:
        return None

    best_tag = max(scores, key=scores.get)

    if best_tag.startswith("en_"):
        pass

    has_english = any(c in sentence_lower for c in ["what", "who", "how", "where", "when", "can", "does", "is ", "are "])
    if has_english:
        en_tag = "en_" + best_tag
        for intent in intents['intents']:
            if intent['tag'] == en_tag:
                return [{"intent": en_tag, "probability": "0.85"}]

    return [{"intent": best_tag, "probability": "0.85"}]


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words_list, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words_list)
    for s in sentence_words:
        for i, w in enumerate(words_list):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def ai_predict(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.55
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def predict_class(sentence):
    sentence_lower = normalize_text(sentence)

    if check_toxic(sentence):
        for intent in intents['intents']:
            if intent['tag'] in ['kata_kasar', 'en_kata_kasar']:
                return [{"intent": intent['tag'], "probability": "1.0"}]

    result = exact_match(sentence_lower)
    if result:
        return result

    result = fuzzy_match(sentence_lower, threshold=0.78)
    if result:
        return result

    result = keyword_match(sentence_lower)
    if result:
        return result

    ai_result = ai_predict(sentence)
    if ai_result:
        return ai_result

    result = fuzzy_match(sentence_lower, threshold=0.55)
    if result:
        return result

    return []


def get_response(ints, intents_json):
    if not ints:
        return None

    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return None


FALLBACK_RESPONSES = [
    "Hmm, saya belum bisa menjawab itu. Coba tanyakan tentang **skill, pengalaman, atau proyek Iko** — itu spesialisasi saya!",
    "Maaf, pertanyaan itu di luar jangkauan saya. Saya Asia Assistant, spesialis info tentang **profil dan keahlian Iko**. Coba tanya hal lain!",
    "Wah, saya kurang paham maksudnya. Tapi saya bisa bantu jawab tentang: **Skill Iko, Proyek, Pendidikan, Kontak, atau Jasa Freelance**.",
    "Pertanyaan menarik, tapi saya hanya tahu tentang Iko. Coba tanya: 'Apa skill Iko?' atau 'Siapa Iko?'",
]


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    client_key = request.headers.get('X-Secret-Token')
    if client_key != API_KEY:
        return jsonify({"response": "Error: Kunci API tidak valid.", "type": "error"}), 401

    if not load_resources_if_needed():
        return jsonify({"response": "Server sedang sibuk (Gagal load model).", "type": "error"}), 500

    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"response": "Error: Pesan kosong.", "type": "error"}), 400

        message = data['message'].strip()
        if not message:
            return jsonify({"response": "Error: Pesan kosong.", "type": "error"}), 400

        if len(message) > 500:
            return jsonify({"response": "Pesan terlalu panjang. Coba lebih singkat ya!", "type": "error"}), 400

        ints = predict_class(message)

        if ints:
            res = get_response(ints, intents)
            if res:
                return jsonify({"response": res, "type": "success"})

        return jsonify({"response": random.choice(FALLBACK_RESPONSES), "type": "unknown"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"response": "Error internal server.", "type": "error"}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
