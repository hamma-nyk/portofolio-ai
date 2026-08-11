import random
import json
import pickle
import re
import numpy as np
import nltk
from difflib import SequenceMatcher
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

pattern_index = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern_index.append((pattern.lower().strip(), intent['tag']))

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
    "status_kerja": ["open to work", "kerja", "rekrut", "hire", "available", "nganggur", "fulltime", "freelance"],
    "skill_network_strict": ["network", "jaringan", "mikrotik", "ubiquiti", "ruijie", "fiber optic", "vlan", "vpn", "firewall", "bandwidth", "cctv", "lan", "wan", "router", "switch"],
    "skill_backend_strict": ["backend", "server side", "nodejs", "node.js", "express", "flask", "laravel", "php", "rest api", "api", "microservice"],
    "skill_frontend_strict": ["frontend", "front end", "react", "vue", "tailwind", "html", "css", "bootstrap", "figma", "responsive", "ui", "ux"],
    "skill_database_strict": ["database", "sql", "postgresql", "mysql", "mongodb", "nosql", "mariadb", "sql server", "erd", "query"],
    "skill_it_support_strict": ["it support", "troubleshoot", "komputer rusak", "laptop", "maintenance", "windows", "printer", "scanner", "helpdesk", "hardware"],
    "skill_ai_strict": ["ai", "artificial intelligence", "machine learning", "chatbot", "nlp", "tensorflow", "kecerdasan buatan", "llm"],
    "pendidikan_strict": ["pendidikan", "kuliah", "lulusan", "sarjana", "jurusan", "gelar", "universitas", "education", "degree"],
    "proyek_strict": ["proyek", "project", "portofolio", "portfolio", "karya", "dibuat", "myctn"],
    "jasa_strict": ["jasa", "freelance", "commission", "rate card", "hire", "service"],
    "lokasi_strict": ["lokasi", "domisili", "tinggal", "kota", "alamat", "onsite", "remote", "relokasi", "location"],
    "kontak_strict": ["kontak", "contact", "whatsapp", "wa", "email", "linkedin", "sosmed", "hubungi"],
    "salam_general": ["halo", "hai", "hi", "hello", "hey", "pagi", "siang", "sore", "malam", "permisi", "tes", "ping"],
    "salam_islam": ["assalamualaikum", "salam", "askum", "mikum"],
    "identitas_diri": ["bot", "robot", "siapa kamu", "mesin", "manusia", "asia siapa", "who are you"],
    "kata_kasar": toxic_words,
    "bye": ["terima kasih", "makasih", "thanks", "bye", "dadah", "sampai jumpa", "goodbye"],
    "tutorial_strict": ["tutorial", "ajarin", "cara", "hack", "spill", "teach"],
}

SYNONYM_MAP = {
    "skill": ["keahlian", "kemampuan", "bisa", "jago", "mahir", "expertise"],
    "kerja": ["work", "job", "karir", "career"],
    "proyek": ["project", "portofolio", "portfolio", "karya"],
    "kontak": ["contact", "hubungi", "reach"],
    "network": ["jaringan", "networking", "net"],
    "coding": ["programming", "ngoding", "koding", "develop"],
}


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
    return False


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


def predict_class(sentence):
    sentence_lower = normalize_text(sentence)

    if check_toxic(sentence):
        for intent in intents['intents']:
            if intent['tag'] in ['kata_kasar', 'en_kata_kasar']:
                return [{"intent": intent['tag'], "probability": "1.0"}]

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if pattern.lower().strip() == sentence_lower:
                return [{"intent": intent['tag'], "probability": "1.0"}]

    best_score = 0
    best_tag = None
    for pattern, tag in pattern_index:
        score = SequenceMatcher(None, sentence_lower, pattern).ratio()
        if score > best_score:
            best_score = score
            best_tag = tag
    if best_score >= 0.78:
        return [{"intent": best_tag, "probability": str(best_score)}]

    expanded = expand_synonyms(sentence_lower)
    scores = {}
    for tag, keywords in KEYWORD_MAP.items():
        score = 0
        for kw in keywords:
            if kw in expanded:
                score += len(kw)
        if score > 0:
            scores[tag] = score
    if scores:
        best_tag = max(scores, key=scores.get)
        return [{"intent": best_tag, "probability": "0.85"}]

    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]), verbose=0)[0]
    ERROR_THRESHOLD = 0.55
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    if results:
        return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

    if best_score >= 0.55:
        return [{"intent": best_tag, "probability": str(best_score)}]

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
    "Hmm, saya belum bisa menjawab itu. Coba tanyakan tentang **skill, pengalaman, atau proyek Iko**!",
    "Maaf, pertanyaan itu di luar jangkauan saya. Saya Asia Assistant, spesialis info tentang **profil dan keahlian Iko**.",
    "Wah, saya kurang paham maksudnya. Tapi saya bisa bantu jawab tentang: **Skill Iko, Proyek, Pendidikan, Kontak, atau Jasa Freelance**.",
]

print("---------------------------------------")
print("Asia Assistant Siap! (Ketik 'quit' untuk keluar)")
print("---------------------------------------")

while True:
    message = input("Kamu: ")
    if message.lower() == "quit":
        break

    ints = predict_class(message)

    if ints:
        res = get_response(ints, intents)
        if res:
            print(f"Asia: {res}")
            continue

    print(f"Asia: {random.choice(FALLBACK_RESPONSES)}")
