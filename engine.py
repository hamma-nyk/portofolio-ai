import json
import math
import os
import pickle
import random
import re
import tempfile
import threading
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOXIC_WORDS = {
    "anjing", "babi", "bangsat", "tolol", "goblok", "bodoh", "tai", "jancok",
    "asu", "kampret", "bego", "sialan", "fuck", "shit", "idiot", "kontol",
    "memek", "bajingan", "keparat", "setan", "damn", "bastard", "stupid",
    "dumb", "useless", "penipu", "scammer", "jelek", "gila"
}

SLANG_MAP = {
    "gw": "saya", "gue": "saya", "aku": "saya", "lu": "kamu", "lo": "kamu",
    "yg": "yang", "dg": "dengan", "dgn": "dengan", "sm": "sama", "dr": "dari",
    "utk": "untuk", "krn": "karena", "tp": "tapi", "klo": "kalau", "kl": "kalau",
    "ga": "tidak", "gak": "tidak", "gk": "tidak", "nggak": "tidak", "tdk": "tidak",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah", "blm": "belum", "bs": "bisa",
    "gmn": "gimana", "bgmn": "gimana", "dmn": "dimana", "kpn": "kapan",
    "knp": "kenapa", "brp": "berapa", "thn": "tahun", "skrg": "sekarang",
    "bgt": "banget", "lg": "lagi", "msh": "masih", "pny": "punya",
    "bkn": "buat", "bikin": "buat", "liat": "lihat", "tny": "tanya", "nanya": "tanya",
    "wa": "whatsapp", "porto": "portofolio", "jarkom": "jaringan",
    "sosmed": "sosial media", "ngoding": "coding", "koding": "coding",
    "fe": "frontend", "be": "backend", "db": "database", "ml": "machine learning",
    "u": "you", "ur": "your", "r": "are", "pls": "please", "thx": "thanks", "ty": "thanks"
}

SYNONYM_MAP = {
    "skill": ["keahlian", "kemampuan", "mahir", "jago", "expertise", "capable", "ability", "spesialisasi", "kuasai"],
    "kerja": ["work", "job", "karir", "career", "posisi", "position", "rekrut", "hire"],
    "proyek": ["project", "portofolio", "portfolio", "karya", "showcase"],
    "kontak": ["contact", "hubungi", "reach", "telepon", "chat"],
    "network": ["jaringan", "networking", "net", "infra", "infrastruktur"],
    "coding": ["programming", "develop", "developer", "programmer", "software"],
    "harga": ["biaya", "tarif", "budget", "rate", "cost", "pricing", "bayar", "mahal", "murah"],
    "pengalaman": ["experience", "riwayat", "karir", "pernah", "tahun"],
}

KEYWORD_MAP = {
    "salam_general": ["halo", "hai", "hi", "hello", "hey", "pagi", "siang", "sore", "malam", "permisi", "tes", "ping", "morning", "afternoon", "evening", "yo"],
    "salam_islam": ["assalamualaikum", "assalamu'alaikum", "salam", "askum", "mikum", "samlekom", "assalam"],
    "identitas_diri": ["bot", "robot", "siapa kamu", "mesin", "manusia", "admin asli", "asia siapa", "who are you", "are you a bot", "real person"],
    "tentang_iko": ["siapa iko", "tentang iko", "profil iko", "about iko", "who is iko", "iko itu", "background", "bio", "owner", "pemilik web"],
    "status_kerja": ["open to work", "cari kerja", "rekrut", "hire", "available", "nganggur", "fulltime", "lowongan", "status kerja"],
    "pengalaman_kerja": ["pengalaman kerja", "pernah kerja", "riwayat pekerjaan", "work experience", "years of experience", "career history", "berapa tahun kerja"],
    "cv_resume": ["cv", "resume", "curriculum vitae", "download cv", "minta cv"],
    "skill_summary": ["skill", "keahlian", "kemampuan", "bisa apa", "tech stack", "expertise", "spesialisasi", "jago apa", "what can he do"],
    "skill_network_strict": ["network", "jaringan", "mikrotik", "ubiquiti", "unifi", "ruijie", "cisco", "fiber optic", "splicing", "vlan", "vpn", "firewall", "bandwidth", "cctv", "lan", "wan", "router", "switch"],
    "skill_backend_strict": ["backend", "server side", "nodejs", "node js", "express", "flask", "laravel", "php", "golang", "rest api", "api", "microservice"],
    "skill_frontend_strict": ["frontend", "front end", "react", "reactjs", "react native", "vue", "tailwind", "html", "css", "bootstrap", "figma", "responsive", "ui", "ux"],
    "skill_database_strict": ["database", "sql", "postgresql", "mysql", "mongodb", "nosql", "mariadb", "sql server", "erd", "query"],
    "skill_devops_server": ["linux", "ubuntu", "debian", "docker", "git", "github", "vps", "deploy", "nginx", "apache", "cloudflare", "proxmox", "devops", "hosting"],
    "skill_iot": ["iot", "internet of things", "arduino", "esp32", "esp8266", "mikrokontroler", "sensor", "smart home"],
    "skill_it_support_strict": ["it support", "troubleshoot", "komputer rusak", "laptop", "maintenance", "windows", "printer", "scanner", "helpdesk", "hardware", "rakit pc"],
    "skill_ai_strict": ["ai", "artificial intelligence", "machine learning", "chatbot", "nlp", "tensorflow", "kecerdasan buatan", "llm"],
    "skill_app_dev_general": ["aplikasi", "software", "app", "program", "development", "sistem informasi", "buat web", "bikin web", "buat aplikasi", "bikin aplikasi"],
    "skill_desktop_app": ["desktop", "electron", "tkinter", "exe", "gui", "aplikasi pc", "kasir desktop"],
    "development_workflow": ["workflow", "alur kerja", "sdlc", "proses", "cara kerja", "berapa lama", "how long"],
    "soft_skills": ["soft skill", "kelebihan", "why hire", "kenapa harus hire", "bahasa inggris", "english", "teamwork", "komunikasi", "problem solving"],
    "pendidikan_strict": ["pendidikan", "kuliah", "lulusan", "sarjana", "jurusan", "gelar", "universitas", "education", "degree", "university", "graduated", "almamater"],
    "tahunlulus_strict": ["lulus tahun", "wisuda", "angkatan", "graduation year", "when did he graduate", "fresh graduate"],
    "sertifikasi": ["sertifikat", "sertifikasi", "certificate", "lisensi", "mtcna", "bnsp", "course"],
    "proyek_strict": ["proyek", "project", "portofolio", "portfolio", "karya", "hasil", "showcase"],
    "jasa_strict": ["jasa", "freelance", "commission", "service", "terima project", "pesen web"],
    "harga_estimasi": ["harga", "biaya", "tarif", "rate card", "budget", "nego", "pricing", "cost", "how much"],
    "lokasi_strict": ["lokasi", "domisili", "tinggal", "kota", "alamat", "onsite", "remote", "relokasi", "location", "city", "domicile"],
    "ketersediaan_waktu": ["kapan online", "fast respon", "meeting", "zoom", "google meet", "jadwal", "when can he start", "schedule a call"],
    "kontak_strict": ["kontak", "contact", "whatsapp", "email", "linkedin", "sosial media", "hubungi", "telepon", "instagram", "github"],
    "tutorial_strict": ["tutorial", "ajarin", "cara hack", "spill", "bagi ilmu", "teach me"],
    "pujian_reaksi": ["keren", "mantap", "bagus", "hebat", "pintar", "canggih", "sip", "noted", "awesome", "cool", "great", "nice", "smart", "good job", "impressive"],
    "bantuan_menu": ["menu", "help", "panduan", "bisa bantu apa", "what can you do", "daftar topik", "commands"],
    "humor_santai": ["apa kabar", "gimana kabarnya", "lagi ngapain", "udah makan", "ngantuk", "bosan", "how are you", "what are you doing"],
    "bye": ["terima kasih", "makasih", "thanks", "thank you", "bye", "dadah", "sampai jumpa", "goodbye", "see you", "pamit"],
}

EN_MARKERS = {
    "what", "who", "how", "where", "when", "why", "can", "does", "do", "is",
    "are", "his", "he", "him", "your", "you", "tell", "me", "about", "please",
    "thanks", "thank", "hello", "hi", "hey", "good", "bye", "goodbye", "work",
    "skills", "projects", "contact", "experience", "resume", "pricing", "much"
}
ID_MARKERS = {
    "apa", "siapa", "gimana", "bagaimana", "dimana", "kapan", "kenapa", "bisa",
    "apakah", "dia", "kamu", "saya", "aku", "tentang", "tolong", "terima", "kasih",
    "makasih", "halo", "hai", "selamat", "kerja", "keahlian", "proyek", "kontak",
    "pengalaman", "harga", "biaya", "nya", "ga", "gak", "ya", "dong", "nih", "aja"
}

FOLLOWUP_TRIGGERS = {
    "terus", "lanjut", "detail", "detailnya", "lengkapnya", "ada lagi", "apa lagi",
    "contoh", "contohnya", "maksudnya", "jelasin", "more", "details", "what else",
    "elaborate", "tell me more", "example", "examples", "continue"
}

GREETING_TAGS = {"salam_general", "salam_islam"}

FALLBACK_ID = [
    "Hmm, saya belum menangkap maksudnya. Coba tanyakan tentang **skill, proyek, pengalaman, CV, atau kontak Iko** — atau ketik `menu` untuk daftar topik!",
    "Maaf, pertanyaan itu di luar cakupan saya. Saya **Asia Assistant**, spesialis info tentang **profil dan keahlian Iko**. Ketik `help` untuk melihat topik yang bisa ditanyakan.",
]
FALLBACK_EN = [
    "Hmm, I didn't quite catch that. Try asking about **Iko's skills, projects, experience, CV, or contact info** — or type `help` for a topic list!",
    "Sorry, that's outside my scope. I'm **Asia Assistant**, specialized in **Iko's portfolio and expertise**. Type `menu` to see what I can answer.",
]


def _leet_decode(text):
    table = str.maketrans("013457@", "oieasta")
    return text.translate(table)


def normalize_text(text, expand_slang=True):
    text = text.lower().strip()
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # haloooo -> haloo
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    if expand_slang:
        tokens = [SLANG_MAP.get(t, t) for t in tokens]
    return " ".join(tokens)


def detect_english(text):
    tokens = set(re.findall(r'\b[a-z]+\b', text.lower()))
    en_hits = len(tokens & EN_MARKERS)
    id_hits = len(tokens & ID_MARKERS)
    return en_hits > id_hits


def check_toxic(message):
    norm = normalize_text(message, expand_slang=False)
    decoded = _leet_decode(norm)
    for variant in (norm, decoded):
        tokens = set(variant.split())
        if tokens & TOXIC_WORDS:
            return True
    return False


class AsiaEngine:
    def __init__(self):
        self.intents_data = None
        self.intent_map = {}
        self.patterns = []        # list of (norm_pattern, tag)
        self.vocab = {}           # token -> idx
        self.idf = None           # np.array
        self.pattern_matrix = None  # normalized TF-IDF matrix (N x V)
        self.keras_model = None
        self.keras_words = None
        self.keras_classes = None
        self.lemmatizer = None
        self.nltk = None
        self.sessions = {}        # session_id -> {"last_tag": str, "seen": set}
        self._loaded = False
        self._keras_started = False
        self._load_lock = threading.Lock()

    def load(self):
        if self._loaded:
            return True
        with self._load_lock:
            if self._loaded:
                return True
            try:
                with open(os.path.join(BASE_DIR, "intents.json"), encoding="utf-8") as f:
                    self.intents_data = json.load(f)

                self.intent_map = {i["tag"]: i for i in self.intents_data["intents"]}
                self.patterns = []
                for intent in self.intents_data["intents"]:
                    for p in intent["patterns"]:
                        norm_p = normalize_text(p)
                        if norm_p:
                            self.patterns.append((norm_p, intent["tag"]))

                self._build_tfidf()
                self._loaded = True
                if not self._keras_started:
                    self._keras_started = True
                    threading.Thread(target=self._try_load_keras, daemon=True).start()
                return True
            except Exception as e:
                print(f"❌ Gagal memuat engine: {e}")
                return False

    def _tokenize_features(self, norm_text):
        words = norm_text.split()
        expanded = list(words)
        for base, syns in SYNONYM_MAP.items():
            for s in syns:
                if re.search(rf'\b{re.escape(s)}\b', norm_text):
                    expanded.append(base)
        ngrams = []
        padded = f" {norm_text} "
        for n in (3, 4):
            for i in range(max(0, len(padded) - n + 1)):
                ngrams.append(f"c_{padded[i:i+n]}")
        return expanded + ngrams

    def _build_tfidf(self):
        docs = [self._tokenize_features(p) for p, _ in self.patterns]
        df = Counter()
        for doc in docs:
            df.update(set(doc))

        self.vocab = {tok: idx for idx, tok in enumerate(df.keys())}
        n_docs = len(docs)
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for tok, idx in self.vocab.items():
            self.idf[idx] = math.log((1 + n_docs) / (1 + df[tok])) + 1.0

        mat = np.zeros((n_docs, len(self.vocab)), dtype=np.float32)
        for row, doc in enumerate(docs):
            counts = Counter(doc)
            for tok, cnt in counts.items():
                idx = self.vocab.get(tok)
                if idx is not None:
                    mat[row, idx] = (1.0 + math.log(cnt)) * self.idf[idx]
            norm = np.linalg.norm(mat[row])
            if norm > 0:
                mat[row] /= norm
        self.pattern_matrix = mat

    def _try_load_keras(self):
        try:
            import nltk
            from nltk.stem import WordNetLemmatizer
            from tensorflow.keras.models import load_model

            nltk_dir = os.path.join(tempfile.gettempdir(), "nltk_data")
            os.makedirs(nltk_dir, exist_ok=True)
            if nltk_dir not in nltk.data.path:
                nltk.data.path.insert(0, nltk_dir)

            for pkg in ("tokenizers/punkt", "tokenizers/punkt_tab", "corpora/wordnet"):
                try:
                    nltk.data.find(pkg)
                except LookupError:
                    nltk.download(pkg.split("/")[-1], download_dir=nltk_dir, quiet=True)

            self.nltk = nltk
            self.lemmatizer = WordNetLemmatizer()
            with open(os.path.join(BASE_DIR, "words.pkl"), "rb") as f:
                self.keras_words = pickle.load(f)
            with open(os.path.join(BASE_DIR, "classes.pkl"), "rb") as f:
                self.keras_classes = pickle.load(f)
            self.keras_model = load_model(os.path.join(BASE_DIR, "chatbot_model.h5"), compile=False)
        except Exception:
            self.keras_model = None

    def _tfidf_scores(self, norm_text):
        feats = self._tokenize_features(norm_text)
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        counts = Counter(feats)
        for tok, cnt in counts.items():
            idx = self.vocab.get(tok)
            if idx is not None:
                vec[idx] = (1.0 + math.log(cnt)) * self.idf[idx]
        norm = np.linalg.norm(vec)
        if norm == 0:
            return {}
        vec /= norm

        sims = self.pattern_matrix @ vec
        tag_scores = defaultdict(float)
        for idx, sim in enumerate(sims):
            tag = self.patterns[idx][1]
            if sim > tag_scores[tag]:
                tag_scores[tag] = float(sim)
        return tag_scores

    def _keyword_scores(self, norm_text):
        expanded = norm_text
        for base, syns in SYNONYM_MAP.items():
            for s in syns:
                if re.search(rf'\b{re.escape(s)}\b', norm_text):
                    expanded += f" {base}"

        scores = defaultdict(float)
        for tag, keywords in KEYWORD_MAP.items():
            raw = 0.0
            for kw in keywords:
                if re.search(rf'\b{re.escape(kw)}\b', expanded):
                    raw += 1.0 + 0.15 * len(kw.split())
            if raw > 0:
                scores[tag] = min(1.0, 0.45 + 0.25 * raw)
        return scores

    def _fuzzy_scores(self, norm_text):
        tag_scores = defaultdict(float)
        for pat, tag in self.patterns:
            ratio = SequenceMatcher(None, norm_text, pat).ratio()
            if ratio > tag_scores[tag]:
                tag_scores[tag] = ratio
        return tag_scores

    def _keras_scores(self, raw_text):
        if self.keras_model is None:
            return {}
        try:
            tokens = [self.lemmatizer.lemmatize(w.lower()) for w in self.nltk.word_tokenize(raw_text)]
            bag = np.array([[1 if w in tokens else 0 for w in self.keras_words]])
            probs = self.keras_model.predict(bag, verbose=0)[0]
            return {self.keras_classes[i]: float(p) for i, p in enumerate(probs) if p > 0.25}
        except Exception:
            return {}

    def _resolve_lang_tag(self, tag, is_en):
        base = tag[3:] if tag.startswith("en_") else tag
        if is_en and f"en_{base}" in self.intent_map:
            return f"en_{base}"
        if not is_en and base in self.intent_map:
            return base
        return tag

    def _pick_response(self, tag, session_id=None):
        intent = self.intent_map.get(tag)
        if not intent:
            return None
        responses = intent["responses"]
        if not session_id:
            return random.choice(responses)

        state = self.sessions.setdefault(session_id, {"last_tag": None, "seen": set()})
        unseen = [r for r in responses if (tag, r) not in state["seen"]]
        choice = random.choice(unseen if unseen else responses)
        state["last_tag"] = tag
        state["seen"].add((tag, choice))
        return choice

    def predict(self, message, session_id=None):
        self.load()
        is_en = detect_english(message)

        if check_toxic(message):
            tag = "en_kata_kasar" if is_en else "kata_kasar"
            return {"response": self._pick_response(tag, session_id), "type": "toxic", "intents": [tag], "confidence": 1.0}

        norm = normalize_text(message)
        if not norm:
            return {"response": random.choice(FALLBACK_EN if is_en else FALLBACK_ID), "type": "unknown", "intents": [], "confidence": 0.0}

        # Follow-up memory
        if session_id and norm in FOLLOWUP_TRIGGERS and self.sessions.get(session_id, {}).get("last_tag"):
            last_tag = self._resolve_lang_tag(self.sessions[session_id]["last_tag"], is_en)
            if last_tag not in GREETING_TAGS and last_tag not in {"en_salam_general", "en_salam_islam"}:
                return {"response": self._pick_response(last_tag, session_id), "type": "success", "intents": [last_tag], "confidence": 0.9}

        # Exact normalized match
        for pat, tag in self.patterns:
            if pat == norm:
                resolved = self._resolve_lang_tag(tag, is_en)
                return {"response": self._pick_response(resolved, session_id), "type": "success", "intents": [resolved], "confidence": 1.0}

        # Ensemble scoring (merged by base tag so ID & EN patterns reinforce each other)
        tfidf = self._tfidf_scores(norm)
        kw = self._keyword_scores(norm)
        fuzzy = self._fuzzy_scores(norm)
        keras = self._keras_scores(message)

        base_scores = defaultdict(float)
        all_tags = set(tfidf) | set(kw) | set(fuzzy) | set(keras)
        for tag in all_tags:
            base = tag[3:] if tag.startswith("en_") else tag
            score = (
                0.42 * tfidf.get(tag, 0.0)
                + 0.35 * kw.get(base, kw.get(tag, 0.0))
                + 0.15 * fuzzy.get(tag, 0.0)
                + 0.18 * keras.get(tag, 0.0)
            )
            if score > base_scores[base]:
                base_scores[base] = score

        ranked = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
        if not ranked or ranked[0][1] < 0.34:
            return {"response": random.choice(FALLBACK_EN if is_en else FALLBACK_ID), "type": "unknown", "intents": [], "confidence": round(ranked[0][1], 3) if ranked else 0.0}

        # Multi-intent: drop trivial greeting if a real question also scored well, combine up to 2 strong intents
        top_score = ranked[0][1]
        substantive = [(t, s) for t, s in ranked if t not in GREETING_TAGS and s >= max(0.45, top_score * 0.78)]
        chosen_bases = [t for t, _ in substantive[:2]] if substantive else [ranked[0][0]]

        resolved_tags = [self._resolve_lang_tag(b, is_en) for b in chosen_bases]
        replies = [self._pick_response(t, session_id) for t in resolved_tags]
        replies = [r for r in replies if r]

        return {
            "response": "\n\n".join(replies),
            "type": "success",
            "intents": resolved_tags,
            "confidence": round(top_score, 3),
        }


engine = AsiaEngine()
