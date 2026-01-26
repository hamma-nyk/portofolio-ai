import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load data
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

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
    
    # --- LOGIKA KETAT (STRICT MODE) ---
    # Threshold 0.85 artinya bot harus 85% yakin.
    # Jika di bawah itu, dia akan dianggap tidak mengerti.
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

# --- LOOP UTAMA ---
print("---------------------------------------")
print("Bot Portofolio Budi Siap! (Ketik 'quit' untuk keluar)")
print("---------------------------------------")

while True:
    message = input("Kamu: ")
    if message.lower() == "quit":
        break
    
    # Prediksi Intent
    ints = predict_class(message, model)
    
    if len(ints) > 0:
        # Jika bot yakin (di atas threshold 0.85)
        res = get_response(ints, intents)
        print("Bot:", res)
    else:
        # Jika bot TIDAK yakin atau pertanyaan melenceng
        print("Bot: Maaf, pertanyaan Anda sepertinya di luar konteks Data Diri atau Skill saya.")
        print("Bot: Silakan tanya spesifik tentang Pengalaman, Keahlian IT, atau Kontak saya.")