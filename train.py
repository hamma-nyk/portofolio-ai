import json
import numpy as np
import nltk
import pickle
import random
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Download resource NLTK (cukup sekali jalan)
nltk.download('punkt')
nltk.download('punkt_tab')  # <--- INI YANG KURANG
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# 1. Load Data
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

data_file = open('intents.json').read()
intents = json.loads(data_file)

# 2. Preprocessing (Tokenization & Lemmatization)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Pecah kalimat jadi kata
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Simpan pasangan (kata, tag)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize (mengubah kata ke bentuk dasar) dan hapus duplikat
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Simpan data words dan classes untuk dipanggil saat chat nanti
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# 3. Siapkan Training Data (Bag of Words)
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Tandai 1 jika kata ada di kamus, 0 jika tidak
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Acak data dan ubah ke array numpy
random.shuffle(training)
training = np.array(training, dtype=object)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

# 4. Buat Model Neural Network
model = Sequential()
# Layer Input: 128 neuron, input shape sesuai jumlah kata unik
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
# Layer Hidden
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
# Layer Output: sesuai jumlah kategori intent (softmax untuk probabilitas)
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile Model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# 5. Training
print("Mulai training model...")
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Simpan Model
model.save('chatbot_model.h5')
print("Model berhasil dibuat dan disimpan!")