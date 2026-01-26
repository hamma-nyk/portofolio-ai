import requests
import json
import time

# --- KONFIGURASI URL ---
# Ganti dengan URL Render kamu jika sudah deploy
# URL = "https://nama-project-kamu.onrender.com/chat"
# URL = "https://portofolio-ai.onrender.com/chat"  # Localhost
URL = "http://127.0.0.1:5000/chat"
def send_message(message):
    payload = {"message": message}
    headers = {"Content-Type": "application/json"}
    
    try:
        start_time = time.time()
        response = requests.post(URL, json=payload, headers=headers)
        duration = round(time.time() - start_time, 2)
        
        if response.status_code == 200:
            data = response.json()
            return data['response'], data['type'], duration
        else:
            return f"Error {response.status_code}", "error", duration
    except Exception as e:
        return f"Connection Refused. Pastikan server nyala! ({e})", "error", 0

def mode_manual():
    print(f"\n🚀 Memulai Test Chat ke: {URL}")
    print("Ketik 'quit' untuk keluar.\n")
    
    while True:
        user_input = input("Kamu: ")
        if user_input.lower() in ['quit', 'exit']:
            break
            
        bot_reply, msg_type, duration = send_message(user_input)
        
        # Pewarnaan output sederhana
        type_indicator = "[TOXIC DETECTED]" if msg_type == 'toxic' else ""
        print(f"Bot : {bot_reply} {type_indicator}")
        print(f"      (Latency: {duration}s)\n")

def mode_auto_test():
    print(f"\n🤖 Memulai Automated Test ke: {URL}\n")
    
    test_cases = [
        "Halo Budi",                    # Salam
        "Apa skill backend kamu?",      # Skill
        "Dasar anjing lu",              # Toxic (Harus diblokir)
        "Cara masak nasi goreng",       # Unknown (Harus ditolak)
        "Kamu lulusan mana?",           # Pendidikan
    ]
    
    for msg in test_cases:
        print(f"Input : '{msg}'")
        bot_reply, msg_type, duration = send_message(msg)
        print(f"Output: {bot_reply}")
        print(f"Type  : {msg_type} | Time: {duration}s")
        print("-" * 40)

if __name__ == "__main__":
    print("Pilih Mode Testing:")
    print("1. Chat Manual (Ketik sendiri)")
    print("2. Auto Test (Cek semua skenario)")
    pilihan = input("Pilih (1/2): ")
    
    if pilihan == "1":
        mode_manual()
    else:
        mode_auto_test()