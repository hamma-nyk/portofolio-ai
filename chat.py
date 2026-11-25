from engine import engine

if __name__ == "__main__":
    engine.load()
    print("---------------------------------------")
    print("Asia Assistant Siap! (Ketik 'quit' untuk keluar)")
    print("---------------------------------------")

    while True:
        message = input("Kamu: ").strip()
        if message.lower() in {"quit", "exit"}:
            break
        if not message:
            continue
        result = engine.predict(message, session_id="cli")
        print(f"Asia [{result['type']} | {', '.join(result['intents']) or '-'}]: {result['response']}")
