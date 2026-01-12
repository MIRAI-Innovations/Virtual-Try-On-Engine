from gradio_client import Client

SPACE = "Kwai-Kolors/Kolors-Virtual-Try-On"
TOKEN = "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"

print(f"Connecting to {SPACE}...")
client = Client(SPACE, hf_token=TOKEN)

try:
    print("Forcing error on fn_index=1...")
    client.predict(fn_index=1)
except Exception as e:
    print(f"\n--- API Signature Hint ---\n{e}")
