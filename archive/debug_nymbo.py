from gradio_client import Client

SPACE = "Nymbo/Virtual-Try-On"
TOKEN = "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"

print(f"Connecting to {SPACE}...")
try:
    client = Client(SPACE, hf_token=TOKEN)
    client.view_api()
    print("Success")
except Exception as e:
    print(f"Error: {e}")
