from gradio_client import Client

SPACE = "Kwai-Kolors/Kolors-Virtual-Try-On"
TOKEN = "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"

print(f"Connecting to {SPACE}...")
try:
    client = Client(SPACE, hf_token=TOKEN)
    print("Connected.")
    
    # Print raw endpoints usage
    print("\n--- API View ---")
    client.view_api(all_endpoints=True)

except Exception as e:
    print(f"Error: {e}")
