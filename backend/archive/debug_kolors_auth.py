from gradio_client import Client

SPACES = [
    "Kwai-Kolors/Kolors-Virtual-Try-On",
]

TOKEN = "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"

for space in SPACES:
    print(f"\n--- Checking {space} with TOKEN ---")
    try:
        client = Client(space, hf_token=TOKEN)
        client.view_api()
        print(f"Successfully connected to {space}")
    except Exception as e:
        print(f"Failed to connect to {space}: {e}")
