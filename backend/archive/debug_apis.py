
from gradio_client import Client

token = "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"

print("--- Kwai-Kolors/Kolors-Virtual-Try-On ---")
try:
    client = Client("Kwai-Kolors/Kolors-Virtual-Try-On", hf_token=token)
    client.view_api()
except Exception as e:
    print(f"Error inspecting Kolors: {e}")

print("\n--- levihsu/OOTDiffusion ---")
try:
    client = Client("levihsu/OOTDiffusion", hf_token=token)
    client.view_api()
except Exception as e:
    print(f"Error inspecting OOTDiffusion: {e}")
