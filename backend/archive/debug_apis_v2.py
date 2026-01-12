from gradio_client import Client
import os

token = "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"

print("--- OOTD API ---")
try:
    client = Client("levihsu/OOTDiffusion", hf_token=token)
    print(client.view_api(return_format="str"))
except Exception as e:
    print(e)

print("\n--- Kolors API ---")
try:
    client = Client("Kwai-Kolors/Kolors-Virtual-Try-On", hf_token=token)
    print(client.view_api(return_format="str"))
except Exception as e:
    print(e)
