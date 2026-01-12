from gradio_client import Client, handle_file
import os

SPACE = "Kwai-Kolors/Kolors-Virtual-Try-On"
TOKEN = "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"
IMG = r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\my photos\IMG_1337.jpg"
CLOTH = r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\datasets\test\cloth\00013_00.jpg"

print(f"Connecting to {SPACE}...")
try:
    client = Client(SPACE, hf_token=TOKEN)
    
    # Guessing the inputs: Person, Cloth, Seed?
    # Or Person, Cloth?
    # Trying fn_index 1
    print("Attempting predict on fn_index 1...")
    result = client.predict(
        handle_file(IMG),
        handle_file(CLOTH),
        0, # Seed
        True, # Random switch?
        fn_index=1
    )
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {e}")
