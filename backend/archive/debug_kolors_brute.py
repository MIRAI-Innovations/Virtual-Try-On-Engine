from gradio_client import Client, handle_file
import time

SPACE = "Kwai-Kolors/Kolors-Virtual-Try-On"
TOKEN = "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"
IMG = r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\my photos\IMG_1337.jpg"
CLOTH = r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\datasets\test\cloth\00013_00.jpg"

print(f"Connecting to {SPACE}...")
try:
    client = Client(SPACE, hf_token=TOKEN)
    
    # Define trial configurations
    trials = [
        {"fn_index": 0, "args": [handle_file(IMG), handle_file(CLOTH), 0, True], "desc": "Idx 0, 4 args (Img, Cloth, Seed, Rand)"},
        {"fn_index": 1, "args": [handle_file(IMG), handle_file(CLOTH), 0, True], "desc": "Idx 1, 4 args (Img, Cloth, Seed, Rand)"},
        {"fn_index": 2, "args": [handle_file(IMG), handle_file(CLOTH), 0, True], "desc": "Idx 2, 4 args (Img, Cloth, Seed, Rand)"},
        {"fn_index": 0, "args": [handle_file(IMG), handle_file(CLOTH)], "desc": "Idx 0, 2 args (Img, Cloth)"},
        {"fn_index": 1, "args": [handle_file(IMG), handle_file(CLOTH)], "desc": "Idx 1, 2 args (Img, Cloth)"},
        {"fn_index": 2, "args": [handle_file(IMG), handle_file(CLOTH)], "desc": "Idx 2, 2 args (Img, Cloth)"},
    ]

    for t in trials:
        print(f"\n--- Trying {t['desc']} ---")
        try:
            result = client.predict(*t['args'], fn_index=t['fn_index'])
            print(f"SUCCESS! Result: {result}")
            break
        except Exception as e:
            err = str(e)
            if "index out of range" in err:
                print("Failed: Index Error (Args mismatch?)")
            else:
                print(f"Failed: {err[:200]}") # Truncate

except Exception as e:
    print(f"Connection Error: {e}")
