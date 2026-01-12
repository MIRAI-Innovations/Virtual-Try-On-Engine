from gradio_client import Client

SPACES = [
    "Kwai-Kolors/Kolors-Virtual-Try-On",
    "smrasmy/IDM-VTON",
    "Nymbo/Virtual-Try-On"
]

for space in SPACES:
    print(f"\n--- Checking {space} ---")
    try:
        client = Client(space)
        client.view_api()
        print(f"Successfully connected to {space}")
    except Exception as e:
        print(f"Failed to connect to {space}: {e}")
