from gradio_client import Client
import json

SPACE = "Kwai-Kolors/Kolors-Virtual-Try-On"
TOKEN = "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"

print(f"Connecting to {SPACE}...")
try:
    client = Client(SPACE, hf_token=TOKEN)
    print("Connected.")
    
    # Introspect endpoints
    print("\n--- Endpoints Inspection ---")
    if hasattr(client, "endpoints"):
        for i, endpoint in enumerate(client.endpoints):
            print(f"Fn Index {i}: {str(endpoint)}")
            try:
                # deeper info
                print(f"  Input: {endpoint.input_component_types}")
                print(f"  Output: {endpoint.output_component_types}")
            except:
                pass
    else:
        print("No .endpoints property found on client.")

    # Try view_api again
    print("\n--- API View ---")
    client.view_api(all_endpoints=True)

except Exception as e:
    print(f"Error: {e}")
