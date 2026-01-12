from gradio_client import Client
import json

SPACE = "Kwai-Kolors/Kolors-Virtual-Try-On"
TOKEN = "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"

output_file = "kolors_api_info.txt"

with open(output_file, "w") as f:
    f.write(f"Connecting to {SPACE}...\n")
    try:
        client = Client(SPACE, hf_token=TOKEN)
        f.write("Connected.\n\n")
        
        # Introspect endpoints
        f.write("--- Endpoints Inspection ---\n")
        if hasattr(client, "endpoints"):
            for i, endpoint in enumerate(client.endpoints):
                f.write(f"Fn Index {i}: {str(endpoint)}\n")
                try:
                    f.write(f"  Input Types: {endpoint.input_component_types}\n")
                    f.write(f"  Output Types: {endpoint.output_component_types}\n")
                    # Try to get labels usually hidden in serializers
                    # f.write(f"  Serializers: {endpoint.serializers}\n") 
                except:
                    pass
                f.write("\n")
        else:
            f.write("No .endpoints property found on client.\n")

    except Exception as e:
        f.write(f"Error: {e}\n")

print(f"Done. Saved to {output_file}")
