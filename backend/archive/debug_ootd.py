import os
from gradio_client import Client

def check_ootd():
    print("--- Checking OOTDiffusion API Schema ---")
    log_file = "ootd_schema_log.txt"
    
    try:
        client = Client("levihsu/OOTDiffusion")
        info = client.view_api(all_endpoints=True, return_format="dict")
        
        with open(log_file, "w") as f:
            f.write(str(info))
            
        print(f"\n[SUCCESS] Connected to OOTDiffusion. Schema saved to {log_file}")
    except Exception as e:
        print(f"\n[ERROR] Failed to connect to OOTDiffusion: {e}")
        with open(log_file, "w") as f:
            f.write(f"Error: {str(e)}")

if __name__ == "__main__":
    check_ootd()
