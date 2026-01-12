import os
import json
from gradio_client import Client

def check_api_and_save_schema():
    print("--- Checking IDM-VTON API Schema ---")
    log_file = "api_schema_log.txt"
    
    try:
        client = Client("yisol/IDM-VTON")
        info = client.view_api(all_endpoints=True, return_format="dict")
        
        with open(log_file, "w") as f:
            f.write(str(info))
            
        print(f"\n[SUCCESS] Connected to IDM-VTON. Schema saved to {log_file}")
    except Exception as e:
        print(f"\n[ERROR] Failed to connect to IDM-VTON: {e}")
        with open(log_file, "w") as f:
            f.write(f"Error: {str(e)}")

if __name__ == "__main__":
    check_api_and_save_schema()
