import os
import shutil
import time
from gradio_client import Client, handle_file

# Configuration
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define tasks: (person_image_path, cloth_image_path, output_filename)
TASKS = [
    (
        r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\my photos\IMG_1337.jpg",
        r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\datasets\test\cloth\00013_00.jpg",
        "final_output.jpg"
    ),
    (
        r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\my photos\IMG_1522.jpg",
        r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\datasets\test\cloth\00013_00.jpg",
        "test_output_1522.jpg"
    )
]

# List of spaces to try in order of preference/stability
SPACES = [
    {
        "name": "levihsu/OOTDiffusion",
        "type": "ootd",
        "notes": "Highly stable, usually works."
    },
    {
        "name": "zhengchong/CatVTON", 
        "type": "catvton",
        "notes": "Fast, but had downtime recently."
    },
    {
        "name": "yisol/IDM-VTON",
        "type": "idm",
        "notes": "Original SOTA, frequent runtime errors."
    }
]

def run_prediction(client, space_config, person_path, cloth_path):
    space_name = space_config["name"]
    space_type = space_config["type"]
    
    print(f"  Attempting prediction with {space_name} ({space_type})...")
    
    if space_type == "ootd":
        # OOTDiffusion usually has /process_hd for half-body or /process_dc for full/dress
        # We will try /process_hd (half-body) as it's safer for general upper-body images.
        return client.predict(
            vton_img=handle_file(person_path),
            garm_img=handle_file(cloth_path),
            n_samples=1,
            n_steps=20,
            image_scale=2,
            seed=42,
            api_name="/process_hd" 
        )
    
    elif space_type == "catvton":
        return client.predict(
            person_image=handle_file(person_path),
            cloth_image=handle_file(cloth_path),
            cloth_type="upper",
            num_steps=40,
            guidance_scale=2.5,
            seed=42,
            show_type="result only",
            api_name="/submit"
        )
        
    elif space_type == "idm":
         return client.predict(
            dict={"background": handle_file(person_path), "layers": [], "composite": None},
            garm_img=handle_file(cloth_path),
            garment_des="A cool garment",
            is_checked=True, 
            is_checked_crop=False, 
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )
    return None

def run_vton():
    active_client = None
    active_config = None

    # 1. Connect to a working space
    for config in SPACES:
        name = config["name"]
        print(f"Trying to connect to {name}...")
        try:
            client = Client(name)
            # Simple check if api is viewable (means loaded)
            client.view_api(return_format="dict")
            print(f"Successfully connected to {name}!")
            active_client = client
            active_config = config
            break
        except Exception as e:
            print(f"Failed to connect to {name}: {e}")
            continue
    
    if not active_client:
        print("\nCRITICAL: All VTON spaces are currently down or unreachable.")
        print("Please try again later or check your internet connection.")
        return

    # 2. Process Tasks
    print(f"\nProcessing tasks using {active_config['name']}...")
    
    for person_path, cloth_path, output_name in TASKS:
        if not os.path.exists(person_path):
            print(f"Skipping {output_name}: Person image not found at {person_path}")
            continue
        if not os.path.exists(cloth_path):
            print(f"Skipping {output_name}: Cloth image not found at {cloth_path}")
            continue

        print(f"Processing {output_name}...")
        
        try:
            # Run prediction with the active client
            # Note: OOTD returns a list of result images usually
            result = run_prediction(active_client, active_config, person_path, cloth_path)
            
            # Handle results
            final_image = None
            
            # Unpack result if it's a list/tuple
            if isinstance(result, (list, tuple)):
                # OOTDiffusion returns a list of file paths usually
                for item in result:
                    if isinstance(item, str) and os.path.exists(item):
                        final_image = item
                        break
                    elif isinstance(item, dict) and 'image' in item:
                        final_image = item['image']
                        break
            elif isinstance(result, str) and os.path.exists(result):
                final_image = result

            if final_image:
                output_full_path = os.path.join(OUTPUT_DIR, output_name)
                shutil.copy(final_image, output_full_path)
                print(f"  Success! Saved to {output_full_path}\n")
            else:
                print(f"  Failed: API returned Unknown format: {result}")

        except Exception as e:
            print(f"  Failed to process {output_name}: {e}")
            try:
                print("  API Details for debugging:")
                active_client.view_api()
            except:
                pass

if __name__ == "__main__":
    run_vton()
