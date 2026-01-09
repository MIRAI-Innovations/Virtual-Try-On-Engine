"""
Standalone Virtual Try-On Engine
Uses Hugging Face Gradio API for VTON inference (Intel Arc GPU compatible)
"""

import argparse
import os
import time
from pathlib import Path
from PIL import Image
from gradio_client import Client, handle_file

# Configuration
HF_TOKEN = "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"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
MAX_IMAGE_SIZE = 1024

def resize_image(image_path, max_size=MAX_IMAGE_SIZE):
    """
    Resize image to max dimension while maintaining aspect ratio.
    Returns path to temporary resized file.
    """
    try:
        img = Image.open(image_path)
        
        # Check if resizing is needed
        if max(img.size) <= max_size:
            return image_path
        
        # Calculate new size maintaining aspect ratio
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        
        # Resize
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Save to temp file
        temp_path = f"temp_resized_{Path(image_path).name}"
        img_resized.save(temp_path)
        
        print(f"[Resize] {Path(image_path).name}: {img.size} → {new_size}")
        return temp_path
    
    except Exception as e:
        print(f"[Resize] Error: {e}, using original image")
        return image_path

def run_tryon(person_path, cloth_path):
    """
    Run VTON inference with retry logic.
    Tries multiple API spaces with automatic fallback.
    """
    # Resize images first
    print("[VTON] Preparing images...")
    person_resized = resize_image(person_path)
    cloth_resized = resize_image(cloth_path)
    
    # API spaces to try (in order of preference)
    api_spaces = [
        {
            "name": "Kwai-Kolors/Kolors-Virtual-Try-On",
            "api_name": None,
            "params": lambda: [
                handle_file(person_resized),
                handle_file(cloth_resized)
            ]
        },
        {
            "name": "yisol/IDM-VTON",
            "api_name": "/tryon",
            "params": lambda: {
                "dict_val": {
                    "background": handle_file(person_resized),
                    "layers": [],
                    "composite": None
                },
                "garm_img": handle_file(cloth_resized),
                "garment_des": "clothing",
                "is_checked": True,
                "is_checked_crop": False,
                "denoise_steps": 30,
                "seed": 42
            }
        },
        {
            "name": "levihsu/OOTDiffusion",
            "api_name": "/process_dc",
            "params": lambda: [
                handle_file(person_resized),
                handle_file(cloth_resized),
                0,  # category: 0=upper, 1=lower, 2=dress
                1,  # n_samples
                20, # n_steps
                2.0, # image_scale
                42  # seed
            ]
        }
    ]
    
    # Try each space with retry logic
    for space_config in api_spaces:
        space_name = space_config["name"]
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"\n[VTON] Trying {space_name} (Attempt {attempt}/{MAX_RETRIES})...")
                
                # Initialize client
                client = Client(space_name, hf_token=HF_TOKEN)
                
                # Get parameters
                params = space_config["params"]()
                api_name = space_config.get("api_name")
                
                # Call API
                if isinstance(params, dict):
                    if api_name:
                        result = client.predict(**params, api_name=api_name)
                    else:
                        result = client.predict(**params)
                else:
                    if api_name:
                        result = client.predict(*params, api_name=api_name)
                    else:
                        result = client.predict(*params)
                
                print(f"[VTON] ✓ Success with {space_name}!")
                
                # Handle different return formats
                if isinstance(result, tuple):
                    result_path = result[0]
                elif isinstance(result, dict) and 'image' in result:
                    result_path = result['image']
                elif isinstance(result, str):
                    result_path = result
                else:
                    result_path = result
                
                # Clean up temp files
                cleanup_temp_files(person_resized, cloth_resized, person_path, cloth_path)
                
                return result_path
            
            except Exception as e:
                error_msg = str(e)
                print(f"[VTON] ✗ Failed: {error_msg[:100]}")
                
                # If not the last attempt, wait before retry
                if attempt < MAX_RETRIES:
                    print(f"[VTON] Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    print(f"[VTON] All retries exhausted for {space_name}")
    
    # Clean up temp files even on failure
    cleanup_temp_files(person_resized, cloth_resized, person_path, cloth_path)
    
    raise Exception("All VTON API spaces failed after retries")

def cleanup_temp_files(person_resized, cloth_resized, person_original, cloth_original):
    """Remove temporary resized files."""
    for temp_file in [person_resized, cloth_resized]:
        if temp_file != person_original and temp_file != cloth_original:
            if os.path.exists(temp_file) and "temp_resized" in temp_file:
                try:
                    os.remove(temp_file)
                except:
                    pass

def main():
    parser = argparse.ArgumentParser(description="Standalone VTON Engine")
    parser.add_argument("--person", required=True, help="Path to person image")
    parser.add_argument("--cloth", required=True, help="Path to cloth image")
    parser.add_argument("--output", default="output_vton.jpg", help="Output filename")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.person):
        print(f"Error: Person image not found: {args.person}")
        return
    
    if not os.path.exists(args.cloth):
        print(f"Error: Cloth image not found: {args.cloth}")
        return
    
    print("=" * 70)
    print("VTON Engine - Standalone Mode")
    print("=" * 70)
    print(f"Person: {args.person}")
    print(f"Cloth: {args.cloth}")
    print("=" * 70)
    
    try:
        # Run VTON
        result_path = run_tryon(args.person, args.cloth)
        
        # Save result
        if result_path and os.path.exists(result_path):
            import shutil
            shutil.copy(result_path, args.output)
            print(f"\n✓ SUCCESS: Result saved to {os.path.abspath(args.output)}")
        else:
            print("\n✗ ERROR: No result file generated")
    
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {e}")

if __name__ == "__main__":
    main()
