
import argparse
import os
import random
import time
from gradio_client import Client, handle_file
from PIL import Image

def smart_resize(image_path, target_size=(768, 1024)):
    """
    Resizes an image to fit within target_size while preserving aspect ratio,
    padding with WHITE (255, 255, 255) to fill the remaining space.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_ratio = img.width / img.height
        target_ratio = target_size[0] / target_size[1]

        if img_ratio > target_ratio:
            # Image is wider than target
            new_width = target_size[0]
            new_height = int(new_width / img_ratio)
        else:
            # Image is taller than target
            new_height = target_size[1]
            new_width = int(new_height * img_ratio)

        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create white background
        new_img = Image.new("RGB", target_size, (255, 255, 255))
        
        # Center the resized image
        paste_x = (target_size[0] - new_width) // 2
        paste_y = (target_size[1] - new_height) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        
        # Save to a temporary file
        temp_path = f"temp_resized_{os.path.basename(image_path)}"
        new_img.save(temp_path)
        return temp_path
    
    except Exception as e:
        print(f"[Smart Resize] Error: {e}")
        return image_path

def run_kolors(person_img, cloth_img, seed, token):
    """
    Attempts to run Kwai-Kolors/Kolors-Virtual-Try-On.
    Implements a brute-force retry mechanism for robustness.
    """
    print(f"[Kolors] Initializing client with token ending in ...{token[-4:]}")
    client = Client("Kwai-Kolors/Kolors-Virtual-Try-On", hf_token=token)
    
    # Brute-force parameter combinations based on common API changes
    attempts = [
        # Attempt 1: Try without api_name (use default endpoint)
        lambda: client.predict(
            handle_file(person_img),
            handle_file(cloth_img)
        ),
        # Attempt 2: Try with positional args and seed
        lambda: client.predict(
            handle_file(person_img),
            handle_file(cloth_img),
            seed
        ),
        # Attempt 3: Try with named parameters
        lambda: client.predict(
            person_img=handle_file(person_img),
            garment_img=handle_file(cloth_img)
        )
    ]

    for i, attempt in enumerate(attempts):
        try:
            print(f"[Kolors] Attempt {i+1}...")
            result = attempt()
            # Result format check: usually returns tuple (image_path, seed_info) or just image_path
            if isinstance(result, tuple):
                return result[0]
            return result
        except Exception as e:
            print(f"[Kolors] Attempt {i+1} failed: {e}")
            time.sleep(1) # Brief pause
            
    raise Exception("All Kolors attempts failed.")

def run_ootd(person_img, cloth_img, category, seed, token):
    """
    Fallback to levihsu/OOTDiffusion.
    Uses /process_dc endpoint which accepts category parameter.
    """
    print(f"[OOTD] Initializing Fallback Client...")
    client = Client("levihsu/OOTDiffusion", hf_token=token)
    
    # Map category to OOTD integer expectations for /process_dc
    # 0 = upperbody, 1 = lowerbody, 2 = dress
    if category.lower() in ["dress", "full-body", "full"]:
        ootd_category = 2
        category_name = "Dress"
    elif category.lower() in ["lower", "lower-body", "bottom"]:
        ootd_category = 1
        category_name = "Lower-body"
    else:
        ootd_category = 0
        category_name = "Upper-body"
        
    print(f"[OOTD] Using category: {category_name} (code: {ootd_category})")

    # OOTD DC signature with category as integer
    return client.predict(
		handle_file(person_img),
		handle_file(cloth_img),
		ootd_category,
		1,  # n_samples
		20, # n_steps
		2.0, # image_scale
		seed,
		api_name="/process_dc"
    )

def main():
    parser = argparse.ArgumentParser(description="Robust Virtual Try-On Script")
    parser.add_argument("--person", required=True, help="Path to person image")
    parser.add_argument("--cloth", required=True, help="Path to cloth image")
    parser.add_argument("--category", default="Upper-body", help="Category: Upper-body, Lower-body, Dress")
    parser.add_argument("--token", default="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", help="Hugging Face Token")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # 1. Validation
    if not os.path.exists(args.person):
        print(f"Error: Person image not found at {args.person}")
        return
    if not os.path.exists(args.cloth):
        print(f"Error: Cloth image not found at {args.cloth}")
        return

    # 2. Smart Resize (White Padding)
    print("--- Step 1: Smart Resizing ---")
    resized_person = smart_resize(args.person)
    resized_cloth = smart_resize(args.cloth)
    print(f"Resized images saved to temp: {resized_person}, {resized_cloth}")

    final_result_path = None
    
    # 3. Try Kolors (Primary)
    try:
        print("\n--- Step 2: Attempting Kwai-Kolors (Primary) ---")
        # Generate a random seed if default is used to prompt variety if needed, 
        # but args.seed is respected if manually set.
        # Actually, let's just use the arg seed to ensure reproducibility if desired.
        
        result_path = run_kolors(resized_person, resized_cloth, args.seed, args.token)
        print("Kolors Success!")
        final_result_path = result_path
        
    except Exception as e:
        print(f"\n[WARNING] Kolors Failed: {e}")
        print("--- Step 3: Switch to OOTDiffusion (Fallback) ---")
        try:
            # Note: OOTD generally works better with the raw resized cloth, 
            # but sometimes passing the original cloth (masked by OOTD internally) is fine.
            # We will use our resized one to ensure 768x1024 consistency.
            result_path = run_ootd(resized_person, resized_cloth, args.category, args.seed, args.token)
            
            # OOTD often returns a list of files or a single file path depending on n_samples
            if isinstance(result_path, (list, tuple)):
                final_result_path = result_path[0]['image'] if isinstance(result_path[0], dict) else result_path[0]
            else:
                final_result_path = result_path
                
            print("OOTDiffusion Success!")
            
        except Exception as ootd_e:
            print(f"\n[CRITICAL ERROR] OOTDiffusion also failed: {ootd_e}")
            return

    # 4. Save Output
    if final_result_path and os.path.exists(final_result_path):
        output_filename = "output_vton.jpg"
        # Move/Copy to current dir
        import shutil
        shutil.copy(final_result_path, output_filename)
        print(f"\nSUCCESS: Result saved to {os.path.abspath(output_filename)}")
        
        # Clean up temp files
        try:
            if os.path.exists(resized_person) and "temp_resized" in resized_person:
                os.remove(resized_person)
            if os.path.exists(resized_cloth) and "temp_resized" in resized_cloth:
                os.remove(resized_cloth)
        except:
            pass
    else:
        print("Error: No output file was generated.")

if __name__ == "__main__":
    main()
