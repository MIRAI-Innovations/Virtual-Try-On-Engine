import os
import argparse
import base64
import requests
import time

# --- CONFIGURATION ---
API_KEY = os.getenv("SEGMIND_API_KEY")

# Endpoints
VTON_URL = "https://api.segmind.com/v1/segfit-v1.3"

def encode_image_to_base64(image_path, max_size=1024):
    """Reads a local file, resizes if needed, and converts to raw Base64 string."""
    from PIL import Image
    import io
    
    # Open and resize image
    img = Image.open(image_path)
    
    # Resize if too large
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        print(f"      Resized: {Image.open(image_path).size} → {new_size}")
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=95)
    img_byte_arr = img_byte_arr.getvalue()
    
    # Encode to base64
    encoded_string = base64.b64encode(img_byte_arr).decode('utf-8')
    return encoded_string

def run_segmind_vton(person_path, cloth_path, output_path="result.jpg"):
    if not API_KEY:
        print("❌ Error: SEGMIND_API_KEY not found. Run 'export SEGMIND_API_KEY=SG_...'")
        return False

    print("🚀 Starting Commercial VTON (Segmind SegFit v1.3)...")
    
    # 1. Encode images to base64
    print(f"   📸 Encoding {os.path.basename(person_path)}...")
    person_b64 = encode_image_to_base64(person_path)
    
    print(f"   📸 Encoding {os.path.basename(cloth_path)}...")
    cloth_b64 = encode_image_to_base64(cloth_path)

    # 2. Call VTON API with base64 images directly
    print("   ✨ Generating Try-On...")
    payload = {
        "model_image": person_b64,
        "outfit_image": cloth_b64,
        "category": "upper_body",  # Options: upper_body, lower_body, dress
        "base64": False  # Return binary image
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(VTON_URL, json=payload, headers=headers, timeout=120)
        
        # 3. Handle Result
        if response.status_code == 200:
            # Check content type
            content_type = response.headers.get('content-type', '')
            
            if 'image' in content_type:
                # Binary image response
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Success! Saved to: {os.path.abspath(output_path)}")
                return True
            else:
                # Try JSON response
                try:
                    data = response.json()
                    if 'image' in data:
                        # Base64 encoded image
                        img_data = base64.b64decode(data['image'])
                        with open(output_path, 'wb') as f:
                            f.write(img_data)
                        print(f"✅ Success! Saved to: {os.path.abspath(output_path)}")
                        return True
                    elif 'output' in data:
                        # URL to download
                        img_resp = requests.get(data['output'])
                        with open(output_path, 'wb') as f:
                            f.write(img_resp.content)
                        print(f"✅ Success! Saved to: {os.path.abspath(output_path)}")
                        return True
                except:
                    pass
                
                # Fallback: save raw content
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"✅ Saved response to: {os.path.abspath(output_path)}")
                return True
        else:
            print(f"❌ VTON Failed (Status {response.status_code}): {response.text}")
            return False

    except requests.exceptions.Timeout:
        print("❌ Request timed out. The API might be processing - try again in a moment.")
        return False
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Commercial VTON using Segmind SegFit")
    parser.add_argument("--person", required=True, help="Path to person image")
    parser.add_argument("--cloth", required=True, help="Path to cloth image")
    parser.add_argument("--output", default="result.jpg", help="Output filename")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.person):
        print(f"❌ Error: Person image not found: {args.person}")
        exit(1)
    
    if not os.path.exists(args.cloth):
        print(f"❌ Error: Cloth image not found: {args.cloth}")
        exit(1)
    
    success = run_segmind_vton(args.person, args.cloth, args.output)
    exit(0 if success else 1)
