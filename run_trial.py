import os
import argparse
import shutil
import random
import time
import pandas as pd
import cv2
import mediapipe as mp
import math
import traceback
from PIL import Image, ImageOps
from gradio_client import Client, handle_file
try:
    import detect_metadata
except ImportError:
    detect_metadata = None

# =============================================================================
# CONFIGURATION
# =============================================================================

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_processed")
os.makedirs(TEMP_DIR, exist_ok=True)

DEFAULT_SUBJECT_IMAGE_PATH = r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\my photos\IMG_1337.jpg"
CLOTH_DIR = r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\datasets\test\cloth"

# User Manual Inputs
# User Manual Inputs (Now handled via CLI args)
# Defaults moved to main() parser


# VTON Space
VTON_SPACE = "yisol/IDM-VTON"

# =============================================================================
# FEATURE SCANNER (MediaPipe)
# =============================================================================

# =============================================================================
# FEATURE SCANNER (MediaPipe)
# =============================================================================

class FeatureScanner:
    def __init__(self):
        self.enabled = False
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_face_mesh = mp.solutions.face_mesh
            self.pose = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
            self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
            self.enabled = True
        except AttributeError:
            print("Warning: mediapipe.solutions not found. Feature extraction disabled.")
        except Exception as e:
            print(f"Warning: Failed to initialize MediaPipe: {e}. Feature extraction disabled.")

    def analyze_image(self, image_path):
        """Runs MediaPipe analysis on the image and returns feature dictionary."""
        results = {
            "Body Shape": "Unknown",
            "Face Shape": "Unknown",
            "Skin Tone": "Unknown"
        }
        
        if not self.enabled:
            return results

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            # 1. Body Shape Analysis
            pose_results = self.pose.process(image_rgb)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                
                # Get key coordinates (normalized)
                l_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                r_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                l_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                r_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
                
                # Calculate widths (Euclidean distance on x,y plane)
                shoulder_width = math.sqrt((l_shoulder.x - r_shoulder.x)**2 + (l_shoulder.y - r_shoulder.y)**2)
                hip_width = math.sqrt((l_hip.x - r_hip.x)**2 + (l_hip.y - r_hip.y)**2)
                
                ratio = shoulder_width / hip_width if hip_width > 0 else 1.0
                
                print(f"  [Debug] Shoulder/Hip Ratio: {ratio:.2f}")

                if ratio > 1.05:
                    results["Body Shape"] = "Inverted Triangle"
                elif 0.95 <= ratio <= 1.05:
                    results["Body Shape"] = "Hourglass" # Simplified
                else:
                    results["Body Shape"] = "Rectangle"
            
            # 2. Face Shape Analysis
            face_results = self.face_mesh.process(image_rgb)
            if face_results.multi_face_landmarks:
                landmarks = face_results.multi_face_landmarks[0].landmark
                
                # Approximations only. Indices based on canonical face mesh model.
                left_cheek = landmarks[454]
                right_cheek = landmarks[234]
                top_forehead = landmarks[10]
                chin = landmarks[152]
                
                face_width = math.sqrt((left_cheek.x - right_cheek.x)**2 + (left_cheek.y - right_cheek.y)**2)
                face_height = math.sqrt((top_forehead.x - chin.x)**2 + (top_forehead.y - chin.y)**2)
                
                ratio = face_width / face_height if face_height > 0 else 1.0
                print(f"  [Debug] Face Width/Height Ratio: {ratio:.2f}")
                
                if ratio > 0.9:
                    results["Face Shape"] = "Round" 
                elif ratio < 0.8:
                    results["Face Shape"] = "Oval" 
                else:
                    results["Face Shape"] = "Square"

                # 3. Skin Tone Analysis (roi on cheek)
                h, w, c = image.shape
                cx, cy = int(right_cheek.x * w), int(right_cheek.y * h)
                
                roi_size = 10
                y1, y2 = max(0, cy-roi_size), min(h, cy+roi_size)
                x1, x2 = max(0, cx-roi_size), min(w, cx+roi_size)
                
                roi = image_rgb[y1:y2, x1:x2]
                if roi.size > 0:
                    avg_color = roi.mean(axis=(0,1))
                    intensity = sum(avg_color) / 3
                    print(f"  [Debug] Skin Intensity: {intensity:.2f}")
                    
                    if intensity > 170:
                        results["Skin Tone"] = "Fair"
                    elif intensity > 100:
                        results["Skin Tone"] = "Medium"
                    else:
                        results["Skin Tone"] = "Dark"

            # 3. Fallback Skin Tone (using Pose Nose landmark) if Face Mesh failed or didn't run
            if results["Skin Tone"] == "Unknown" and pose_results.pose_landmarks:
                 landmarks = pose_results.pose_landmarks.landmark
                 nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
                 
                 # Check if nose is within image bounds
                 if 0 <= nose.x <= 1 and 0 <= nose.y <= 1:
                     h, w, c = image.shape
                     cx, cy = int(nose.x * w), int(nose.y * h)
                     
                     roi_size = 5
                     y1, y2 = max(0, cy-roi_size), min(h, cy+roi_size)
                     x1, x2 = max(0, cx-roi_size), min(w, cx+roi_size)
                     
                     roi = image_rgb[y1:y2, x1:x2]
                     if roi.size > 0:
                        avg_color = roi.mean(axis=(0,1))
                        intensity = sum(avg_color) / 3
                        print(f"  [Debug] Fallback Skin Intensity (Nose): {intensity:.2f}")
                        
                        if intensity > 170:
                            results["Skin Tone"] = "Fair"
                        elif intensity > 100:
                            results["Skin Tone"] = "Medium"
                        else:
                            results["Skin Tone"] = "Dark"

        except Exception as e:
            print(f"Error during feature analysis: {e}")
            
        return results

# =============================================================================
# VTON EXECUTOR
# =============================================================================

# VTON Spaces Configuration
# VTON Spaces Configuration
SPACES = [
    {
        "name": "levihsu/OOTDiffusion",
        "type": "ootd",
        "notes": "Primary Working Model (Stable)"
    },
    {
        "name": "yisol/IDM-VTON",
        "type": "idm",
        "notes": "SOTA (Currently Unstable/Quota Limit)"
    },
    {
        "name": "zhengchong/CatVTON", 
        "type": "catvton",
        "notes": "Backup"
    }
]

def run_prediction(client, space_config, person_path, cloth_path, category="Upper-body", steps=30, scale=2.5, seed=-1):
    space_name = space_config["name"]
    space_type = space_config["type"]
    
    print(f"  Attempting prediction with {space_name} ({space_type})...")
    
    try:
        if space_type == "ootd":
            return client.predict(
                vton_img=handle_file(person_path),
                garm_img=handle_file(cloth_path),
                category=category, 
                n_samples=1,
                n_steps=steps,
                image_scale=scale,
                seed=seed,
                api_name="/process_dc" 
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
                garment_des="A stylish garment",
                is_checked=True, 
                is_checked_crop=False, 
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
    except Exception as e:
        print(f"  Error during prediction with {space_name}: {e}")
        print(traceback.format_exc())
        return None
    return None

def run_vton_trial(person_path, cloth_path, category="Upper-body", steps=30, scale=2.5, seed=-1, hf_token=None):
    final_image = None
    
    # Iterate through spaces and try to run the FULL pipeline (Connect -> Predict)
    # If any step fails, move to the next space.
    
    for config in SPACES:
        name = config["name"]
        print(f"\n--- Trying {name} ---")
        
        client = None
        
        # 1. Connect
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if hf_token:
                    print(f"  Authenticating with token ending in ...{hf_token[-4:]}")
                    try:
                        client = Client(name, hf_token=hf_token)
                    except TypeError:
                        print("  (Warning: hf_token not supported by this gradio_client version. Connecting without token...)")
                        client = Client(name)
                else:
                    client = Client(name)
                print(f"  Connected to {name}!")
                break
            except Exception as e:
                print(f"  Connection Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1: time.sleep(1)

        if not client:
            print(f"  Could not connect to {name}. Skipping.")
            continue

        # 2. Run Prediction with this client
        result = run_prediction(client, config, person_path, cloth_path, category, steps, scale, seed)
        
        # 3. Handle Result
        if result:
            # Check if result is valid path
            if isinstance(result, (list, tuple)):
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
                print(f"  > Success with {name}!")
                return final_image
            else:
                print(f"  > {name} returned empty result. Trying next...")
        else:
            print(f"  > Prediction failed with {name}. Trying next...")

    print("CRITICAL: All VTON spaces failed.")
    return None

def smart_resize(image_path, target_size=(768, 1024)):
    """
    Resizes image to target_size while maintaining aspect ratio and adding padding (letterboxing).
    Returns path to the temporary processed image.
    """
    try:
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img) # Fix orientation if needed
        
        # Calculate aspect ratios
        target_ratio = target_size[0] / target_size[1]
        img_ratio = img.width / img.height
        
        if img_ratio > target_ratio:
            # Image is wider than target: resize by width
            new_width = target_size[0]
            new_height = int(new_width / img_ratio)
        else:
            # Image is taller than target: resize by height
            new_height = target_size[1]
            new_width = int(new_height * img_ratio)
            
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new blank image (White background 255,255,255)
        # Black (0,0,0) can be interpreted as 'mask' or 'hair' by VTON models, causing warping.
        new_img = Image.new("RGB", target_size, (255, 255, 255))
        
        # Paste centered
        x_offset = (target_size[0] - new_width) // 2
        y_offset = (target_size[1] - new_height) // 2
        new_img.paste(img_resized, (x_offset, y_offset))
        
        # Save
        filename = f"processed_{os.path.basename(image_path)}"
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filename += ".jpg"
            
        save_path = os.path.join(TEMP_DIR, filename)
        new_img.save(save_path, quality=95)
        print(f"  [Smart Resize] Processed: {image_path} -> {save_path} ({target_size})")
        return save_path
        
    except Exception as e:
        print(f"  [Error] Smart resize failed: {e}")
        return image_path # Fallback to original

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run MIRAI Virtual Try-On Trial")
    parser.add_argument("--image", default=DEFAULT_SUBJECT_IMAGE_PATH, help="Path to the subject image")
    parser.add_argument("--cloth", default=None, help="Path to specific cloth image (optional)")
    parser.add_argument("--category", default="Upper-body", choices=["Upper-body", "Lower-body", "Dress"], help="Garment category (Upper-body, Lower-body, Dress)")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps (Quality). Default 30.")
    parser.add_argument("--scale", type=float, default=2.5, help="Guidance scale (Adherence). Default 2.5.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed. Default -1 (Random).")
    parser.add_argument("--token", type=str, default="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", help="Hugging Face Token for higher quota limits.")
    
    # User Metadata Arguments
    parser.add_argument("--gender", default="Female", help="User Gender (Default: Female)")
    parser.add_argument("--age", type=int, default=25, help="User Age (Default: 25)")
    parser.add_argument("--height", default="170 cm", help="User Height (Default: 170 cm)")
    
    args = parser.parse_args()

    subject_image_path = args.image
    
    print("=== Starting MIRAI Trial Runner ===")
    print(f"Subject: {subject_image_path}")
    
    # 1. Validation
    if not os.path.exists(subject_image_path):
        print(f"Error: Subject image not found at {subject_image_path}")
        return
    if not os.path.exists(CLOTH_DIR):
        print(f"Error: Cloth directory not found at {CLOTH_DIR}")
        return

    # 2. Pick Cloth (Random or Specified)
    if args.cloth and os.path.exists(args.cloth):
        selected_cloth_path = args.cloth
        selected_cloth_name = os.path.basename(selected_cloth_path)
    else:
        cloth_files = [f for f in os.listdir(CLOTH_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not cloth_files:
            print("Error: No cloth images found.")
            return
        
        selected_cloth_name = random.choice(cloth_files)
        selected_cloth_path = os.path.join(CLOTH_DIR, selected_cloth_name)
    
    print(f"Selected Cloth: {selected_cloth_name}")

    # 3. Feature Scanner
    print("\n--- Running Feature Scanner ---")
    scanner = FeatureScanner()
    try:
        features = scanner.analyze_image(subject_image_path)
        print("Detected Features:", features)
    except Exception as e:
        print(f"Feature scanning failed: {e}")
        features = {"Body Shape": "Error", "Face Shape": "Error", "Skin Tone": "Error"}

    # 4. Derive Data & Auto-Detect (if needed)
    detected_meta = None
    if detect_metadata and (args.age == 25 and args.gender == "Female"): # Logic: If user didn't change defaults (naive check)
        print("\n--- Running AI Age/Gender Detection ---")
        detected_meta = detect_metadata.analyze_dataset(subject_image_path)
    
    # Priority: CLI Args > AI Detection > Defaults
    # Since we have defaults in parser, we use them unless AI overrides and user didn't specify.
    # Actually, better logic: If user *explicitly* set them, use them. 
    # But argparse doesn't tell us if it was default. 
    # Let's trust the AI if it works, otherwise use args.
    
    final_age = args.age
    final_gender = args.gender
    
    if detected_meta:
        print(f"  AI Detected: {detected_meta['Gender']}, Age {detected_meta['Age']}")
        final_age = detected_meta['Age']
        final_gender = detected_meta['Gender']
        
    age = final_age
    gender = final_gender
    
    if age < 20: 
        age_group = "Teen"
    elif age <= 50:
        age_group = "Adult"
    else:
        age_group = "Mature"
    
    arm_preference = "Cover" if age_group == "Mature" else "Any"

    # 4.5 Pre-process (Smart Resize) for VTON
    print("\n--- Processing Image for VTON ---")
    processed_subject_path = smart_resize(subject_image_path)

    # 5. Run VTON
    print("\n--- Running Virtual Try-On ---")
    # Use the processed path for VTON!
    result_path = run_vton_trial(
        processed_subject_path, 
        selected_cloth_path, 
        category=args.category,
        steps=args.steps,
        scale=args.scale,
        seed=args.seed,
        hf_token=args.token
    )
    
    if not result_path:
        print("VTON Failed. Aborting save.")
        return

    # 6. Save Session
    print("\n--- Saving Session ---")
    
    # Find next trial number
    MIRROR_SESSIONS_DIR = os.path.join(BASE_DIR, "Mirror_Sessions")
    os.makedirs(MIRROR_SESSIONS_DIR, exist_ok=True)
    
    existing_trials = [d for d in os.listdir(MIRROR_SESSIONS_DIR) if d.startswith("Trial_")]
    trial_nums = []
    for t in existing_trials:
        try:
            trial_nums.append(int(t.replace("Trial_", "")))
        except:
            pass
    
    next_trial_num = max(trial_nums) + 1 if trial_nums else 1
    session_dir = os.path.join(MIRROR_SESSIONS_DIR, f"Trial_{next_trial_num}")
    os.makedirs(session_dir, exist_ok=True)
    
    # Copy files
    dest_subject = os.path.join(session_dir, "original.jpg")
    dest_cloth = os.path.join(session_dir, "cloth.jpg")
    dest_result = os.path.join(session_dir, "result.jpg")
    
    shutil.copy(subject_image_path, dest_subject)
    shutil.copy(selected_cloth_path, dest_cloth)
    shutil.copy(result_path, dest_result)
    
    # Create CSV Data
    session_data = {
        # User Inputs
        "Height": args.height,
        "Age": final_age,
        "Gender": final_gender,
        "Style Preference": "Casual",
        
        # Detected
        "Body Shape": features["Body Shape"],
        "Face Shape": features["Face Shape"],
        "Skin Tone": features["Skin Tone"],
        
        # Derived
        "Age Group": age_group,
        "Arm Preference": arm_preference,
        
        # Files
        "Subject File": "original.jpg",
        "Cloth File": "cloth.jpg",
        "Result File": "result.jpg"
    }
    
    csv_path = os.path.join(session_dir, "session_data.csv")
    df = pd.DataFrame([session_data])
    df.to_csv(csv_path, index=False)
    
    print(f"Trial {next_trial_num} saved successfully at {session_dir}")

if __name__ == "__main__":
    main()
