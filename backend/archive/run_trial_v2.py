"""
CatVTON Trial Runner v2
Automated Virtual Try-On with MediaPipe Feature Detection and Trial Management
"""

import os
import random
import csv
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from gradio_client import Client, handle_file

# ============================================================================
# CONFIGURATION SECTION - DEFAULT PATHS (Can be overridden via CLI)
# ============================================================================
DEFAULT_SUBJECT_IMAGE = r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\my photos\IMG_0928.jpg"
DEFAULT_CLOTH_DIRECTORY = r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\datasets\test\cloth"
HF_TOKEN = "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"

# Manual User Inputs
HEIGHT_CM = 175
AGE = 22
GENDER = "Male"
STYLE_PREFERENCE = "Casual"

# ============================================================================
# TRIAL FOLDER MANAGEMENT
# ============================================================================
def get_next_trial_number():
    """Scan current directory for Trial folders and return next number."""
    current_dir = Path.cwd()
    trial_folders = [d for d in current_dir.iterdir() if d.is_dir() and d.name.startswith("Trial ")]
    
    if not trial_folders:
        return 1
    
    # Extract numbers from folder names
    numbers = []
    for folder in trial_folders:
        try:
            num = int(folder.name.replace("Trial ", "").strip())
            numbers.append(num)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1

def create_trial_folder(trial_num):
    """Create Trial folder and return path."""
    folder_name = f"Trial {trial_num}"
    folder_path = Path.cwd() / folder_name
    folder_path.mkdir(exist_ok=True)
    print(f"[Trial Manager] Created folder: {folder_path}")
    return folder_path

# ============================================================================
# MEDIAPIPE FEATURE DETECTION
# ============================================================================
def detect_skin_tone(image_path):
    """Extract skin tone from cheek region using MediaPipe Face Mesh."""
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        
        image = cv2.imread(str(image_path))
        if image is None:
            return "Unknown"
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return "Unknown"
        
        # Get cheek landmarks (approximate indices for cheek region)
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = image.shape[:2]
        
        # Sample cheek region (landmarks around 330-340 for right cheek)
        cheek_indices = [330, 331, 332, 333, 334]
        cheek_pixels = []
        
        for idx in cheek_indices:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                if 0 <= x < w and 0 <= y < h:
                    pixel = rgb_image[y, x]
                    cheek_pixels.append(pixel)
        
        if not cheek_pixels:
            return "Unknown"
        
        # Average color
        avg_color = np.mean(cheek_pixels, axis=0)
        brightness = np.mean(avg_color)
        
        # Classify skin tone
        if brightness > 200:
            return "Fair"
        elif brightness > 150:
            return "Medium"
        else:
            return "Dark"
    
    except Exception as e:
        print(f"[MediaPipe] Skin tone detection failed: {e}")
        return "Unknown"

def detect_face_shape(image_path):
    """Calculate face shape from jaw/face height ratio."""
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        
        image = cv2.imread(str(image_path))
        if image is None:
            return "Unknown"
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return "Unknown"
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = image.shape[:2]
        
        # Approximate landmarks: top of head (10), chin (152), jaw width (234, 454)
        top_y = landmarks[10].y * h
        chin_y = landmarks[152].y * h
        left_jaw_x = landmarks[234].x * w
        right_jaw_x = landmarks[454].x * w
        
        face_height = abs(chin_y - top_y)
        jaw_width = abs(right_jaw_x - left_jaw_x)
        
        if face_height == 0:
            return "Unknown"
        
        ratio = jaw_width / face_height
        
        # Classify based on ratio
        if ratio > 0.75:
            return "Round"
        elif ratio > 0.65:
            return "Oval"
        else:
            return "Square"
    
    except Exception as e:
        print(f"[MediaPipe] Face shape detection failed: {e}")
        return "Unknown"

def detect_body_shape(image_path):
    """Calculate body shape from shoulder/waist ratio using MediaPipe Pose."""
    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True)
        
        image = cv2.imread(str(image_path))
        if image is None:
            return "Unknown"
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return "Unknown"
        
        landmarks = results.pose_landmarks.landmark
        h, w = image.shape[:2]
        
        # Shoulder landmarks: 11 (left), 12 (right)
        # Hip landmarks: 23 (left), 24 (right)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        shoulder_width = abs((right_shoulder.x - left_shoulder.x) * w)
        hip_width = abs((right_hip.x - left_hip.x) * w)
        
        if hip_width == 0:
            return "Unknown"
        
        ratio = shoulder_width / hip_width
        
        # Classify body shape
        if ratio > 1.1:
            return "Inverted Triangle"
        elif ratio > 0.95:
            return "Rectangle"
        else:
            return "Hourglass"
    
    except Exception as e:
        print(f"[MediaPipe] Body shape detection failed: {e}")
        return "Unknown"

def derive_age_group(age):
    """Derive age group from age."""
    if age < 20:
        return "Teen"
    elif age < 40:
        return "Adult"
    else:
        return "Mature"

def derive_arm_preference(age):
    """Derive arm coverage preference from age."""
    if age > 50:
        return "Cover"
    else:
        return "Any"

# ============================================================================
# CATVTON API INTEGRATION
# ============================================================================
def run_catvton(person_img, cloth_img, token):
    """
    Call VTON API via gradio_client.
    Tries multiple working VTON spaces.
    """
    print("[VTON] Initializing client...")
    
    # Try different working VTON spaces
    spaces_to_try = [
        {
            "name": "yisol/IDM-VTON",
            "api_name": "/tryon",
            "params": lambda: {
                "dict_val": {
                    "background": handle_file(person_img),
                    "layers": [],
                    "composite": None
                },
                "garm_img": handle_file(cloth_img),
                "garment_des": "clothing",
                "is_checked": True,
                "is_checked_crop": False,
                "denoise_steps": 30,
                "seed": 42
            }
        },
        {
            "name": "Kwai-Kolors/Kolors-Virtual-Try-On",
            "api_name": None,  # Try default endpoint
            "params": lambda: [
                handle_file(person_img),
                handle_file(cloth_img)
            ]
        },
        {
            "name": "miragic/miragic_tryon",
            "api_name": "/tryon",
            "params": lambda: [
                handle_file(person_img),
                handle_file(cloth_img)
            ]
        },
        {
            "name": "levihsu/OOTDiffusion",
            "api_name": "/process_dc",
            "params": lambda: [
                handle_file(person_img),
                handle_file(cloth_img),
                0,  # category: 0=upper, 1=lower, 2=dress
                1,  # n_samples
                20, # n_steps
                2.0, # image_scale
                42  # seed
            ]
        }
    ]
    
    for space_config in spaces_to_try:
        try:
            space_name = space_config["name"]
            print(f"[VTON] Trying space: {space_name}")
            client = Client(space_name, hf_token=token)
            
            # Get parameters
            params = space_config["params"]()
            api_name = space_config["api_name"]
            
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
            
            print(f"[VTON] Success with {space_name}!")
            
            # Handle different return formats
            if isinstance(result, tuple):
                # IDM-VTON returns (result_image, masked_image)
                return result[0]
            elif isinstance(result, dict) and 'image' in result:
                return result['image']
            elif isinstance(result, str):
                # Direct file path
                return result
            return result
            
        except Exception as e:
            print(f"[VTON] Failed with {space_config['name']}: {e}")
            continue
    
    raise Exception("All VTON spaces failed")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CatVTON Trial Runner v2")
    parser.add_argument("--subject", type=str, default=DEFAULT_SUBJECT_IMAGE,
                        help="Path to subject image")
    parser.add_argument("--cloth", type=str, default=DEFAULT_CLOTH_DIRECTORY,
                        help="Path to cloth image or directory of cloth images")
    parser.add_argument("--height", type=int, default=175,
                        help="Height in cm (default: 175)")
    parser.add_argument("--age", type=int, default=22,
                        help="Age (default: 22)")
    parser.add_argument("--gender", type=str, default="Male",
                        help="Gender (default: Male)")
    parser.add_argument("--style", type=str, default="Casual",
                        help="Style preference (default: Casual)")
    
    args = parser.parse_args()
    
    SUBJECT_IMAGE = args.subject
    CLOTH_PATH = args.cloth
    HEIGHT_CM = args.height
    AGE = args.age
    GENDER = args.gender
    STYLE_PREFERENCE = args.style
    
    print("=" * 70)
    print("CatVTON Trial Runner v2")
    print("=" * 70)
    print(f"Subject: {SUBJECT_IMAGE}")
    print(f"Cloth: {CLOTH_PATH}")
    print(f"Profile: {AGE}yo {GENDER}, {HEIGHT_CM}cm, {STYLE_PREFERENCE}")
    print("=" * 70)
    
    # 1. Validate inputs
    if not os.path.exists(SUBJECT_IMAGE):
        print(f"[ERROR] Subject image not found: {SUBJECT_IMAGE}")
        return
    
    # Check if cloth path is a directory or a file
    if os.path.isdir(CLOTH_PATH):
        CLOTH_DIRECTORY = CLOTH_PATH
        cloth_files = [f for f in os.listdir(CLOTH_DIRECTORY) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not cloth_files:
            print(f"[ERROR] No cloth images found in {CLOTH_DIRECTORY}")
            return
        
        selected_cloth = random.choice(cloth_files)
        cloth_path = os.path.join(CLOTH_DIRECTORY, selected_cloth)
        print(f"[Cloth Selection] Randomly selected: {selected_cloth}")
    elif os.path.isfile(CLOTH_PATH):
        cloth_path = CLOTH_PATH
        print(f"[Cloth Selection] Using specified cloth: {os.path.basename(cloth_path)}")
    else:
        print(f"[ERROR] Cloth path not found: {CLOTH_PATH}")
        return

    
    # 2. Create Trial folder
    trial_num = get_next_trial_number()
    trial_folder = create_trial_folder(trial_num)
    
    # 3. Run MediaPipe Detection
    print("\n[MediaPipe] Running feature detection...")
    skin_tone = detect_skin_tone(SUBJECT_IMAGE)
    face_shape = detect_face_shape(SUBJECT_IMAGE)
    body_shape = detect_body_shape(SUBJECT_IMAGE)
    age_group = derive_age_group(AGE)
    arm_pref = derive_arm_preference(AGE)
    
    print(f"  Skin Tone: {skin_tone}")
    print(f"  Face Shape: {face_shape}")
    print(f"  Body Shape: {body_shape}")
    print(f"  Age Group: {age_group}")
    print(f"  Arm Preference: {arm_pref}")
    
    # 4. Save original and cloth images
    print("\n[File Manager] Saving input files...")
    shutil.copy(SUBJECT_IMAGE, trial_folder / "original.jpg")
    shutil.copy(cloth_path, trial_folder / "cloth.jpg")
    
    # 5. Run VTON
    result_path = None
    try:
        print("\n[VTON] Running virtual try-on...")
        result_path = run_catvton(SUBJECT_IMAGE, cloth_path, HF_TOKEN)
        
        if result_path and os.path.exists(result_path):
            shutil.copy(result_path, trial_folder / "result.jpg")
            print(f"[VTON] Result saved to {trial_folder / 'result.jpg'}")
        else:
            print("[VTON] No result file generated")
    
    except Exception as e:
        print(f"[VTON] API call failed: {e}")
        print("[VTON] Continuing to save metadata...")
    
    # 6. Generate CSV
    print("\n[CSV Generator] Creating session_data.csv...")
    csv_path = trial_folder / "session_data.csv"
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'Height', 'Age', 'Gender', 'Style', 
            'Detected_SkinTone', 'Detected_FaceShape', 'Detected_BodyShape',
            'Derived_AgeGroup', 'Derived_ArmPref', 'Timestamp'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow({
            'Height': HEIGHT_CM,
            'Age': AGE,
            'Gender': GENDER,
            'Style': STYLE_PREFERENCE,
            'Detected_SkinTone': skin_tone,
            'Detected_FaceShape': face_shape,
            'Detected_BodyShape': body_shape,
            'Derived_AgeGroup': age_group,
            'Derived_ArmPref': arm_pref,
            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    print(f"[CSV Generator] Saved to {csv_path}")
    
    # 7. Summary
    print("\n" + "=" * 70)
    print(f"TRIAL {trial_num} COMPLETE")
    print("=" * 70)
    print(f"Location: {trial_folder}")
    print(f"Files saved:")
    print(f"  - original.jpg")
    print(f"  - cloth.jpg")
    print(f"  - result.jpg {'✓' if result_path else '✗ (API failed)'}")
    print(f"  - session_data.csv ✓")
    print("=" * 70)

if __name__ == "__main__":
    main()
