import os
import argparse
import shutil
import random
import time
import pandas as pd
import cv2
import mediapipe as mp
import math
from gradio_client import Client, handle_file

# =============================================================================
# CONFIGURATION
# =============================================================================

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SUBJECT_IMAGE_PATH = r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\my photos\IMG_1337.jpg"
CLOTH_DIR = r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\datasets\test\cloth"

# User Manual Inputs
USER_INPUTS = {
    "Height": "175 cm",
    "Age": 22,
    "Gender": "Male",
    "Style Preference": "Casual" 
}

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
        except Exception as e:
            print(f"Error during feature analysis: {e}")
            
        return results

# =============================================================================
# VTON EXECUTOR
# =============================================================================

# VTON Spaces Configuration
SPACES = [
    {
        "name": "yisol/IDM-VTON",
        "type": "idm",
        "notes": "Primary SOTA model"
    },
    {
        "name": "levihsu/OOTDiffusion",
        "type": "ootd",
        "notes": "Fallback, stable"
    },
    {
        "name": "zhengchong/CatVTON", 
        "type": "catvton",
        "notes": "Fast, backup"
    }
]

def run_prediction(client, space_config, person_path, cloth_path):
    space_name = space_config["name"]
    space_type = space_config["type"]
    
    print(f"  Attempting prediction with {space_name} ({space_type})...")
    
    try:
        if space_type == "ootd":
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
                garment_des="A stylish garment",
                is_checked=True, 
                is_checked_crop=False, 
                denoise_steps=30,
                seed=42,
                api_name="/tryon"
            )
    except Exception as e:
        print(f"  Error during prediction with {space_name}: {e}")
        return None
    return None

def run_vton_trial(person_path, cloth_path):
    active_client = None
    active_config = None

    # 1. Connect to a working space
    for config in SPACES:
        name = config["name"]
        print(f"Trying to connect to {name}...")
        
        # Retry logic for connection
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client = Client(name)
                # Simple check if api is viewable
                # client.view_api(return_format="dict") 
                print(f"Successfully connected to {name}!")
                active_client = client
                active_config = config
                break # Break retry loop
            except Exception as e:
                print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2) # Wait a bit before retrying
                else:
                    print(f"  Failed to connect to {name} after {max_retries} attempts.")
        
        if active_client:
            break # Break space loop if connected

    if not active_client:
        print("CRITICAL: All VTON spaces are unreachable.")
        return None

    # 2. Run Prediction
    result = run_prediction(active_client, active_config, person_path, cloth_path)
    
    # 3. Handle Result
    final_image = None
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
        
    return final_image

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run MIRAI Virtual Try-On Trial")
    parser.add_argument("image", nargs="?", default=DEFAULT_SUBJECT_IMAGE_PATH, help="Path to the subject image")
    parser.add_argument("--cloth", default=None, help="Path to specific cloth image (optional)")
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

    # 4. Derive Data
    age = USER_INPUTS["Age"]
    if age < 20: 
        age_group = "Teen"
    elif age <= 50:
        age_group = "Adult"
    else:
        age_group = "Mature"
    
    arm_preference = "Cover" if age_group == "Mature" else "Any"

    # 5. Run VTON
    print("\n--- Running Virtual Try-On ---")
    result_path = run_vton_trial(subject_image_path, selected_cloth_path)
    
    if not result_path:
        print("VTON Failed. Aborting save.")
        return

    # 6. Save Session
    print("\n--- Saving Session ---")
    
    # Find next trial number
    existing_trials = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("Trial ")]
    trial_nums = []
    for t in existing_trials:
        try:
            trial_nums.append(int(t.split(" ")[1]))
        except:
            pass
    
    next_trial_num = max(trial_nums) + 1 if trial_nums else 1
    session_dir = os.path.join(BASE_DIR, f"Trial {next_trial_num}")
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
        "Height": USER_INPUTS["Height"],
        "Age": USER_INPUTS["Age"],
        "Gender": USER_INPUTS["Gender"],
        "Style Preference": USER_INPUTS["Style Preference"],
        
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
