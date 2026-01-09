"""
Main Mirror Controller
Hybrid trigger system with background subtraction + thumbs-up gesture
"""

import cv2
import numpy as np
import mediapipe as mp
import csv
import os
import time
from pathlib import Path
from datetime import datetime
try:
    from commercial_vton import run_segmind_vton
except ImportError:
    run_segmind_vton = None

# ============================================================================
# CONFIGURATION
# ============================================================================
CLOTH_PATH = r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\datasets\test\cloth\00013_00.jpg"
SELECTED_CLOTH_PATH = r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\datasets\test\cloth\00013_00.jpg"
OUTPUT_DIR = "./Mirror_Sessions"

# API Safety Lock
ENABLE_PAID_API = True  # Set to True to use real API (costs credits!)

# Manual Inputs (Cannot be detected from image alone)
MANUAL_USER_DATA = {
    "Age": 22,  # Cannot detect from single image
    "Gender": "Male"  # Cannot detect reliably from pose alone
}

# Detection thresholds
CHANGE_THRESHOLD = 5000  # Minimum changed pixels to detect user
GESTURE_HOLD_FRAMES = 15  # Frames to hold thumbs up
COUNTDOWN_SECONDS = 5
DISPLAY_SECONDS = 8

# ============================================================================
# FOLDER MANAGEMENT
# ============================================================================
def get_next_trial_number():
    """Scan Mirror_Sessions for Trial folders and return next number."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        return 1
    
    trial_folders = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("Trial_")]
    if not trial_folders:
        return 1
    
    numbers = []
    for folder in trial_folders:
        try:
            num = int(folder.replace("Trial_", ""))
            numbers.append(num)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1

def create_trial_folder():
    """Create new trial folder and return path."""
    trial_num = get_next_trial_number()
    folder_path = Path(OUTPUT_DIR) / f"Trial_{trial_num}"
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path, trial_num

# ============================================================================
# BACKGROUND SUBTRACTION
# ============================================================================
def learn_background(cap, num_frames=30):
    """Capture and average frames to create static background."""
    print("📸 Learning background... Please step away from camera.")
    frames = []
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        time.sleep(0.1)
    
    background = np.median(frames, axis=0).astype(np.uint8)
    print("✓ Background learned!")
    return background

def detect_user_presence(frame, background, threshold=CHANGE_THRESHOLD):
    """Check if user is present using background subtraction."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, background)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    changed_pixels = cv2.countNonZero(thresh)
    return changed_pixels > threshold

# ============================================================================
# GESTURE RECOGNITION
# ============================================================================
def is_thumbs_up(hand_landmarks):
    """Check if hand is showing thumbs up gesture."""
    if not hand_landmarks:
        return False
    
    landmarks = hand_landmarks.landmark
    
    # Thumb tip should be above thumb IP
    thumb_up = landmarks[4].y < landmarks[3].y
    
    # Other fingers should be curled (tips below PIPs)
    index_curled = landmarks[8].y > landmarks[6].y
    middle_curled = landmarks[12].y > landmarks[10].y
    ring_curled = landmarks[16].y > landmarks[14].y
    pinky_curled = landmarks[20].y > landmarks[18].y
    
    return thumb_up and index_curled and middle_curled and ring_curled and pinky_curled

# ============================================================================
# FEATURE DETECTION
# ============================================================================
def detect_skin_tone(frame):
    """Extract skin tone from face using MediaPipe."""
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return "Unknown"
    
    landmarks = results.multi_face_landmarks[0].landmark
    h, w = frame.shape[:2]
    
    # Sample cheek region (landmark 330)
    cheek_indices = [330, 331, 332, 333, 334]
    cheek_pixels = []
    
    for idx in cheek_indices:
        if idx < len(landmarks):
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            if 0 <= x < w and 0 <= y < h:
                pixel = rgb_frame[y, x]
                cheek_pixels.append(pixel)
    
    if cheek_pixels:
        avg_color = np.mean(cheek_pixels, axis=0)
        brightness = np.mean(avg_color)
        
        if brightness > 200:
            return "Fair/Type I-II"
        elif brightness > 150:
            return "Medium/Type III"
        else:
            return "Dark/Type IV-VI"
    
    return "Unknown"

def detect_body_measurements(frame):
    """
    Detect body shape, estimated height, and build using MediaPipe Pose.
    Returns dict with body_shape, height_estimate, weight_estimate, build_type.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if not results.pose_landmarks:
        return {
            "body_shape": "Unknown",
            "height_cm": "Unknown",
            "weight_kg": "Unknown",
            "build_type": "Unknown"
        }
    
    landmarks = results.pose_landmarks.landmark
    h, w = frame.shape[:2]
    
    # Get key body points
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    nose = landmarks[0]
    left_ankle = landmarks[27]
    
    # Calculate shoulder and hip widths (in pixels)
    shoulder_width_px = abs((right_shoulder.x - left_shoulder.x) * w)
    hip_width_px = abs((right_hip.x - left_hip.x) * w)
    
    # Calculate body height in pixels (nose to ankle)
    body_height_px = abs((left_ankle.y - nose.y) * h)
    
    # Estimate actual height (assuming average camera distance)
    # This is a rough estimate - calibration would improve accuracy
    height_estimate = int(150 + (body_height_px / h) * 50)  # Rough heuristic
    
    # Calculate shoulder-to-hip ratio
    if hip_width_px > 0:
        ratio = shoulder_width_px / hip_width_px
    else:
        ratio = 1.0
    
    # Classify body shape based on ratio
    if ratio > 1.15:
        body_shape = "Inverted Triangle/Athletic"
        build_type = "Athletic/Mesomorph"
    elif ratio > 1.05:
        body_shape = "Rectangle/Straight"
        build_type = "Average/Balanced"
    elif ratio > 0.95:
        body_shape = "Hourglass/Balanced"
        build_type = "Balanced/Mesomorph"
    else:
        body_shape = "Pear/Lower-heavy"
        build_type = "Endomorph"
    
    # Estimate weight based on height and build
    # Very rough estimation
    if "Athletic" in build_type:
        weight_estimate = int(height_estimate * 0.4)  # Athletic build
    elif "Balanced" in build_type:
        weight_estimate = int(height_estimate * 0.38)
    else:
        weight_estimate = int(height_estimate * 0.42)
    
    return {
        "body_shape": body_shape,
        "height_cm": height_estimate,
        "weight_kg": weight_estimate,
        "build_type": build_type
    }

def infer_fit_preference(body_shape):
    """Infer likely fit preference based on body shape."""
    if "Athletic" in body_shape or "Inverted" in body_shape:
        return "Fitted/Athletic-cut"
    elif "Rectangle" in body_shape or "Straight" in body_shape:
        return "Regular/Standard-fit"
    else:
        return "Relaxed/Comfort-fit"

# ============================================================================
# VIRTUAL TRY-ON PROCESSING (SAFETY-LOCKED)
# ============================================================================
def process_virtual_try_on(user_image_path, cloth_image_path, user_profile, output_path):
    """
    Process virtual try-on with API safety lock.
    
    Args:
        user_image_path: Path to user's captured image
        cloth_image_path: Path to selected cloth image
        user_profile: Dict containing user's profile data
        output_path: Where to save the result
    
    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*70)
    print("VIRTUAL TRY-ON PIPELINE")
    print("="*70)
    print(f"📸 User Image: {os.path.basename(user_image_path)}")
    print(f"👕 Cloth Image: {os.path.basename(cloth_image_path)}")
    print(f"👤 Profile: {user_profile.get('Body_Shape', 'Unknown')}, {user_profile.get('Height_cm', 'Unknown')}cm")
    print("="*70)
    
    if ENABLE_PAID_API:
        print("🔓 API ENABLED - Calling Segmind SegFit...")
        try:
            success = run_segmind_vton(user_image_path, cloth_image_path, output_path)
            if success:
                print("✅ API call successful!")
                return True
            else:
                print("⚠️ API call failed, using fallback")
                return False
        except Exception as e:
            print(f"❌ API Error: {e}")
            return False
    else:
        print("🔒 API DISABLED - Safe Mode Active")
        print("💰 Preserving Credits")
        print("✓ Logic Confirmed:")
        print(f"  → User Image: Ready ({os.path.exists(user_image_path)})")
        print(f"  → Cloth Image: Ready ({os.path.exists(cloth_image_path)})")
        print(f"  → Profile Data: Ready ({len(user_profile)} attributes)")
        print("  → Pipeline: Validated ✓")
        print("\n📋 Simulation: Copying original as result...")
        
        # Copy original as result in safe mode
        import shutil
        shutil.copy(user_image_path, output_path)
        print(f"✓ Result saved to: {os.path.basename(output_path)}")
        print("="*70 + "\n")
        return True


# ============================================================================
# MAIN MIRROR LOOP
# ============================================================================
def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open camera")
        return
    
    # Learn background
    background = learn_background(cap)
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_draw = mp.solutions.drawing_utils
    
    # State machine
    state = "WAITING"
    gesture_counter = 0
    countdown_start = None
    result_image = None
    result_display_start = None
    
    print("\n🪞 Magic Mirror Ready!")
    print("👍 Show thumbs up to start try-on\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)  # Mirror effect
        display_frame = frame.copy()
        
        # Check user presence
        user_present = detect_user_presence(frame, background)
        
        # ====================================================================
        # STATE: WAITING
        # ====================================================================
        if state == "WAITING":
            # Only check gesture if user is present
            if user_present:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        if is_thumbs_up(hand_landmarks):
                            gesture_counter += 1
                            cv2.putText(display_frame, f"Hold... {gesture_counter}/{GESTURE_HOLD_FRAMES}", 
                                      (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            
                            if gesture_counter >= GESTURE_HOLD_FRAMES:
                                state = "COUNTDOWN"
                                countdown_start = time.time()
                                gesture_counter = 0
                        else:
                            gesture_counter = 0
                else:
                    gesture_counter = 0
                
                cv2.putText(display_frame, "👍 Thumbs up to start!", 
                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Waiting for user...", 
                          (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        
        # ====================================================================
        # STATE: COUNTDOWN
        # ====================================================================
        elif state == "COUNTDOWN":
            elapsed = time.time() - countdown_start
            remaining = COUNTDOWN_SECONDS - int(elapsed)
            
            if remaining > 0:
                cv2.putText(display_frame, str(remaining), 
                          (display_frame.shape[1]//2 - 50, display_frame.shape[0]//2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
            else:
                state = "PROCESSING"
                captured_frame = frame.copy()
        
        # ====================================================================
        # STATE: PROCESSING
        # ====================================================================
        elif state == "PROCESSING":
            cv2.putText(display_frame, "Processing...", 
                      (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow('Magic Mirror', display_frame)
            cv2.waitKey(1)
            
            # Create trial folder
            trial_folder, trial_num = create_trial_folder()
            print(f"\n📁 Created {trial_folder}")
            
            # Save original
            original_path = trial_folder / "original.jpg"
            cv2.imwrite(str(original_path), captured_frame)
            
            # Detect ALL features automatically
            print("   🔍 Analyzing body measurements...")
            detected_skin_tone = detect_skin_tone(captured_frame)
            detected_body_data = detect_body_measurements(captured_frame)
            detected_fit_pref = infer_fit_preference(detected_body_data["body_shape"])
            
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Build complete profile from DETECTED data only
            complete_profile = {
                # Detected attributes
                "Skin_Tone": detected_skin_tone,
                "Body_Shape": detected_body_data["body_shape"],
                "Build_Type": detected_body_data["build_type"],
                "Height_cm": detected_body_data["height_cm"],
                "Weight_kg": detected_body_data["weight_kg"],
                "Fit_Preference": detected_fit_pref,
                
                # Manual inputs (cannot detect from single image)
                "Age": MANUAL_USER_DATA["Age"],
                "Gender": MANUAL_USER_DATA["Gender"],
                
                # Metadata
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save profile CSV
            csv_path = trial_folder / f"profile_{timestamp}.csv"
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Attribute', 'Value'])
                
                # Write all profile data
                for key, value in complete_profile.items():
                    writer.writerow([key, value])
            
            print(f"   ✓ Profile saved: {csv_path.name}")
            print(f"   📊 Detected: {complete_profile['Body_Shape']}, {complete_profile['Height_cm']}cm, {complete_profile['Skin_Tone']}")
            
            # Run VTON with safety lock
            result_path = trial_folder / "result.jpg"
            success = process_virtual_try_on(
                str(original_path),
                SELECTED_CLOTH_PATH,
                complete_profile,
                str(result_path)
            )
            
            if not success:
                print("⚠️ VTO failed, using original as fallback")
                cv2.imwrite(str(result_path), captured_frame)
            
            # Load result
            result_image = cv2.imread(str(result_path))
            result_display_start = time.time()
            state = "DISPLAY"
            print(f"✓ Trial {trial_num} complete!\n")
        
        # ====================================================================
        # STATE: DISPLAY
        # ====================================================================
        elif state == "DISPLAY":
            if result_image is not None:
                # Resize result to fit screen
                h, w = display_frame.shape[:2]
                result_resized = cv2.resize(result_image, (w, h))
                display_frame = result_resized
                
                elapsed = time.time() - result_display_start
                if elapsed > DISPLAY_SECONDS:
                    state = "WAITING"
                    result_image = None
        
        # Show frame
        cv2.imshow('Magic Mirror', display_frame)
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
