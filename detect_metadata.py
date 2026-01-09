import cv2
import os
import subprocess
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Helper to download file using curl
def download_file_from_mirrors(filename):
    filepath = os.path.join(MODEL_DIR, filename)
    # Check if exists and valid (>1KB)
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
        return filepath

    # List of Mirrors to try
    mirrors = [
        "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/", # Working Mirror!
        "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/",
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/", 
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/", 
    ]
    
    # Specific known-good URLs for the caffemodels if generic mirrors fail
    # Note: 'raw' github content is the best bet.
    
    url_candidates = [m + filename for m in mirrors]
    
    # Special overrides for specific files if needed
    if filename == "age_net.caffemodel":
        url_candidates.insert(0, "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/age_net.caffemodel")
    if filename == "gender_net.caffemodel":
        url_candidates.insert(0, "https://raw.githubusercontent.com/eveningglow/age-and-gender-classification/master/model/gender_net.caffemodel")
    
    for url in url_candidates:
        print(f"Trying to download {filename} from {url}...")
        try:
            # Use curl -L (follow redirects) -f (fail on error) -o (output)
            cmd = ["curl", "-L", "-f", url, "-o", filepath]
            subprocess.run(cmd, check=True, capture_output=True)
            
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
                print(f"Success: Downloaded {filename}")
                return filepath
            else:
                # Cleanup empty files
                if os.path.exists(filepath): os.remove(filepath)
        except Exception:
            pass
            
    print(f"Error: Could not download {filename} from any source.")
    return None

FILES = {
    "age_proto": "age_deploy.prototxt",
    "age_model": "age_net.caffemodel",
    "gender_proto": "gender_deploy.prototxt",
    "gender_model": "gender_net.caffemodel",
}

# Mean values for normalization
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Labels
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

def ensure_models():
    paths = {}
    for key, filename in FILES.items():
        path = download_file_from_mirrors(filename)
        if path:
            paths[key] = path
        else:
            print(f"FAILED to download {filename}")
            return None
    return paths

def detect_face_haar(image_gray):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image_gray, 1.1, 4)
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    return faces[0]

def analyze_dataset(image_path):
    print(f"Analyzing {image_path}...")
    paths = ensure_models()
    if not paths:
        print("Cannot run analysis due to missing models.")
        return None

    try:
        age_net = cv2.dnn.readNet(paths["age_model"], paths["age_proto"])
        gender_net = cv2.dnn.readNet(paths["gender_model"], paths["gender_proto"])
    except Exception as e:
        print(f"Failed to load models: {e}")
        return None

    frame = cv2.imread(image_path)
    if frame is None:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_rect = detect_face_haar(gray)
    if face_rect is None:
        print("No face detected.")
        return {"Age": 25, "Gender": "Female"} 
        
    x, y, w, h = face_rect
    padding = 20
    face_img = frame[max(0, y-padding):min(y+h+padding, frame.shape[0]-1),
                     max(0, x-padding):min(x+w+padding, frame.shape[1]-1)]
                     
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]
    print(f"Gender: {gender}")

    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_LIST[age_preds[0].argmax()]
    print(f"Age Group: {age}")
    
    age_map = {
        '(0-2)': 1, '(4-6)': 5, '(8-12)': 10, '(15-20)': 18,
        '(25-32)': 28, '(38-43)': 40, '(48-53)': 50, '(60-100)': 70
    }
    est_age = age_map.get(age, 25)

    return {"Age": est_age, "Gender": gender}

if __name__ == "__main__":
    import sys
    img = r"C:\Users\dhaks\OneDrive\Documents\MIRAI\model\my photos\IMG_1337.jpg"
    if len(sys.argv) > 1:
        img = sys.argv[1]
    analyze_dataset(img)
