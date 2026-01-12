import os
import argparse
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import mediapipe as mp  # Standard import
from tqdm import tqdm

# Add HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess images for VITON-HD')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output_dir', type=str, default='./datasets/test', help='Output directory base')
    return parser.parse_args()


def generate_openpose_data(image_path, output_dir_img, output_dir_json):
    # --- MediaPipe Tasks API (Python 3.13 Compatible) ---
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    # Download the model file if it doesn't exist
    model_path = 'pose_landmarker_full.task'
    if not os.path.exists(model_path):
        import urllib.request
        print(f"Downloading MediaPipe Pose Model to {model_path}...")
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download MediaPipe model: {e}")

    # Create an PoseLandmarker object.
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    detector = vision.PoseLandmarker.create_from_options(options)

    # VITON-HD target resolution
    target_w, target_h = 768, 1024
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    detection_result = detector.detect(mp_image)
    
    img_name = os.path.basename(image_path)
    file_name = os.path.splitext(img_name)[0]

    keypoints = []
    
    # Check if any landmarks were detected
    if detection_result.pose_landmarks:
        # Tasks API returns a list of lists (one per person), we take the first
        landmarks = detection_result.pose_landmarks[0]
        
        # Helper to get [x, y, conf]
        # Note: Tasks API landmarks have x, y, z, visibility, presence. 
        # We use visibility for confidence/presence.
        def get_kp(idx):
            if idx < len(landmarks):
                lm = landmarks[idx]
                return [lm.x * target_w, lm.y * target_h, lm.visibility]
            return [0, 0, 0]

        # 0 Nose
        keypoints.extend(get_kp(0)) 
        
        # 1 Neck - Average of shoulders (11 & 12)
        l_sh = landmarks[11]
        r_sh = landmarks[12]
        neck_x = (l_sh.x + r_sh.x) / 2
        neck_y = (l_sh.y + r_sh.y) / 2
        neck_conf = (l_sh.visibility + r_sh.visibility) / 2
        keypoints.extend([neck_x * target_w, neck_y * target_h, neck_conf])
        
        # Upper Body: 2:RShoulder(12), 3:RElbow(14), 4:RWrist(16), 5:LShoulder(11), 6:LElbow(13), 7:LWrist(15)
        keypoints.extend(get_kp(12)) # RShoulder
        keypoints.extend(get_kp(14)) # RElbow
        keypoints.extend(get_kp(16)) # RWrist
        keypoints.extend(get_kp(11)) # LShoulder
        keypoints.extend(get_kp(13)) # LElbow
        keypoints.extend(get_kp(15)) # LWrist
        
        # Lower Body: 8:MidHip (Avg of 23 & 24)
        l_hip = landmarks[23]
        r_hip = landmarks[24]
        mid_x = (l_hip.x + r_hip.x) / 2
        mid_y = (l_hip.y + r_hip.y) / 2
        mid_conf = (l_hip.visibility + r_hip.visibility) / 2
        keypoints.extend([mid_x * target_w, mid_y * target_h, mid_conf])
        
        # 9:RHip(24), 10:RKnee(26), 11:RAnkle(28)
        keypoints.extend(get_kp(24)) # RHip
        keypoints.extend(get_kp(26)) # RKnee
        keypoints.extend(get_kp(28)) # RAnkle
        
        # 12:LHip(23), 13:LKnee(25), 14:LAnkle(27)
        keypoints.extend(get_kp(23)) # LHip
        keypoints.extend(get_kp(25)) # LKnee
        keypoints.extend(get_kp(27)) # LAnkle
        
        # Face/Feet (OpenPose Indices 15-24)
        # Mapping MediaPipe (MP) to OpenPose (OP) roughly
        # OP 15: REye -> MP 5
        keypoints.extend(get_kp(5))
        # OP 16: LEye -> MP 2
        keypoints.extend(get_kp(2))
        # OP 17: REar -> MP 8
        keypoints.extend(get_kp(8))
        # OP 18: LEar -> MP 7
        keypoints.extend(get_kp(7))
        # OP 19: LBigToe -> MP 31
        keypoints.extend(get_kp(31))
        # OP 20: LSmallToe -> MP 31 (Approx)
        keypoints.extend([0, 0, 0])
        # OP 21: LHeel -> MP 29
        keypoints.extend(get_kp(29))
        # OP 22: RBigToe -> MP 32
        keypoints.extend(get_kp(32))
        # OP 23: RSmallToe -> MP 32 (Approx)
        keypoints.extend([0, 0, 0])
        # OP 24: RHeel -> MP 30
        keypoints.extend(get_kp(30))
        
        # Render Pose Image
        # To reuse existing rendering loop, we need 'results.pose_landmarks' structure or just rely on 'keypoints'
        # The existing loop uses 'keypoints' list, so we can reuse it!


        # Render Pose Image
        pose_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        pairs = [
            (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
            (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
            (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
            (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)
        ]
        
        colors = [
            [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
            [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
            [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], [255, 0, 0], [255, 31, 0], [0, 255, 0],
            [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]
        ]

        for i, (p1, p2) in enumerate(pairs):
            if keypoints[p1*3+2] > 0.05 and keypoints[p2*3+2] > 0.05:
                pt1 = (int(keypoints[p1*3]), int(keypoints[p1*3+1]))
                pt2 = (int(keypoints[p2*3]), int(keypoints[p2*3+1]))
                cv2.line(pose_img, pt1, pt2, colors[i], 8)

        for i in range(25):
            if keypoints[i*3+2] > 0.05:
                pt = (int(keypoints[i*3]), int(keypoints[i*3+1]))
                cv2.circle(pose_img, pt, 8, colors[i % len(colors)], -1)

        pose_img_bgr = cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir_img, file_name + '_rendered.png'), pose_img_bgr)
        
    else:
        keypoints = [0.0] * 75
        pose_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(output_dir_img, file_name + '_rendered.png'), pose_img)

    # Save JSON
    data = {
        "version": 1.3,
        "people": [{
            "person_id": [-1],
            "pose_keypoints_2d": keypoints,
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": []
        }]
    }
    
    with open(os.path.join(output_dir_json, file_name + '_keypoints.json'), 'w') as f:
        json.dump(data, f)
        
    return pose_img, keypoints

def generate_segmentation_data(image_path, output_dir_parse, keypoints=None):
    # Use SegFormer pretrained on clothes (mattmdjaga) which matches our mapping logic
    model_name = "mattmdjaga/segformer_b2_clothes"
    try:
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    
    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy().astype(np.uint8)
    
    cihp_map = np.zeros_like(pred_seg)
    mapping = {0:0, 1:1, 2:2, 3:4, 4:5, 5:12, 6:9, 7:6, 8:0, 9:18, 10:19, 11:13, 12:16, 13:17, 14:14, 15:15, 16:0, 17:7}
    for src, dst in mapping.items():
        cihp_map[pred_seg == src] = dst

    # Synthetic Neck Generation
    if keypoints and len(keypoints) >= 6:
        neck_conf = keypoints[5]
        if neck_conf > 0.1:
            neck_mask = Image.new('L', (cihp_map.shape[1], cihp_map.shape[0]), 0)
            draw = ImageDraw.Draw(neck_mask)
            
            if len(keypoints) >= 18:
                r_sh = (keypoints[6], keypoints[7])
                l_sh = (keypoints[15], keypoints[16])
                nose = (keypoints[0], keypoints[1])
                base_mid_x = (r_sh[0] + l_sh[0]) / 2
                base_mid_y = (r_sh[1] + l_sh[1]) / 2
                neck_w = abs(r_sh[0] - l_sh[0]) / 3
                
                p1 = (base_mid_x - neck_w/2, nose[1] + (base_mid_y - nose[1]) * 0.4)
                p2 = (base_mid_x + neck_w/2, nose[1] + (base_mid_y - nose[1]) * 0.4)
                p3 = (l_sh[0] + (base_mid_x - l_sh[0]) * 0.5, base_mid_y)
                p4 = (r_sh[0] + (base_mid_x - r_sh[0]) * 0.5, base_mid_y)
                draw.polygon([p1, p2, p3, p4], fill=255)
            else:
                draw.ellipse([keypoints[3]-20, keypoints[4]-40, keypoints[3]+20, keypoints[4]+10], fill=255)
            
            neck_arr = np.array(neck_mask)
            cihp_map[(neck_arr > 0) & ((cihp_map == 0) | (cihp_map == 13))] = 10
        
    img_name = os.path.basename(image_path)
    file_name = os.path.splitext(img_name)[0]
    Image.fromarray(cihp_map, mode='L').save(os.path.join(output_dir_parse, file_name + '.png'))


def main():
    args = parse_args()
    
    out_img_dir = os.path.join(args.output_dir, 'image')
    out_pose_img_dir = os.path.join(args.output_dir, 'openpose_img')
    out_pose_json_dir = os.path.join(args.output_dir, 'openpose_json')
    out_parse_dir = os.path.join(args.output_dir, 'image-parse')
    
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_pose_img_dir, exist_ok=True)
    os.makedirs(out_pose_json_dir, exist_ok=True)
    os.makedirs(out_parse_dir, exist_ok=True)
    
    print(f"Processing {args.image_path}...")
    target_size = (768, 1024)
    img_name = os.path.basename(args.image_path)
    file_name = os.path.splitext(img_name)[0]
    
    src_img = Image.open(args.image_path).convert('RGB')
    src_img = src_img.resize(target_size, Image.LANCZOS)
    jpg_path = os.path.join(out_img_dir, file_name + '.jpg')
    src_img.save(jpg_path)

    print("Generating Pose...")
    pose_img, keypoints = generate_openpose_data(jpg_path, out_pose_img_dir, out_pose_json_dir)

    print("Enhancing contrast for better white-shirt segmentation...")
    img_gray = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gray)
    contrast_path = jpg_path.replace('.jpg', '_contrast.jpg')
    cv2.imwrite(contrast_path, img_clahe)
    
    print("Generating Segmentation...")
    generate_segmentation_data(contrast_path, out_parse_dir, keypoints=keypoints)
    
    if os.path.exists(contrast_path): os.remove(contrast_path)
    
    parse_path = os.path.join(out_parse_dir, file_name + '.png')
    if os.path.exists(parse_path):
        ps_img = Image.open(parse_path)
        if ps_img.size != target_size:
            ps_img.resize(target_size, Image.NEAREST).save(parse_path)
    
    print("Pre-processing Complete.")

if __name__ == "__main__":
    main()