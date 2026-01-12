import os
import argparse
import subprocess
import sys

def run_command(command):
    print(f"\n[EXEC] {command}")
    process = subprocess.Popen(command, shell=True)
    process.wait()
    if process.returncode != 0:
        print(f"!!! ERROR !!! Command failed with return code {process.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Robust VITON-HD Demo Runner')
    parser.add_argument('--img', type=str, required=True, help='Full path to person image')
    parser.add_argument('--cloth', type=str, required=True, help='Filename of cloth (e.g. 00013_00.jpg)')
    # Default to looking one level up, but we will verify this
    parser.add_argument('--dataset_dir', type=str, default='../datasets', help='Path to datasets directory')
    args = parser.parse_args()

    # --- 1. RESOLVE ABSOLUTE PATHS (The Fix) ---
    # Convert everything to absolute paths so there is NO confusion
    current_dir = os.getcwd()
    
    # Handle Image Path
    img_path = os.path.abspath(args.img)
    if not os.path.exists(img_path):
        print(f"!!! ERROR: Person image not found at: {img_path}")
        sys.exit(1)

    # Handle Dataset Directory
    # Check the user provided one first
    ds_dir = os.path.abspath(args.dataset_dir)
    if not os.path.exists(ds_dir):
        # Fallback: Check if it's inside the current folder
        internal_ds = os.path.join(current_dir, 'datasets')
        if os.path.exists(internal_ds):
            print(f"Found datasets folder inside current directory. Switching to: {internal_ds}")
            ds_dir = internal_ds
        else:
            print(f"!!! ERROR: Could not find datasets folder at: {ds_dir}")
            sys.exit(1)

    # Verify Cloth exists
    cloth_path = os.path.join(ds_dir, 'test', 'cloth', args.cloth)
    if not os.path.exists(cloth_path):
        print(f"!!! ERROR: Cloth image not found at: {cloth_path}")
        print(f"Make sure you moved {args.cloth} into {os.path.join(ds_dir, 'test', 'cloth')}")
        sys.exit(1)

    print(f"--- CONFIGURATION ---")
    print(f"Work Dir:  {current_dir}")
    print(f"Person:    {img_path}")
    print(f"Cloth:     {args.cloth}")
    print(f"Datasets:  {ds_dir}")
    print(f"---------------------")

    # --- 2. UPDATE PAIR LIST ---
    # We write the pair list to the detected dataset directory
    pair_file_path = os.path.join(ds_dir, 'test_pairs.txt')
    img_name_only = os.path.basename(img_path)
    # Ensure extension is .jpg for the list (VITON requirement)
    img_base = os.path.splitext(img_name_only)[0] + '.jpg'
    
    with open(pair_file_path, 'w') as f:
        f.write(f"{img_base} {args.cloth}")
    print(f"[INFO] Updated {pair_file_path}")

    # --- 3. RUN PREPROCESS ---
    # We pass the Absolute Output Directory to ensure it goes to the right place
    test_dir = os.path.join(ds_dir, 'test')
    
    # IMPORTANT: We quote the paths to handle spaces in "my photos"
    cmd_pre = f'"{sys.executable}" preprocess.py --image_path "{img_path}" --output_dir "{test_dir}"'
    run_command(cmd_pre)

    # --- 4. RUN TEST ---
    # We pass the Absolute Dataset Directory
    cmd_test = f'"{sys.executable}" test.py --name demo --dataset_mode test --dataset_list test_pairs.txt --dataset_dir "{ds_dir}" --checkpoint_dir ./checkpoints'
    run_command(cmd_test)

    print("\n--- SUCCESS! Check ./results/demo for your output ---")

if __name__ == '__main__':
    main()