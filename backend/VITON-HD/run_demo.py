import os
import argparse
import subprocess
import sys

def run_command(command):
    print(f"Running: {command}")
    # Using shell=True requires proper quoting for Windows paths with spaces
    process = subprocess.Popen(command, shell=True)
    process.wait()
    if process.returncode != 0:
        raise Exception(f"Command failed with return code {process.returncode}")

def main():
    parser = argparse.ArgumentParser(description='Run VITON-HD Demo')
    parser.add_argument('--img', type=str, required=True, help='Path to image of person')
    parser.add_argument('--cloth', type=str, required=True, help='Filename of cloth in datasets/test/cloth (e.g. 12345_00.jpg)')
    parser.add_argument('--dataset_dir', type=str, default='../datasets', help='Path to datasets directory')
    args = parser.parse_args()

    # Ensure dataset_dir is absolute to avoid confusion
    dataset_dir = os.path.abspath(args.dataset_dir)
    test_dir = os.path.join(dataset_dir, 'test')
    
    # Check if dataset dir exists
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory not found at: {dataset_dir}")
        print("Please check your --dataset_dir path.")
        sys.exit(1)

    # 1. Preprocess
    print("--- 1. Running Pre-processing ---")
    # FIX: Added quotes around sys.executable and paths to handle spaces in "Program Files" or "my photos"
    cmd_pre = f'"{sys.executable}" preprocess.py --image_path "{args.img}" --output_dir "{test_dir}"'
    run_command(cmd_pre)

    # 2. Update test_pairs.txt
    print("--- 2. Updating Pair List ---")
    # Standardize image name for the text file (VITON expects .jpg in the list)
    img_basename = os.path.basename(args.img)
    img_name_clean = os.path.splitext(img_basename)[0] + '.jpg'
    pair_line = f"{img_name_clean} {args.cloth}"
    
    pairs_file = os.path.join(dataset_dir, 'test_pairs.txt')
    with open(pairs_file, 'w') as f:
        f.write(pair_line)
    
    # 3. Run Test
    print("--- 3. Running VITON-HD Test ---")
    # FIX: Added quotes here as well
    # Added --checkpoint_dir ./checkpoints explicitly to be safe
    cmd_test = f'"{sys.executable}" test.py --name demo --dataset_mode test --dataset_list test_pairs.txt --dataset_dir "{dataset_dir}" --checkpoint_dir ./checkpoints'
    run_command(cmd_test)

    print("--- Done! Check ./results/demo for output ---")

if __name__ == '__main__':
    main()