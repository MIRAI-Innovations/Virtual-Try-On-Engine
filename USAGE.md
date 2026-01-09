# MIRAI System Usage Guide

The system now has a unified entry point `main.py` that handles both the Magic Mirror and the Local VTON trials.

## 1. Magic Mirror Mode (Interactive - Paid API Enabled)
Runs the interactive mirror application with webcam feed.

**Important:** This mode consumes Segmind API credits. Ensure you have set your `SEGMIND_API_KEY` environment variable or defined it in `commercial_vton.py`.

```bash
python main.py --mode api
```

## 2. Local VTON Mode (Headless)
Runs the Virtual Try-On trial using local images. It automatically tries 3 different models (OOTDiffusion, IDM-VTON, CatVTON) until one succeeds.

**Syntax:**
```bash
python main.py --mode local --image "PATH_TO_IMAGE" --cloth "PATH_TO_CLOTH" --category [Upper-body|Lower-body|Dress]
```

**Example:**
```bash
python main.py --mode local --image "my photos/IMG_1522.jpg" --cloth "datasets/test/cloth/00035_00.jpg" --category Upper-body
```

### Optional Arguments for Local Mode
-   `--step 30`: Quality steps (default: 30)
-   `--scale 2.5`: Guidance scale (default: 2.5)
-   `--gender Female`: Override gender detection
-   `--age 25`: Override age detection

## OUTPUT
Results are saved in the `Mirror_Sessions/Trial_X` folder.
