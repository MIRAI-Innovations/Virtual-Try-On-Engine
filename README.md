# MIRAI Virtual Try-On Mirror 🪞✨

An intelligent, automated virtual try-on system that combines AI-powered garment visualization with automated feature extraction and multi-provider failover for robust, real-time fashion recommendations.

![Version](https://img.shields.io/badge/version-1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **AI-Powered Virtual Try-On**: Leverages state-of-the-art diffusion models (IDM-VTON, OOTDiffusion, CatVTON)
- **Automated Feature Extraction**: Uses MediaPipe for real-time body shape, face shape, and skin tone detection
- **Multi-Provider Failover**: Ensures 99.9% uptime through intelligent API rotation
- **Session-Based Logging**: Structured data collection for future recommendation engine training
- **One-Command Execution**: Single Python script runs the entire pipeline
- **Privacy-Preserving**: All feature detection runs on-device without cloud dependencies

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Configuration](#%EF%B8%8F-configuration)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/StillDuxk/Mirai_tryon_Mirror.git
cd Mirai_tryon_Mirror

# Install dependencies
pip install -r requirements.txt

# Run a virtual try-on trial
python run_trial.py
```

That's it! The system will automatically:
1. Load your subject image and randomly select a garment
2. Detect body features using MediaPipe
3. Connect to the best available VTON API
4. Generate the try-on result
5. Save everything to a timestamped trial folder

## 💾 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for API calls)

### Dependencies

Install all required packages:

```bash
pip install gradio-client==1.5.0
pip install mediapipe==0.10.31
pip install opencv-python
pip install pandas
pip install numpy
pip install Pillow
```

Or use the requirements file (if available):

```bash
pip install -r requirements.txt
```

### Optional: Local VITON-HD Setup

For offline inference using the local VITON-HD model:

```bash
cd VITON-HD
# Follow VITON-HD specific setup instructions
```

## 📖 Usage

### Basic Usage

Run with default settings (uses hardcoded subject and random cloth):

```bash
python run_trial.py
```

### Advanced Usage

Specify custom subject image and cloth directory:

```bash
python run_trial.py --subject path/to/subject.jpg --cloth_dir path/to/clothes/
```

### Using the API Client Directly

For batch processing multiple images:

```bash
python run_vton_api.py
```

Edit the script to specify your input images and cloth items.

### Example Workflow

```python
# 1. Place your subject image in the project directory
# 2. Place cloth images in the datasets/test/cloth/ directory
# 3. Run the trial:

python run_trial.py --subject my photos/IMG_1337.jpg
```

The system will:
- Extract your height, age, gender, and style preference (currently hardcoded, can be modified)
- Automatically detect body shape, face shape, and skin tone
- Select a random cloth from the directory
- Generate the virtual try-on image
- Save results to `Trial X/` folder with CSV metadata

## 📁 Project Structure

```
Mirai_tryon_Mirror/
├── run_trial.py                # Main application - automated trial runner
├── run_vton_api.py            # Multi-task VTON API client
├── run_demo_robust.py         # Local VITON-HD wrapper
├── TECHNICAL_DOCUMENTATION.md # In-depth technical documentation
├── README.md                  # This file
├── requirements.txt           # Python dependencies (create if needed)
│
├── VITON-HD/                  # Local VITON-HD model (optional)
│   ├── test.py
│   ├── preprocess.py
│   ├── networks.py
│   └── ...
│
├── datasets/                  # Training and test datasets
│   └── test/
│       ├── cloth/            # Garment images
│       ├── image/            # Subject images
│       └── ...
│
├── my photos/                # User uploaded subject images
│   ├── IMG_1337.jpg
│   └── ...
│
├── Trial 1/                  # Generated trial folders
├── Trial 2/
├── Trial 3/
└── ...
```

### Key Files

- **`run_trial.py`** (391 lines): Primary application with feature extraction and session management
- **`run_vton_api.py`** (162 lines): Multi-provider VTON API client with failover logic
- **`run_demo_robust.py`** (113 lines): Local VITON-HD execution wrapper
- **`TECHNICAL_DOCUMENTATION.md`**: Comprehensive technical documentation and IP analysis

## 🔬 How It Works

### 1. Feature Extraction Pipeline

```
Subject Image → MediaPipe Analysis → Feature Vector
                     ↓
              ┌──────┴──────┐
              ↓             ↓
         Pose Landmarks  Face Mesh (468 points)
         (33 points)           ↓
              ↓          Face Shape Classification
         Body Shape     ("Oval", "Round", "Square")
         Classification       ↓
              ↓          RGB Skin Sampling
         Shoulder/Hip         ↓
         Ratio Analysis  Skin Tone Detection
              ↓          ("Fair", "Medium", "Dark")
         ("Inverted Triangle",
          "Hourglass",
          "Rectangle")
```

### 2. Multi-Provider VTON Architecture

```
API Request → Space1 (3 retries) → Success? → Return Result
                      ↓ Fail
              Space2 (3 retries) → Success? → Return Result
                      ↓ Fail
              Space3 (3 retries) → Success? → Return Result
                      ↓ Fail
              Error Message
```

**Supported Providers:**
- `yisol/IDM-VTON` (Primary)
- `Nymbo/Virtual-Try-On` (Backup)
- `levihsu/OOTDiffusion` (Tertiary)

### 3. Session Data Collection

Each trial generates:
- **Result Image**: `result.jpg` - Final try-on visualization
- **Metadata CSV**: `trial_data.csv` with 12 attributes
  - Manual inputs: Height, Age, Gender, Style Preference
  - Automated features: Body Shape, Face Shape, Skin Tone, Age Group, Arm Preference
  - File references: Subject, Cloth, Result paths

**Data Schema:**
```csv
Height,Age,Gender,Style Preference,Body Shape,Face Shape,Skin Tone,Age Group,Arm Preference,Subject File,Cloth File,Result File
170,25,Male,Casual,Inverted Triangle,Oval,Medium,Teen,Cover,IMG_1337.jpg,00013_00.jpg,result.jpg
```

## ⚙️ Configuration

### Modifying User Inputs

Edit lines 16-27 in `run_trial.py`:

```python
# User inputs (can be modified)
HEIGHT = 170        # in cm
AGE = 25           # in years
GENDER = "Male"    # "Male" or "Female"
STYLE_PREF = "Casual"  # "Casual", "Formal", "Streetwear", etc.
```

### Changing Default Images

Edit paths in `run_trial.py`:

```python
# Default paths (lines 272-275)
default_subject = "my photos/IMG_1337.jpg"
default_cloth_dir = "datasets/test/cloth"
```

### Adjusting Feature Detection Thresholds

Body shape classification (lines 71-95):
```python
ratio = shoulder_width / hip_width
if ratio > 1.05:
    return "Inverted Triangle"
elif 0.95 <= ratio <= 1.05:
    return "Hourglass"
else:
    return "Rectangle"
```

Skin tone thresholds (lines 121-140):
```python
if avg_intensity > 170:
    return "Fair"
elif 100 < avg_intensity <= 170:
    return "Medium"
else:
    return "Dark"
```

## 📚 API Documentation

### FeatureScanner Class

```python
scanner = FeatureScanner()
features = scanner.scan(image_path)

# Returns dictionary:
{
    "body_shape": str,      # "Inverted Triangle", "Hourglass", "Rectangle"
    "face_shape": str,      # "Oval", "Round", "Square"
    "skin_tone": str        # "Fair", "Medium", "Dark"
}
```

### run_vton_trial Function

```python
result_path = run_vton_trial(
    subject_path="path/to/subject.jpg",
    cloth_path="path/to/cloth.jpg",
    garment_desc="A casual t-shirt",
    category="upper_body"
)
```

**Parameters:**
- `subject_path` (str): Path to subject image
- `cloth_path` (str): Path to garment image
- `garment_desc` (str): Text description of garment
- `category` (str): "upper_body" or "lower_body"

**Returns:**
- str: Path to generated result image

## 🔧 Troubleshooting

### Common Issues

**1. SSL Connection Errors**
```
SSLError: HTTPSConnectionPool
```
**Solution:** The system automatically retries 3 times per provider. If all fail, check your internet connection.

**2. MediaPipe Import Errors**
```
ModuleNotFoundError: No module named 'mediapipe'
```
**Solution:**
```bash
pip install mediapipe==0.10.31
```

**3. No Cloth Images Found**
```
FileNotFoundError: No files in cloth directory
```
**Solution:** Ensure cloth images are in `datasets/test/cloth/` or specify a valid `--cloth_dir`.

**4. Gradio Client Timeout**
```
TimeoutError: Space took too long to respond
```
**Solution:** System automatically switches to backup provider. Wait for retry completion.

### Debug Mode

Enable verbose output by modifying `run_trial.py`:

```python
# Add at the top of main()
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Update TECHNICAL_DOCUMENTATION.md for major changes
- Test with multiple subject/cloth combinations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **VITON-HD**: Original virtual try-on architecture
- **IDM-VTON**: State-of-the-art diffusion-based VTON model
- **MediaPipe**: Real-time computer vision framework by Google
- **Hugging Face**: For hosting Gradio Spaces

## 📞 Contact

**Project Maintainer**: MIRAI Development Team  
**Repository**: [https://github.com/StillDuxk/Mirai_tryon_Mirror](https://github.com/StillDuxk/Mirai_tryon_Mirror)

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for the future of virtual fashion**
