# Requirements Guide

This project depends on the following Python libraries. Run the command below to install all of them.

## System Requirements
1.  **Python 3.10+**: For the VTON Engine.
2.  **Node.js & npm**: For the API Server. [Download here](https://nodejs.org/)

## Python Installation Command
```bash
pip install mediapipe gradio_client pandas opencv-python pillow requests
```

## Detailed Dependencies
-   **opencv-python**: For computer vision and image processing.
-   **mediapipe**: For body tracking, face mesh, and pose estimation.
-   **gradio_client**: To interact with the Hugging Face Spaces (VTON models).
-   **pandas**: For handling data and saving session CSVs.
-   **requests**: For HTTP requests to the Segmind API (Paid Mode).
-   **pillow**: For image manipulation (smart resizing).
