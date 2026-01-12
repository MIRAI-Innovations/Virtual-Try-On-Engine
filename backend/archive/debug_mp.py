import sys
import traceback

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")

try:
    print("Attempting to import mediapipe...")
    import mediapipe as mp
    print("Success: import mediapipe")
    print(f"MediaPipe Version: {mp.__version__}")
    
    print("Attempting to access mp.solutions...")
    print(mp.solutions)
    print("Success: mp.solutions accessed")

except Exception:
    print("FAILED to import/use mediapipe:")
    traceback.print_exc()
