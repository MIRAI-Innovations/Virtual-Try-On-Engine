import mediapipe as mp
try:
    import mediapipe.solutions
    print("Imported mediapipe.solutions successfully")
    mp.solutions = mediapipe.solutions
    print(f"Solutions pose: {mp.solutions.pose}")
except ImportError as e:
    print(f"Explicit import failed: {e}")
except AttributeError as e:
    print(f"Attribute error: {e}")
