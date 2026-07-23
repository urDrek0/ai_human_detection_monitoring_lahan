# Configuration settings for the human detection subsystem

#MODEL_PATH = "yolo26n_float32.tflite"
MODEL_PATH = "best.tflite"

# Detection and NMS Hyperparameters
CONFIDENCE_THRESHOLD = 0.60
NMS_IOU_THRESHOLD = 0.45

CONF_THRESHOLD = CONFIDENCE_THRESHOLD
NMS_THRESHOLD = NMS_IOU_THRESHOLD

# Geometric and temporal filters to reduce false positives
MIN_BOX_AREA = 1500
MIN_ASPECT_RATIO = 0.8
MAX_ASPECT_RATIO = 4.5
TEMPORAL_CONFIRMATION_FRAMES = 3
DETECTION_COOLDOWN_SECONDS = 0

# Performance and Hardware
NUM_THREADS = 4

# WebSocket Client Configuration (Target Backend)
BACKEND_HOST = "localhost"
BACKEND_PORT = 5000
MAX_WS_SIZE = 20 * 1024 * 1024  # 20 MB

# Camera Detection Mode ("AI" or "Pixel")
CAMERA_DETECTION_MODE = "AI"

# Pixel Comparison Settings
PIXEL_MOTION_SENSITIVITY = 10
PIXEL_MOTION_MIN_AREA = 10.0
PIXEL_MOTION_MODE = 0  # 0: Static Reference, 1: Frame-to-Frame
PIXEL_MOTION_MERGE = False
PIXEL_MOTION_RESET_INTERVAL = 1
PIXEL_MOTION_CLUSTER_DIST = 50
PIXEL_MOTION_MIN_SIZE = 10


