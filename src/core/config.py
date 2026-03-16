import os
from dotenv import load_dotenv

load_dotenv()

DEV_BUILD_PATH = os.getenv("DEV_BUILD_PATH")
FUTURE_BUILD_PATH = os.getenv("FUTURE_BUILD_PATH")

REPORTS_DIR = "reports"
LATEST_DIR = os.path.join(REPORTS_DIR, "latest")
HISTORY_DIR = os.path.join(REPORTS_DIR, "history")

def init_directories():
    os.makedirs(LATEST_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

CPU_TOLERANCE_PERCENT = float(os.getenv("CPU_TOLERANCE_PERCENT", "0.1"))
RAM_TOLERANCE_MB = float(os.getenv("RAM_TOLERANCE_MB", "1.0"))
GPU_TOLERANCE_PERCENT = float(os.getenv("GPU_TOLERANCE_PERCENT", "0.1"))
