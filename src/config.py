import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# AWS
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "aml-detection-mvp")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001"
)

# Dataset
DATASET_NAME = "HI-Small_Trans.csv"
SAMPLE_SIZE = 500_000
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "Is Laundering"

# Ensure dirs exist
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
