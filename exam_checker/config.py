"""
Singleton configuration loaded from .env file.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Singleton configuration manager."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Load .env from project root
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        else:
            load_dotenv()

        # --- API Keys ---
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "")

        # --- Database ---
        db_default = str(Path(__file__).parent / "exam_checker.db")
        self.DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{db_default}")

        # --- Grade Boundaries ---
        default_boundaries = json.dumps({
            "A+": 90, "A": 80, "B+": 70, "B": 60,
            "C+": 50, "C": 40, "D": 33, "F": 0
        })
        self.GRADE_BOUNDARIES = json.loads(
            os.getenv("GRADE_BOUNDARIES", default_boundaries)
        )

        # --- Processing ---
        self.NEGATIVE_MARKING_FACTOR = float(
            os.getenv("NEGATIVE_MARKING_FACTOR", "0.0")
        )
        self.RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "50"))
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
        self.RETRY_BASE_DELAY = float(os.getenv("RETRY_BASE_DELAY", "2.0"))

        # --- OpenAI Model Settings ---
        self.OPENAI_EVAL_MODEL = os.getenv("OPENAI_EVAL_MODEL", "gpt-4o")
        self.OPENAI_EMBEDDING_MODEL = os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )

        # --- Paths ---
        self.TEMP_DIR = Path(os.getenv("TEMP_DIR", str(Path(__file__).parent / "temp")))
        self.LOG_DIR = Path(os.getenv("LOG_DIR", str(Path(__file__).parent / "logs")))
        self.UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(Path(__file__).parent / "uploads")))

        # Create directories
        self.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        # --- Portal ---
        self.PORTAL_HOST = os.getenv("PORTAL_HOST", "127.0.0.1")
        self.PORTAL_PORT = int(os.getenv("PORTAL_PORT", "8000"))

    def validate(self):
        """Validate that required configuration is present."""
        errors = []
        if not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is not set")
        if not self.GOOGLE_VISION_API_KEY:
            errors.append("GOOGLE_VISION_API_KEY is not set")
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    def print_summary(self):
        """Print configuration summary."""
        print("=" * 60)
        print("  Hybrid AI Exam Checker — Configuration")
        print("=" * 60)
        print(f"  OpenAI API Key:       {'***' + self.OPENAI_API_KEY[-4:] if self.OPENAI_API_KEY else 'NOT SET'}")
        print(f"  Vision API Key:      {'***' + self.GOOGLE_VISION_API_KEY[-4:] if self.GOOGLE_VISION_API_KEY else 'NOT SET'}")
        print(f"  Eval Model:          {self.OPENAI_EVAL_MODEL}")
        print(f"  Embedding Model:     {self.OPENAI_EMBEDDING_MODEL}")
        print(f"  Database:            {self.DATABASE_URL}")
        print(f"  Rate Limit:          {self.RATE_LIMIT_RPM} req/min")
        print(f"  Grade Boundaries:    {self.GRADE_BOUNDARIES}")
        print(f"  Negative Marking:    {self.NEGATIVE_MARKING_FACTOR}")
        print("=" * 60)


# Module-level singleton
config = Config()
