"""The one place for settings and tunable constants.

`settings` holds everything that varies by environment and is read from the process
environment and `.env` (see `.env.example`). The module-level constants below are domain
tunables that are fixed for a deployment but should never be scattered inline.
"""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- service ---
    log_level: str = "INFO"
    db_path: Path = ROOT / "data" / "warehouse.duckdb"
    dq_report_path: Path = ROOT / "reports" / "dq_report.json"
    model_path: Path = ROOT / "models" / "delay_model.joblib"

    # --- LLM / assistant ---
    gemini_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    assistant_model: str = "gemini-3.5-flash-lite"
    assistant_api_url: str = "http://localhost:8000"
    assistant_root_path: str = ""  # "/assistant" when served behind the gateway prefix
    assistant_session_db: Path = ROOT / "data" / "assistant_sessions.db"
    assistant_log_path: Path = ROOT / "logs" / "assistant.jsonl"
    assistant_max_llm_calls: int = 8  # per user turn; the loop stops at this cap
    assistant_max_tool_calls: int = 6  # per user turn; a guardrail against runaway tool use


settings = Settings()

# --- data semantics (shared by DQ checks, pipeline and API) ---
ON_TIME_THRESHOLD_HOURS = 24  # on_time_flag = actual_delay_hours <= this
CONTAINER_RANGE = (1, 100)  # observed bulk is 1–60; 510–996 are entry errors
API_PAGE_SIZE_DEFAULT = 50
API_PAGE_SIZE_MAX = 500

# --- ML ---
LABEL_THRESHOLD_HOURS = ON_TIME_THRESHOLD_HOURS  # label = late by more than this
TRAIN_END = "2025-04-01"  # bookings before this train the model
VALIDATION_END = "2025-06-01"  # [TRAIN_END, VALIDATION_END) tunes it; the rest is the test set
RISK_LABEL_THRESHOLDS = (0.3, 0.6)  # low < 0.3 <= medium < 0.6 <= high
RANDOM_SEED = 42

# --- assistant ---
ASSISTANT_APP_NAME = "supply-chain-assistant"
ASSISTANT_TOOL_ROW_LIMIT = 20  # max rows a tool hands back to the model
GEMINI_PRICES_PER_MTOK = {  # USD per 1M tokens (input, output), from ai.google.dev pricing
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),
}
