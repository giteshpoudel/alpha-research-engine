"""Dashboard configuration with dev/prod environment toggle.

APP_ENV=dev (default) uses the local containers from the shared .env.
APP_ENV=prod is a reserved slot for a future remote deployment — it reads
the same keys with a PROD_ prefix so a prod .env can diverge without code
changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    app_env: str
    clickhouse_host: str
    clickhouse_port: int
    clickhouse_user: str
    clickhouse_password: str
    clickhouse_db: str
    dashboard_port: int


def get_settings() -> Settings:
    load_dotenv()
    app_env = os.environ.get("APP_ENV", "dev")
    prefix = "PROD_" if app_env == "prod" else ""

    def env(name: str, default: str) -> str:
        return os.environ.get(prefix + name, os.environ.get(name, default))

    return Settings(
        app_env=app_env,
        clickhouse_host=env("CLICKHOUSE_HOST", "localhost"),
        clickhouse_port=int(env("CLICKHOUSE_PORT", "8123")),
        clickhouse_user=env("CLICKHOUSE_USER", "alpha"),
        clickhouse_password=env("CLICKHOUSE_PASSWORD", ""),
        clickhouse_db=env("CLICKHOUSE_DB", "alpha"),
        dashboard_port=int(os.environ.get("DASHBOARD_PORT", "8000")),
    )
