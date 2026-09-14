"""Boot the dashboard: python -m src.dashboard"""

import uvicorn

from src.dashboard.app import create_app
from src.dashboard.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(create_app(), host="127.0.0.1", port=settings.dashboard_port,
                log_level="info")


if __name__ == "__main__":
    main()
