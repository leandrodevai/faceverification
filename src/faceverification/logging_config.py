import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from faceverification.config import settings

request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_context.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id

        for key, value in getattr(record, "extra_fields", {}).items():
            if value is not None:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestContextFilter())

    if settings.log_format == "text":
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] "
            "request_id=%(request_id)s %(message)s"
        )
    else:
        formatter = JsonFormatter()

    handler.setFormatter(formatter)
    return handler


def configure_logging() -> None:
    level = logging.DEBUG if settings.debug else logging.INFO
    handler = _build_handler()

    logging.basicConfig(level=level, handlers=[handler], force=True)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)

    if not settings.debug:
        logging.getLogger("PIL").setLevel(logging.WARNING)
