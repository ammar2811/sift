"""Structured logs and distributed traces.

The infrastructure has been provisioning Application Insights and injecting
``APPLICATIONINSIGHTS_CONNECTION_STRING`` into the API container since the first deploy,
and nothing has ever read it. App Insights received the platform's stdout capture and
no spans, no dependencies and no custom metrics - which is to say the observability was
a line item rather than a capability.

Two things live here:

**Structured logging.** One JSON object per line, because a log line is read by a query
far more often than by a person, and `logger.info("ready: embeddings=%s", name)` cannot
be filtered on `embeddings` without a regex. Every line carries the request id, so one
request's work can be pulled out of a replica serving many at once.

**Tracing.** Optional and best-effort. Without a connection string, or without the
`azure-monitor-opentelemetry` package installed, the API runs exactly as before rather
than failing to start: telemetry that can take down the service it observes is a bad
trade, and local development has no connection string at all.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

from packages.sift_core.config import Settings

logger = logging.getLogger("sift.telemetry")

# Set per request by the API middleware, read by the log formatter. A ContextVar rather
# than a thread local because the request path is async and a single thread interleaves
# many requests.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Attributes LogRecord always carries. Anything outside this set was passed by the call
# site via `extra=` and belongs in the emitted object.
_STANDARD_FIELDS = frozenset(
    [
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    ]
)


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if request_id := request_id_var.get():
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        for key, value in record.__dict__.items():
            if key not in _STANDARD_FIELDS and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def configure_logging(settings: Settings) -> None:
    """Install the JSON formatter on the root handler.

    Replaces the handler rather than adding one, so uvicorn's own configuration does not
    produce every line twice in two different formats.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level)

    # uvicorn installs its own handlers; let its records travel to the root handler so
    # access and error lines are JSON too.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True


def configure_tracing(app: Any = None) -> bool:
    """Send traces to Application Insights when it is configured. Returns whether it was.

    Reads ``APPLICATIONINSIGHTS_CONNECTION_STRING`` from the environment directly, since
    that is the name Azure injects and the name the SDK expects; it is deliberately not
    a ``SIFT_`` setting.
    """
    import os

    if not os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        logger.info("tracing disabled: no APPLICATIONINSIGHTS_CONNECTION_STRING")
        return False

    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(logger_name="sift")
        if app is not None:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app)
        logger.info("tracing enabled")
        return True
    except Exception:
        # Never fatal. A replica that cannot report on itself should still serve.
        logger.warning("tracing could not be configured; continuing without it", exc_info=True)
        return False
