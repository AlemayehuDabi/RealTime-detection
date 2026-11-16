import logging
import time
import uuid
from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")
session_id_ctx: ContextVar[str] = ContextVar("session_id", default="-")


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        record.session_id = session_id_ctx.get()
        return True


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s request_id=%(request_id)s session_id=%(session_id)s %(name)s: %(message)s",
    )
    for handler in logging.root.handlers:
        handler.addFilter(RequestIdFilter())