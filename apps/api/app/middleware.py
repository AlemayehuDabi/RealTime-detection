import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.logging_utils import request_id_ctx, session_id_ctx


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
        session_id = request.headers.get("X-Session-Id", "-")
        token_req = request_id_ctx.set(request_id)
        token_sess = session_id_ctx.set(session_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-Id"] = request_id
            return response
        finally:
            request_id_ctx.reset(token_req)
            session_id_ctx.reset(token_sess)