import hmac
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from queue import Full
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.formparsers import MultiPartException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.config import PROJECT_ROOT, Settings
from app.models import AcceptedJob, Job
from app.rules import RuleCatalog
from app.storage import JobStore
from app.worker import JobWorker

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class APIAccessMiddleware:
    """Authenticate before parsing multipart; bound even chunked request bodies."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope["path"].startswith("/api/v1/"):
            await self.app(scope, receive, send)
            return
        settings: Settings = scope["app"].state.settings
        headers = Headers(scope=scope)
        key = headers.get("x-api-key", "")
        if not hmac.compare_digest(key.encode(), settings.api_key.get_secret_value().encode()):
            await JSONResponse(status_code=401, content={"detail": "Invalid or missing X-API-Key"})(scope, receive, send)
            return
        # Reserve 1 MiB for multipart headers/boundaries, with an exact file check below.
        limit = settings.max_upload_bytes + 1024 * 1024
        received = 0
        exceeded = False
        too_large = JSONResponse(status_code=413, content={"detail": "Upload exceeds the configured size limit"})
        length = headers.get("content-length")
        if length is not None:
            try:
                if int(length) > limit:
                    await too_large(scope, receive, send)
                    return
            except ValueError:
                await JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})(scope, receive, send)
                return

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    # Starlette closes all partially parsed upload files for this exception.
                    raise MultiPartException("Upload exceeds the configured size limit")
            return message

        async def limited_send(message: Message) -> None:
            if exceeded:
                if message["type"] == "http.response.start":
                    await too_large(scope, receive, send)
                return
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), (b"cache-control", b"no-store")]}
            await send(message)

        await self.app(scope, limited_receive, limited_send)


def persist_upload(upload: UploadFile, destination: Path, limit: int) -> None:
    """Copy the spooled request file with bounded memory, before the request closes."""
    size = 0
    with destination.open("xb") as output:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise HTTPException(status_code=413, detail="Video exceeds the configured size limit")
            output.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="Video is empty")


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        config = settings or Settings.from_env()
        rules = RuleCatalog(config.rules_path)
        store = JobStore(config.data_dir / "jobs")
        worker = JobWorker(config, store, rules)
        application.state.settings = config
        application.state.store = store
        application.state.worker = worker
        worker.start()
        try:
            yield
        finally:
            await run_in_threadpool(worker.stop)

    application = FastAPI(title="Cashier Violation Detector", version="0.1.0", lifespan=lifespan)
    application.add_middleware(APIAccessMiddleware)

    @application.get("/health")
    async def health(request: Request) -> dict:
        config: Settings = request.app.state.settings
        return {"status": "ok", "demo": config.pipeline_class == "pipeline.dummy:DummyPipeline",
                "pipeline": config.pipeline_class, "model_quality_verified": False,
                "processing_timeout_sec": config.processing_timeout_sec}

    @application.post("/api/v1/analyze", response_model=AcceptedJob, status_code=202,
                      dependencies=[Depends(api_key_header)])
    async def analyze(request: Request, file: Annotated[UploadFile, File(...)]) -> AcceptedJob:
        config: Settings = request.app.state.settings
        worker: JobWorker = request.app.state.worker
        if Path(file.filename or "").suffix.lower() != ".mov":
            await file.close()
            raise HTTPException(status_code=415, detail="Expected a .mov video in multipart field 'file'")
        job_id = uuid4().hex
        destination = worker.upload_dir / f"{job_id}.mov"
        submitted = False
        try:
            await run_in_threadpool(persist_upload, file, destination, config.max_upload_bytes)
            try:
                worker.submit(job_id, destination)
            except Full as exc:
                raise HTTPException(status_code=503, detail="Analysis queue is full; retry later") from exc
            submitted = True
            return AcceptedJob(job_id=job_id)
        finally:
            await file.close()
            if not submitted:
                destination.unlink(missing_ok=True)

    @application.get("/api/v1/jobs/{job_id}", response_model=Job, dependencies=[Depends(api_key_header)])
    async def get_job(request: Request, job_id: str) -> Job:
        store: JobStore = request.app.state.store
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @application.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(PROJECT_ROOT / "static" / "index.html")

    application.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
    return application


app = create_app()
