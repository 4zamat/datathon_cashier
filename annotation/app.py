"""Loopback-only annotation app. Dataset routes are never mounted in app.main."""

import secrets
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from annotation.models import AnnotationDocument, CaptureRequest, OBJECT_CLASSES, REVIEW_PROFILES
from annotation.workspace import EVENT_NAMES, ROOT, Workspace

STATIC = Path(__file__).parent / "static"


def create_app(data_dir: Path = ROOT / "data") -> FastAPI:
    workspace = Workspace(data_dir)
    token = secrets.token_urlsafe(32)
    app = FastAPI(title="Cashier · Локальная разметка", docs_url=None, redoc_url=None)
    app.state.workspace = workspace
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_access(request: Request, call_next):
        if request.method not in {"GET", "HEAD"}:
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Откройте локальную страницу разметки"}, status_code=403)
            if not secrets.compare_digest(request.headers.get("x-annotation-token", ""), token):
                return JSONResponse({"detail": "Обновите страницу, чтобы продолжить сохранение"}, status_code=403)
            if len(await request.body()) > 2 * 1024 * 1024:
                return JSONResponse({"detail": "Слишком большой документ разметки"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; object-src 'none'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    def config() -> dict:
        return {"token": token, "rules": workspace.rules, "event_names": EVENT_NAMES,
                "object_classes": OBJECT_CLASSES, "scan_errors": workspace.scan_errors,
                "review_profiles": REVIEW_PROFILES,
                "save_directory": str(workspace.output), "video_count": len(workspace.videos)}

    @app.get("/api/videos")
    def videos() -> list[dict]:
        result = []
        for video_id, info in workspace.videos.items():
            try:
                document = workspace.load(video_id)
                status = "reviewed" if document.watched_full else "draft" if document.revision else "new"
                result.append({**info, "status": status, "events": len(document.events), "frames": len(document.frames)})
            except HTTPException:
                result.append({**info, "status": "error", "events": 0, "frames": 0})
        return result

    @app.get("/api/videos/{video_id}/annotation", response_model=AnnotationDocument)
    def load(video_id: str) -> AnnotationDocument:
        return workspace.load(video_id)

    @app.put("/api/videos/{video_id}/annotation", response_model=AnnotationDocument)
    def save(video_id: str, document: AnnotationDocument) -> AnnotationDocument:
        return workspace.save(video_id, document)

    @app.post("/api/videos/{video_id}/capture")
    def capture(video_id: str, body: CaptureRequest) -> dict:
        if body.timestamp_sec > workspace.info(video_id)["duration_sec"]:
            raise HTTPException(422, "Время за пределами видео")
        return workspace.capture(video_id, body.timestamp_sec)

    @app.get("/media/{video_id}")
    def media(video_id: str) -> FileResponse:
        path, media_type = workspace.media(video_id)
        return FileResponse(path, media_type=media_type)

    @app.get("/frames/{video_id}/{frame_id}.jpg")
    def frame(video_id: str, frame_id: str) -> FileResponse:
        path = workspace.frame_path(video_id, frame_id, "jpg")
        if not path.is_file():
            raise HTTPException(404, "Кадр не найден")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/export")
    def export() -> FileResponse:
        path = workspace.export()
        return FileResponse(path, media_type="application/zip", filename=path.name)

    app.mount("/assets", StaticFiles(directory=STATIC), name="assets")
    return app
