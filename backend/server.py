"""
server.py - FastAPI application for real-time speech-to-text.

Provides:
    GET  /           - Serves the frontend index.html
    GET  /config     - Returns model registry, languages, and device info
    GET  /status     - Returns model loading status
    WS   /ws         - WebSocket endpoint for streaming audio to text
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import Config
from backend.model_manager import ModelManager
from backend.pipeline.pipeline_manager import PipelineManager
from backend.realtime_diarization import RealtimeDiarizationSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline models at startup. STT and diarization models stay lazy."""
    logger.info("=" * 60)
    logger.info("  Starting Real-Time Speech-to-Text Server")
    logger.info("  Device: %s", Config.DEVICE)
    logger.info("  STT models: %s available (lazy loading)", len(Config.STT_MODELS))
    logger.info("=" * 60)

    model_manager.load_all()

    logger.info("=" * 60)
    logger.info("  Server ready! Open http://localhost:8000")
    logger.info("=" * 60)
    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="Real-Time Speech-to-Text",
    description="Live speech transcription with modular pipeline",
    version="2.1.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.middleware("http")
async def disable_frontend_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/")
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/config")
async def get_config():
    return JSONResponse(content=Config.summary())


@app.get("/status")
async def get_status():
    return JSONResponse(content=model_manager.status)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for real-time audio streaming."""
    await ws.accept()
    logger.info("WebSocket client connected.")

    pipeline = PipelineManager(model_manager)
    diarization_session = None
    diarization_task = None
    diarization_status_sent = False
    active_diarization_key = "none"

    async def schedule_diarization_updates():
        nonlocal diarization_task, diarization_status_sent, diarization_session
        if diarization_session is None or not diarization_session.has_pending_work():
            diarization_task = None
            return

        try:
            if not diarization_status_sent:
                diarization_status_sent = True
                await ws.send_json({
                    "type": "status",
                    "message": "Loading speaker diarization model (first time may download)...",
                })

            diarization_model = await asyncio.to_thread(
                model_manager.get_diarization_model,
                active_diarization_key,
            )
            if diarization_model is None or not diarization_model.available:
                pipeline.disable_diarization()
                fallback_updates = diarization_session.disable()
                await ws.send_json({
                    "type": "status",
                    "message": "Speaker diarization unavailable. Continuing without labels.",
                })
                if fallback_updates:
                    await ws.send_json({
                        "type": "speaker_update",
                        "updates": fallback_updates,
                    })
                await ws.send_json({
                    "type": "config_update",
                    "status": pipeline.get_status(),
                })
                diarization_session = None
                diarization_task = None
                return

            updates = await asyncio.to_thread(diarization_session.run_pending, diarization_model)
            if updates:
                await ws.send_json({
                    "type": "speaker_update",
                    "updates": updates,
                })

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Speaker diarization background task failed: %s", exc)
        finally:
            diarization_task = None
            if diarization_session is not None and diarization_session.has_pending_work():
                diarization_task = asyncio.create_task(schedule_diarization_updates())

    try:
        config_msg = await ws.receive_text()
        config = json.loads(config_msg)
        pipeline.configure(config)
        logger.info("Pipeline configured: %s", config)

        active_diarization_key = pipeline.diarization_model
        if active_diarization_key != "none":
            diarization_session = RealtimeDiarizationSession(sample_rate=Config.SAMPLE_RATE)

        await ws.send_json({
            "type": "config_ack",
            "status": pipeline.get_status(),
        })

        first_chunk = True
        while True:
            message = await ws.receive()

            if "text" in message:
                if message["text"].upper() == "STOP":
                    logger.info("Client sent STOP, ending session.")
                    break
                continue

            if "bytes" not in message:
                continue

            audio_bytes = message["bytes"]
            if not audio_bytes:
                continue

            if first_chunk:
                first_chunk = False
                await ws.send_json({
                    "type": "status",
                    "message": "Loading selected STT model (first time may download)...",
                })

            result = await asyncio.to_thread(pipeline.process, audio_bytes)
            await ws.send_json({
                "type": "transcription",
                "diarization_model": pipeline.diarization_model,
                **result,
            })

            if diarization_session is None or pipeline.last_chunk_audio is None:
                continue

            diarization_session.append_audio_chunk(
                pipeline.last_chunk_audio,
                result.get("chunk_start_ms", 0),
                result.get("chunk_end_ms", 0),
            )
            diarization_session.register_transcript_items(result.get("items", []))

            if diarization_task is None:
                diarization_task = asyncio.create_task(schedule_diarization_updates())

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
    finally:
        if diarization_task is not None:
            diarization_task.cancel()
            with suppress(asyncio.CancelledError):
                await diarization_task
        pipeline.reset()
        logger.info("Pipeline reset, session ended.")
