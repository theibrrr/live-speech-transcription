"""
server.py — FastAPI application for real-time speech-to-text.

Provides:
    GET  /           — Serves the frontend index.html
    GET  /config     — Returns model registry, languages, and device info
    GET  /status     — Returns model loading status
    WS   /ws         — WebSocket endpoint for streaming audio → text

Pipeline models (VAD, DeepFilterNet) load at startup.
STT models load lazily on first use via ModelManager.
"""

import json
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.config import Config
from backend.model_manager import ModelManager
from backend.pipeline.pipeline_manager import PipelineManager

# ── Logging Setup ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# ── Global Model Manager ───────────────────────────────────────────────
model_manager = ModelManager()


# ── Lifespan ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline models at startup. STT models load lazily."""
    logger.info("=" * 60)
    logger.info("  Starting Real-Time Speech-to-Text Server (v2)")
    logger.info(f"  Device: {Config.DEVICE}")
    logger.info(f"  STT models: {len(Config.STT_MODELS)} available (lazy loading)")
    logger.info("=" * 60)

    model_manager.load_all()

    logger.info("=" * 60)
    logger.info("  Server ready! Open http://localhost:8000")
    logger.info("=" * 60)

    yield

    logger.info("Server shutting down.")


# ── FastAPI App ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Real-Time Speech-to-Text",
    description="Live speech transcription with modular pipeline (v2)",
    version="2.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── REST Endpoints ──────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/config")
async def get_config():
    """Return full configuration: model registry, languages, device info."""
    return JSONResponse(content=Config.summary())


@app.get("/status")
async def get_status():
    return JSONResponse(content=model_manager.status)


# ── WebSocket Endpoint ─────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.

    Protocol:
        1. Client sends JSON config:
           {"ns_enabled": false, "vad_model": "silero",
            "stt_model": "fasterwhisper-base", "language": "en"}
        2. Client sends binary audio chunks (float32 PCM, 16kHz mono).
        3. Server responds with JSON per chunk.
        4. Client sends "STOP" text to end.
    """
    await ws.accept()
    logger.info("WebSocket client connected.")

    pipeline = PipelineManager(model_manager)

    try:
        # Step 1: Receive configuration
        config_msg = await ws.receive_text()
        config = json.loads(config_msg)
        pipeline.configure(config)
        logger.info(f"Pipeline configured: {config}")

        await ws.send_json({
            "type": "config_ack",
            "status": pipeline.get_status(),
        })

        # Step 2: Process audio stream
        first_chunk = True
        while True:
            message = await ws.receive()

            if "text" in message:
                if message["text"].upper() == "STOP":
                    logger.info("Client sent STOP, ending session.")
                    break
                continue

            if "bytes" in message:
                audio_bytes = message["bytes"]
                if audio_bytes and len(audio_bytes) > 0:
                    # First chunk may trigger model download — notify client
                    if first_chunk:
                        first_chunk = False
                        await ws.send_json({
                            "type": "status",
                            "message": "Loading STT model (first time may download)...",
                        })

                    # Run sync ML inference in a thread to not block the event loop
                    result = await asyncio.to_thread(pipeline.process, audio_bytes)
                    await ws.send_json({"type": "transcription", **result})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        pipeline.reset()
        logger.info("Pipeline reset, session ended.")
