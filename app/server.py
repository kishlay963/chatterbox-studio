"""Chatterbox Studio API — voice-clone narration + word-level transcription.

Interactive docs at /docs. If API_KEY is set, send it as the `X-API-Key` header.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import engine as E

API_KEY = os.environ.get("API_KEY", "")
JOBS_DIR = Path(os.environ.get("JOBS_DIR", "/data/jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Chatterbox Studio", version="1.1.0",
              description="Voice-clone narration (Chatterbox, best-of-N quality control) and word-level transcription (faster-whisper large-v3).")


def auth(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "missing or wrong X-API-Key")


# ----------------------------------------------------------------------------------- models
class Voice(BaseModel):
    voice: str | None = Field(None, description="Voice name (GET /voices). Default: the server's default voice.")
    exaggeration: float = Field(0.65, ge=0.25, le=2.0, description="Emotion / expressiveness. 0.5 neutral, 0.65 lively, >0.9 dramatic.")
    cfg_weight: float = Field(0.4, ge=0.0, le=1.0, description="Pacing / adherence. Lower = slower, more deliberate delivery.")
    temperature: float = Field(0.8, ge=0.05, le=2.0)
    takes: int = Field(3, ge=1, le=8, description="Best-of-N per sentence, scored by Whisper accuracy + voice similarity. 1 = fastest.")
    min_accuracy: float = Field(0.9, ge=0.0, le=1.0)
    sentence_pause: float = Field(0.32, ge=0.0, le=3.0, description="Silence between sentences (s).")
    seed: int | None = None

    def params(self) -> E.TTSParams:
        return E.TTSParams(exaggeration=self.exaggeration, cfg_weight=self.cfg_weight, temperature=self.temperature, takes=self.takes,
                           min_accuracy=self.min_accuracy, sentence_pause=self.sentence_pause, question_pause=self.sentence_pause + 0.1,
                           seed=self.seed)


class TTSRequest(Voice):
    text: str = Field(..., min_length=1, max_length=20000)
    format: Literal["wav", "mp3", "flac", "ogg", "json"] = Field("wav", description="'json' returns base64 audio + word timings + QC report.")


class Section(BaseModel):
    id: str | None = Field(None, description="Your id for the section (e.g. 'c2/b03'). Defaults to s01, s02, …")
    title: str | None = None
    text: str = Field(..., min_length=1, max_length=20000)
    pause_after: float = Field(0.6, ge=0.0, le=10.0, description="Silence after this section in the combined file (s).")
    voice: str | None = Field(None, description="Override the voice for this section only.")
    exaggeration: float | None = Field(None, ge=0.25, le=2.0, description="Override expressiveness for this section only.")


class NarrateRequest(Voice):
    sections: list[Section] = Field(..., min_length=1, max_length=500)
    format: Literal["wav", "mp3", "flac"] = "wav"
    wait: bool = Field(False, description="Block until done and return the result (fine for short scripts). Otherwise poll /jobs/{id}.")


# ----------------------------------------------------------------------------------- jobs
JOBS: dict[str, dict] = {}
Q: "queue.Queue[tuple[str, NarrateRequest]]" = queue.Queue()


def _save_job(j: dict):
    (JOBS_DIR / j["id"] / "job.json").write_text(json.dumps({k: v for k, v in j.items() if k != "event"}, indent=1))


def _srt(words: list[dict], path: Path):
    def ts(s):
        h, r = divmod(max(s, 0), 3600)
        m, sec = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(sec):02d},{min(int(round((sec % 1) * 1000)), 999):03d}"
    cues, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if len(cur) >= 9 or (w["word"][-1:] in ".!?" and len(cur) >= 3) or nxt is None or nxt["start"] - w["end"] > 0.35:
            cues.append((cur[0]["start"], cur[-1]["end"] + 0.15, " ".join(x["word"] for x in cur)))
            cur = []
    path.write_text("\n".join(f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n" for i, (a, b, t) in enumerate(cues, 1)), encoding="utf-8")


def _run(jid: str, req: NarrateRequest):
    j = JOBS[jid]
    d = JOBS_DIR / jid
    try:
        j.update(status="running", started=time.time())
        total = sum(len(E.sentences(s.text)) for s in req.sections)
        done = [0]
        combined, all_words, results, sr, t = [], [], [], 24000, 0.0
        for i, s in enumerate(req.sections):
            sid = s.id or f"s{i + 1:02d}"
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in sid)
            p = req.params()
            if s.exaggeration is not None:
                p.exaggeration = s.exaggeration

            def prog(k, n, base=done[0]):
                j["progress"] = round((base + k) / max(total, 1), 3)
            r = E.synthesize(s.text, s.voice or req.voice, p, progress=prog)
            done[0] += len(E.sentences(s.text))
            sr = r["sr"]
            fn = f"{i + 1:03d}_{safe}.{req.format}"
            (d / fn).write_bytes(E.encode(r["wav"], sr, req.format))
            dur = len(r["wav"]) / sr
            results.append({"id": sid, "title": s.title, "file": f"/jobs/{jid}/files/{fn}", "duration": round(dur, 3), "voice": r["voice"],
                            "coverage": r["coverage"], "words": r["words"], "qc": r["picks"]})
            all_words += [{**w, "start": round(w["start"] + t, 3), "end": round(w["end"] + t, 3), "section": sid} for w in r["words"]]
            combined.append(r["wav"])
            pad = np.zeros(int(s.pause_after * sr), dtype=np.float32)
            combined.append(pad)
            t += dur + s.pause_after
            j["sections_done"] = i + 1
        full = np.concatenate(combined[:-1] if len(combined) > 1 else combined)
        (d / f"combined.{req.format}").write_bytes(E.encode(full, sr, req.format))
        _srt(all_words, d / "combined.srt")
        result = {"sections": results, "combined": {"file": f"/jobs/{jid}/files/combined.{req.format}", "duration": round(len(full) / sr, 3),
                                                    "srt": f"/jobs/{jid}/files/combined.srt", "words": all_words}}
        (d / "timings.json").write_text(json.dumps(result, indent=1))
        weak = [q["sentence"] for s in results for q in s["qc"] if not q["ok"]]
        j.update(status="done", progress=1.0, finished=time.time(), result_url=f"/jobs/{jid}/files/timings.json", weak_sentences=weak)
    except Exception as e:  # noqa: BLE001
        j.update(status="error", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
    finally:
        _save_job(j)
        j["event"].set()


def _worker():
    while True:
        jid, req = Q.get()
        _run(jid, req)


threading.Thread(target=_worker, daemon=True).start()


@app.on_event("startup")
def _startup():
    if os.environ.get("WARMUP", "1") == "1":
        threading.Thread(target=E.warmup, daemon=True).start()


# ----------------------------------------------------------------------------------- routes
@app.get("/health")
def health():
    import torch
    return {"ok": True, "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "tts_loaded": E._tts is not None, "whisper_loaded": list(E._whisper), "default_voice": E.default_voice(),
            "queue": Q.qsize()}


@app.get("/voices", dependencies=[Depends(auth)])
def voices():
    return {"default": E.default_voice(), "voices": E.list_voices()}


@app.post("/voices", dependencies=[Depends(auth)])
async def add_voice(name: str = Form(..., description="letters, digits, _ and -"),
                    file: UploadFile = File(..., description="Voice sample (any audio/video; 10-20 s of clean speech is ideal, longer is auto-trimmed) "
                                                              "OR a Chatterbox conditionals .pt file"),
                    start: float | None = Form(None, description="Optional: use the sample from this second…"),
                    end: float | None = Form(None, description="…to this second"),
                    auto_trim: bool = Form(True, description="Pick the cleanest ~14 s window automatically"),
                    make_default: bool = Form(False),
                    description: str = Form("")):
    try:
        v = E.add_voice(name, await file.read(), file.filename, start, end, auto_trim, description)
        if make_default:
            E.set_default(name)
        return v
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/voices/{name}/audio", dependencies=[Depends(auth)])
def voice_audio(name: str):
    """The reference clip used for cloning (audio voices only)."""
    try:
        v = E.get_voice(name)
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e))
    if v.kind != "audio":
        raise HTTPException(404, "this voice has no reference audio (it was created from a .pt file or is built in)")
    return FileResponse(v.path, media_type="audio/wav")


@app.get("/voices/{name}/conds", dependencies=[Depends(auth)])
def voice_conds(name: str):
    """Download the voice as a Chatterbox conditionals .pt: keep it, and re-upload it to any fresh container with POST /voices."""
    try:
        data = E.export_conds(name)
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e))
    return Response(data, media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{name}.pt"'})


@app.put("/voices/default", dependencies=[Depends(auth)])
def set_default(body: dict):
    try:
        E.set_default(body["name"])
        return {"default": body["name"]}
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e))


@app.delete("/voices/{name}", dependencies=[Depends(auth)])
def delete_voice(name: str):
    try:
        E.delete_voice(name)
        return {"deleted": name}
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/tts", dependencies=[Depends(auth)])
def tts(req: TTSRequest):
    """One piece of text -> audio. For long scripts with sections use /narrate."""
    try:
        r = E.synthesize(req.text, req.voice, req.params())
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    hdr = {"X-Coverage": str(r["coverage"]), "X-Voice": r["voice"], "X-Duration": f"{len(r['wav']) / r['sr']:.3f}"}
    if req.format == "json":
        return {"voice": r["voice"], "sample_rate": r["sr"], "duration": round(len(r["wav"]) / r["sr"], 3), "coverage": r["coverage"],
                "words": r["words"], "qc": r["picks"], "audio_wav_base64": base64.b64encode(E.encode(r["wav"], r["sr"], "wav")).decode()}
    mt = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac", "ogg": "audio/ogg"}[req.format]
    return Response(E.encode(r["wav"], r["sr"], req.format), media_type=mt, headers=hdr)


@app.post("/tts/clone", dependencies=[Depends(auth)])
def tts_clone(text: str = Form(..., max_length=20000),
              file: UploadFile = File(..., description="Voice sample (audio/video) or Chatterbox .pt for this request"),
              save_as: str | None = Form(None, description="Also keep this voice under this name"),
              exaggeration: float = Form(0.65), cfg_weight: float = Form(0.4), temperature: float = Form(0.8),
              takes: int = Form(3), format: str = Form("wav")):
    """One-shot: clone from the uploaded sample/.pt and speak `text`, in one call. The same upload is cached, so
    repeating it doesn't re-clone."""
    data = file.file.read()
    name = save_as or "tmp-" + hashlib.sha1(data).hexdigest()[:10]
    try:
        if save_as or not E._find(name)[0]:
            E.add_voice(name, data, file.filename, description="" if save_as else "temporary (from /tts/clone)")
        req = TTSRequest(text=text, voice=name, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature,
                         takes=takes, format=format)
    except ValueError as e:   # pydantic ValidationError is a ValueError too
        raise HTTPException(422, str(e))
    return tts(req)


@app.post("/narrate", dependencies=[Depends(auth)])
def narrate(req: NarrateRequest):
    """A full narration split into sections. Each section becomes its own file with word timings, plus a
    combined file and an .srt. Runs as a background job unless wait=true."""
    for s in req.sections:
        try:
            E.get_voice(s.voice or req.voice)
        except (KeyError, ValueError) as e:
            raise HTTPException(404, str(e))
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    (JOBS_DIR / jid).mkdir(parents=True)
    JOBS[jid] = {"id": jid, "status": "queued", "progress": 0.0, "sections": len(req.sections), "sections_done": 0,
                 "created": time.time(), "event": threading.Event()}
    (JOBS_DIR / jid / "request.json").write_text(req.model_dump_json(indent=1))
    Q.put((jid, req))
    if req.wait:
        JOBS[jid]["event"].wait()
        return job_result(jid)
    return JSONResponse({"job_id": jid, "status_url": f"/jobs/{jid}", "position": Q.qsize()}, status_code=202)


@app.get("/jobs", dependencies=[Depends(auth)])
def jobs():
    return [{k: v for k, v in j.items() if k not in ("event", "trace")} for j in JOBS.values()]


@app.get("/jobs/{jid}", dependencies=[Depends(auth)])
def job(jid: str):
    j = JOBS.get(jid)
    if not j:
        f = JOBS_DIR / jid / "job.json"
        if not f.exists():
            raise HTTPException(404, "no such job")
        return json.loads(f.read_text())
    return {k: v for k, v in j.items() if k != "event"}


@app.get("/jobs/{jid}/result", dependencies=[Depends(auth)])
def job_result(jid: str):
    f = JOBS_DIR / jid / "timings.json"
    if not f.exists():
        return job(jid)
    return json.loads(f.read_text())


@app.get("/jobs/{jid}/files/{name}", dependencies=[Depends(auth)])
def job_file(jid: str, name: str):
    p = (JOBS_DIR / jid / name).resolve()
    if JOBS_DIR.resolve() not in p.parents or not p.exists():
        raise HTTPException(404, "no such file")
    return FileResponse(p)


@app.post("/transcribe", dependencies=[Depends(auth)])
async def transcribe(file: UploadFile = File(..., description="Any audio or video file"),
                     model: str = Form(E.TRANSCRIBE_MODEL, description="large-v3 (best), large-v3-turbo, medium.en, small.en …"),
                     language: str | None = Form(None, description="e.g. 'en'. Empty = auto-detect"),
                     script: str | None = Form(None, description="Optional known script: returns `aligned` = your script's words with real timings"),
                     vad: bool = Form(True, description="Skip silence / music (Silero VAD)"),
                     initial_prompt: str | None = Form(None, description="Vocabulary hints, e.g. product names"),
                     beam_size: int = Form(5)):
    """High-quality word-level transcription -> JSON {text, segments[words], words[], aligned?}."""
    data = await file.read()
    try:
        return E.transcribe(data, model, language or None, script, vad, initial_prompt, beam_size)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"{type(e).__name__}: {e}")
