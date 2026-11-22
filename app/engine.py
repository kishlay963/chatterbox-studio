"""Chatterbox voice-clone TTS + faster-whisper transcription, with the QC loop used for the course videos.

Narration quality loop (per sentence): generate N takes -> transcribe each -> score on word accuracy vs
the script and speaker similarity to the voice -> reject takes with missing words / odd pacing / long
hallucinated tails -> keep the best, trim to its spoken words, join sentences with natural pauses.
"""
from __future__ import annotations

import copy
import difflib
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

VOICES_DIR = Path(os.environ.get("VOICES_DIR", "/data/voices"))
BUILTIN_VOICES = Path(os.environ.get("BUILTIN_VOICES", "/app/voices_builtin"))
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "")
QC_MODEL = os.environ.get("QC_WHISPER_MODEL", "medium.en")
TRANSCRIBE_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
DEVICE = os.environ.get("DEVICE", "cuda")

_lock = threading.RLock()          # one GPU, one job at a time
_tts = None
_whisper: dict[str, object] = {}
_conds_key: tuple | None = None
_centroids: dict[str, np.ndarray] = {}


# ----------------------------------------------------------------------------------- text helpers
NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
       "8": "eight", "9": "nine", "10": "ten"}
NUMW = set("""zero oh one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen
sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million
and dollars dollar point percent""".split())


def norm(tok: str) -> str:
    t = tok.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9'\-]", "", t).strip("'-")
    return NUM.get(t, t)


def tokens(text: str) -> list[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]


def _plain(w: str) -> bool:
    """Words used for accuracy scoring: Whisper writes numbers as digits, so number words are ignored,
    and hyphenated words are split ("double-click" vs "double click")."""
    return bool(w) and not re.search(r"\d", w) and not all(p in NUMW for p in w.split("-"))


def _scoring_words(text_words: list[str]) -> list[str]:
    out = []
    for t in text_words:
        for p in norm(t).split("-"):
            if _plain(p):
                out.append(p)
    return out


def accuracy(ref: str, hyp_words: list[str]) -> float:
    a = _scoring_words(tokens(ref))
    b = _scoring_words(hyp_words)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    m = sum(x.size for x in sm.get_matching_blocks())
    return 2 * m / max(len(a) + len(b), 1)


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    out: list[str] = []
    for p in parts:
        if out and len(p) < 26 and not out[-1].endswith("?"):
            out[-1] += " " + p
        elif out and len(out[-1]) < 20:
            out[-1] += " " + p
        else:
            out.append(p)
    # very long sentences: split at commas so a single generation stays well under the model's limit
    final = []
    for s in out:
        while len(s) > 280:
            cut = s.rfind(", ", 0, 260)
            if cut < 80:
                break
            final.append(s[: cut + 1])
            s = s[cut + 2:]
        final.append(s)
    return [s for s in final if s.strip()]


def align(script: str, asr_words: list[dict]) -> list[dict]:
    """Map every SCRIPT word to real times from the ASR words (captions keep the exact script text)."""
    words = tokens(script)
    if not asr_words or not words:
        return []
    a = [norm(w) for w in words]
    b = [norm(w["word"]) for w in asr_words]
    times: list[tuple[float, float] | None] = [None] * len(words)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal" or (tag == "replace" and (i2 - i1) == (j2 - j1)):
            for k in range(i2 - i1):
                times[i1 + k] = (asr_words[j1 + k]["start"], asr_words[j1 + k]["end"])
        elif tag == "replace" and j2 > j1:
            s, e = asr_words[j1]["start"], asr_words[j2 - 1]["end"]
            n = i2 - i1
            for k in range(n):
                times[i1 + k] = (s + (e - s) * k / n, s + (e - s) * (k + 1) / n)
    known = [i for i, t in enumerate(times) if t]
    for i, t in enumerate(times):
        if t:
            continue
        prev = max((k for k in known if k < i), default=None)
        nxt = min((k for k in known if k > i), default=None)
        s = times[prev][1] if prev is not None else 0.0
        e = times[nxt][0] if nxt is not None else s + 0.3
        times[i] = (s, max(e, s + 0.05))
    return [{"word": w, "start": round(t[0], 3), "end": round(t[1], 3)} for w, t in zip(words, times)]


# ----------------------------------------------------------------------------------- audio helpers
def load_audio(data: bytes | str | Path, sr: int) -> np.ndarray:
    """Any format -> float32 mono at sr (ffmpeg)."""
    src = ["-i", str(data)] if not isinstance(data, (bytes, bytearray)) else ["-i", "pipe:0"]
    r = subprocess.run(["ffmpeg", "-v", "error", *src, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       input=data if isinstance(data, (bytes, bytearray)) else None, capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def encode(wav: np.ndarray, sr: int, fmt: str = "wav") -> bytes:
    buf = io.BytesIO()
    if fmt == "wav":
        sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    codec = {"mp3": ["-c:a", "libmp3lame", "-q:a", "2", "-f", "mp3"], "flac": ["-f", "flac"],
             "ogg": ["-c:a", "libopus", "-b:a", "96k", "-f", "ogg"]}[fmt]
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", *codec, "-"], input=buf.getvalue(), capture_output=True, check=True)
    return r.stdout


def best_window(w: np.ndarray, sr: int, seconds: float = 15.0) -> tuple[int, int]:
    """Pick the most continuously voiced window (for auto-trimming a long clone sample)."""
    if len(w) <= seconds * sr:
        return 0, len(w)
    hop = int(0.05 * sr)
    frames = len(w) // hop
    e = np.sqrt(np.mean(w[: frames * hop].reshape(frames, hop) ** 2, axis=1))
    thr = max(float(np.percentile(e, 30)), 1e-3)
    voiced = (e > thr).astype(float)
    n = int(seconds / 0.05)
    i = int(np.argmax(np.convolve(voiced, np.ones(n), mode="valid")))
    # snap start/end to the quietest frame nearby so we never cut mid-word
    s0 = max(0, i - 8)
    s = s0 + int(np.argmin(e[s0: i + 1]))
    e1 = min(frames, s + n)
    e0 = max(s + 1, e1 - 8)
    end = e0 + int(np.argmin(e[e0:e1])) if e1 > e0 else e1
    return s * hop, min(len(w), end * hop)


# ----------------------------------------------------------------------------------- models
def tts():
    global _tts, _builtin_conds
    if _tts is None:
        import torch  # noqa: F401  (loads CUDA libs that ctranslate2 also needs)
        from chatterbox.tts import ChatterboxTTS
        _tts = ChatterboxTTS.from_pretrained(device=DEVICE)
        if getattr(_tts, "conds", None) is not None:
            _builtin_conds = copy.deepcopy(_tts.conds)
    return _tts


def whisper(name: str):
    if name not in _whisper:
        import torch  # noqa: F401
        from faster_whisper import WhisperModel
        _whisper[name] = WhisperModel(name, device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")
    return _whisper[name]


def emb(w: np.ndarray, sr: int) -> np.ndarray:
    import librosa
    m = tts()
    e = m.ve.embeds_from_wavs([librosa.resample(w.astype("float32"), orig_sr=sr, target_sr=16000)], sample_rate=16000)[0]
    return e / np.linalg.norm(e)


# ----------------------------------------------------------------------------------- voices
BUILTIN = "chatterbox"            # Chatterbox's stock voice, always available
NAME_RE = r"[A-Za-z0-9_\-]{1,64}"
_builtin_conds = None


@dataclass
class Voice:
    name: str
    path: Path | None     # .wav reference clip, .pt conditionals, or None for the built-in voice
    kind: str             # "audio" | "conds" | "builtin"
    meta: dict


def ensure_voices():
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    if BUILTIN_VOICES.exists():
        for f in BUILTIN_VOICES.iterdir():
            dst = VOICES_DIR / f.name
            if f.is_file() and not f.name.startswith(".") and not dst.exists():
                dst.write_bytes(f.read_bytes())


def _find(name: str) -> tuple[Path | None, str | None]:
    for ext, kind in ((".wav", "audio"), (".pt", "conds")):
        p = VOICES_DIR / f"{name}{ext}"
        if p.exists():
            return p, kind
    return None, None


def _meta(name: str) -> dict:
    f = VOICES_DIR / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def list_voices() -> list[dict]:
    ensure_voices()
    d = default_voice()
    out = [{"name": BUILTIN, "kind": "builtin", "default": d == BUILTIN, "description": "Chatterbox stock voice"}]
    for p in sorted([*VOICES_DIR.glob("*.wav"), *VOICES_DIR.glob("*.pt")]):
        if p.stem.endswith("_reference"):
            continue
        out.append({"name": p.stem, "kind": "audio" if p.suffix == ".wav" else "conds", "default": p.stem == d, **_meta(p.stem)})
    return out


def default_voice() -> str:
    f = VOICES_DIR / "_default"
    for cand in ((f.read_text().strip() if f.exists() else ""), DEFAULT_VOICE):
        if cand and (cand == BUILTIN or _find(cand)[0]):
            return cand
    return BUILTIN


def set_default(name: str):
    get_voice(name)
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    (VOICES_DIR / "_default").write_text(name)


def get_voice(name: str | None) -> Voice:
    ensure_voices()
    name = name or default_voice()
    if name == BUILTIN:
        return Voice(BUILTIN, None, "builtin", {})
    if not re.fullmatch(NAME_RE, name):
        raise ValueError("voice names: letters, digits, _ and - only")
    p, kind = _find(name)
    if not p:
        raise KeyError(f"voice '{name}' not found")
    return Voice(name, p, kind, _meta(name))


def load_conds(path: str):
    """Load Chatterbox conditionals safely: weights_only=True means tensors only, no pickled code."""
    import torch
    from chatterbox.tts import Conditionals
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    kw = torch.load(path, map_location="cpu", weights_only=True)
    return Conditionals(T3Cond(**kw["t3"]), kw["gen"])


def _save_conds(c, path: Path):
    import torch
    torch.save(dict(t3=dict(c.t3.__dict__), gen=dict(c.gen)), str(path))


def _is_pt(data: bytes, filename: str | None) -> bool:
    return (filename or "").lower().endswith(".pt") or data[:2] == b"PK"   # torch.save files are zip archives


def add_voice(name: str, data: bytes, filename: str | None = None, start: float | None = None, end: float | None = None,
              auto_trim: bool = True, description: str = "") -> dict:
    """Create a voice from (a) any audio/video sample: keeps a clean ~14 s clip for conditioning plus up to 3 min
    for the QC similarity check; or (b) a Chatterbox conditionals .pt file (from GET /voices/{name}/conds or
    `model.conds.save()`)."""
    global _conds_key
    if not re.fullmatch(NAME_RE, name) or name == BUILTIN:
        raise ValueError("voice names: letters, digits, _ and - only (and not 'chatterbox')")
    ensure_voices()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    is_pt = _is_pt(data, filename)
    if is_pt:
        tmp = tmpfile(".pt")
        try:
            tmp.write_bytes(data)
            c = load_conds(str(tmp))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"not a valid Chatterbox conditionals .pt file ({type(e).__name__}: {e})")
        finally:
            tmp.unlink(missing_ok=True)
        for old in (f"{name}.wav", f"{name}_reference.wav"):
            (VOICES_DIR / old).unlink(missing_ok=True)
        _save_conds(c, VOICES_DIR / f"{name}.pt")
        meta = {"description": description, "created": now, "source": "conds .pt"}
    else:
        sr = 24000
        try:
            full = load_audio(data, sr)
        except subprocess.CalledProcessError:
            raise ValueError("couldn't decode that file as audio")
        if len(full) < sr * 5:
            raise ValueError("need at least 5 seconds of speech (10-20 s of clean speech is ideal)")
        if start is not None or end is not None:
            ref = full[int((start or 0) * sr): int(end * sr) if end else len(full)]
        elif auto_trim:
            a, b = best_window(full, sr)
            ref = full[a:b]
        else:
            ref = full
        ref = ref * (0.9 / (float(np.max(np.abs(ref))) or 1.0))
        (VOICES_DIR / f"{name}.pt").unlink(missing_ok=True)
        sf.write(VOICES_DIR / f"{name}.wav", ref, sr, subtype="PCM_16")
        sf.write(VOICES_DIR / f"{name}_reference.wav", full[: sr * 180], sr, subtype="PCM_16")
        meta = {"description": description, "created": now, "source_seconds": round(len(full) / sr, 2),
                "reference_seconds": round(len(ref) / sr, 2)}
    (VOICES_DIR / f"{name}.json").write_text(json.dumps(meta, indent=1))
    _centroids.pop(name, None)
    _conds_key = None
    return {"name": name, "kind": "conds" if is_pt else "audio", **meta}


def delete_voice(name: str):
    v = get_voice(name)
    if v.kind == "builtin":
        raise ValueError("the built-in voice can't be deleted")
    if v.name == default_voice():
        raise ValueError("can't delete the default voice; set another default first")
    for f in (f"{name}.wav", f"{name}.pt", f"{name}.json", f"{name}_reference.wav"):
        (VOICES_DIR / f).unlink(missing_ok=True)
    _centroids.pop(name, None)


def _condition(v: Voice):
    """Point the model at a voice (cached; exaggeration is applied per generate() call by Chatterbox)."""
    global _conds_key
    m = tts()
    key = (v.name, v.kind, v.path.stat().st_mtime if v.path else 0)
    if _conds_key == key:
        return
    if v.kind == "audio":
        m.prepare_conditionals(str(v.path), exaggeration=0.5)
    elif v.kind == "conds":
        m.conds = load_conds(str(v.path)).to(m.device)
    else:
        if _builtin_conds is None:
            raise ValueError("this Chatterbox build has no stock voice; add one with POST /voices")
        m.conds = copy.deepcopy(_builtin_conds).to(m.device)
    _conds_key = key


def export_conds(name: str | None) -> bytes:
    """A voice as a reusable Chatterbox conditionals .pt (fast to load, no reference audio needed)."""
    v = get_voice(name)
    if v.kind == "conds":
        return v.path.read_bytes()
    with _lock:
        _condition(v)
        tmp = tmpfile(".pt")
        _save_conds(tts().conds, tmp)
        data = tmp.read_bytes()
        tmp.unlink(missing_ok=True)
        return data


def centroid(v: Voice) -> np.ndarray:
    """Speaker embedding used to score takes. Audio voices: mean over the long reference.
    .pt / built-in voices: the speaker embedding stored in the conditionals (call after _condition)."""
    if v.name not in _centroids:
        if v.kind == "audio":
            refp = VOICES_DIR / f"{v.name}_reference.wav"
            w, sr = sf.read(str(refp if refp.exists() else v.path), dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            win = 20 * sr
            chunks = [w[i: i + win] for i in range(0, max(len(w) - win, 0) + 1, win)] or [w]
            c = np.mean([emb(x, sr) for x in chunks if len(x) > 3 * sr] or [emb(w, sr)], axis=0)
        else:
            c = tts().conds.t3.speaker_emb.detach().float().cpu().numpy().reshape(-1)
        _centroids[v.name] = c / np.linalg.norm(c)
    return _centroids[v.name]


def bootstrap_default():
    """DEFAULT_VOICE_URL (audio sample or .pt, e.g. a private blob link) -> registered at start-up and made the
    default unless a default was already chosen on the /data volume."""
    url = os.environ.get("DEFAULT_VOICE_URL", "").strip()
    if not url:
        return
    name = os.environ.get("DEFAULT_VOICE_NAME", "narrator")
    if not _find(name)[0]:
        import urllib.request
        data = urllib.request.urlopen(url, timeout=180).read()
        add_voice(name, data, url.split("?")[0].rsplit("/", 1)[-1], description="loaded from DEFAULT_VOICE_URL")
    if not (VOICES_DIR / "_default").exists():
        set_default(name)


# ----------------------------------------------------------------------------------- synthesis
@dataclass
class TTSParams:
    exaggeration: float = 0.65
    cfg_weight: float = 0.4
    temperature: float = 0.8
    takes: int = 3                 # best-of-N per sentence (1 = fastest, no QC retries)
    min_accuracy: float = 0.9
    sentence_pause: float = 0.32
    question_pause: float = 0.42
    seed: int | None = None


def _transcribe_words(w: np.ndarray, sr: int, model: str) -> list[dict]:
    import librosa
    segs, _ = whisper(model).transcribe(librosa.resample(w, orig_sr=sr, target_sr=16000), language="en", word_timestamps=True,
                                        beam_size=5, condition_on_previous_text=False)
    return [{"word": x.word.strip(), "start": x.start, "end": x.end, "probability": round(x.probability, 3)}
            for s in segs for x in (s.words or [])]


def synthesize(text: str, voice: str | None = None, p: TTSParams | None = None, progress=None) -> dict:
    """Text -> {'wav', 'sr', 'words' (script words with times), 'picks' (QC report per sentence)}."""
    p = p or TTSParams()
    with _lock:
        m = tts()
        v = get_voice(voice)
        _condition(v)
        E0 = centroid(v) if p.takes > 1 else None
        SR = m.sr
        sents = sentences(text)
        pieces, picks = [], []
        import torch
        for si, s in enumerate(sents):
            best = None
            n_words = len(tokens(s))
            tries = max(1, p.takes) + (3 if p.takes > 1 else 0)
            for take in range(tries):
                seed = (p.seed if p.seed is not None else zlib.crc32(text.encode()) % 100000) + 1000 * si + take
                torch.manual_seed(seed)
                w = m.generate(s, exaggeration=p.exaggeration, cfg_weight=p.cfg_weight, temperature=p.temperature).squeeze(0).cpu().numpy()
                if p.takes <= 1:
                    best = {"w": w, "ws": None, "acc": None, "sim": None, "wps": None, "ok": True, "score": 0, "seed": seed}
                    break
                ws = _transcribe_words(w, SR, QC_MODEL)
                acc = accuracy(s, [x["word"] for x in ws])
                sim = float(emb(w, SR) @ E0)
                speech = (ws[-1]["end"] - ws[0]["start"]) if ws else 0
                wps = n_words / max(speech, 0.1)
                tail = len(w) / SR - (ws[-1]["end"] if ws else 0)
                ok = bool(acc >= p.min_accuracy and 1.5 <= wps <= 4.2)
                cand = {"w": w, "ws": ws, "acc": acc, "sim": sim, "wps": wps, "ok": ok, "score": acc * 2 + sim - (0.3 if tail > 1.2 else 0), "seed": seed}
                if best is None or (cand["ok"], cand["score"]) > (best["ok"], best["score"]):
                    best = cand
                if take + 1 >= p.takes and best["ok"]:
                    break
            w, ws = best["w"], best["ws"]
            if ws:
                a = max(0, int((ws[0]["start"] - 0.06) * SR))
                b = min(len(w), int((ws[-1]["end"] + 0.18) * SR))
                w = w[a:b]
            fade = int(0.012 * SR)
            if len(w) > 2 * fade:
                w = w.copy()
                w[:fade] *= np.linspace(0, 1, fade)
                w[-fade:] *= np.linspace(1, 0, fade)
            pieces.append(w.astype(np.float32))
            if si < len(sents) - 1:
                gap = p.question_pause if s.endswith("?") else p.sentence_pause if s.endswith((".", "!")) else 0.22
                pieces.append(np.zeros(int(gap * SR), dtype=np.float32))
            picks.append({"sentence": s, "accuracy": None if best["acc"] is None else round(best["acc"], 3),
                          "similarity": None if best["sim"] is None else round(best["sim"], 3),
                          "words_per_sec": None if best["wps"] is None else round(best["wps"], 2), "ok": best["ok"], "seed": best["seed"]})
            if progress:
                progress(si + 1, len(sents))
        wav = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
        peak = float(np.max(np.abs(wav))) or 1.0
        wav = (wav * min(1.0, 0.89 / peak)).astype(np.float32)
        asr = _transcribe_words(wav, SR, QC_MODEL)
        words = align(re.sub(r"\s+", " ", text).strip(), asr)
        cov = accuracy(text, [x["word"] for x in asr])
        return {"wav": wav, "sr": SR, "words": words, "coverage": round(cov, 3), "picks": picks, "voice": v.name}


# ----------------------------------------------------------------------------------- transcription
def transcribe(data: bytes, model: str | None = None, language: str | None = None, script: str | None = None,
               vad: bool = True, initial_prompt: str | None = None, beam_size: int = 5) -> dict:
    model = model or TRANSCRIBE_MODEL
    w = load_audio(data, 16000)
    with _lock:
        segs, info = whisper(model).transcribe(w, language=language, word_timestamps=True, beam_size=beam_size,
                                               vad_filter=vad, condition_on_previous_text=False, initial_prompt=initial_prompt)
        segments, words = [], []
        for i, s in enumerate(segs):
            ws = [{"word": x.word.strip(), "start": round(x.start, 3), "end": round(x.end, 3), "probability": round(x.probability, 3)}
                  for x in (s.words or [])]
            segments.append({"id": i, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip(), "words": ws})
            words.extend(ws)
    out = {"model": model, "language": info.language, "language_probability": round(info.language_probability, 3),
           "duration": round(len(w) / 16000, 3), "text": " ".join(s["text"] for s in segments), "segments": segments, "words": words}
    if script:
        out["aligned"] = align(re.sub(r"\s+", " ", script).strip(), words)
        out["script_coverage"] = round(accuracy(script, [x["word"] for x in words]), 3)
    return out


def warmup():
    tts()
    whisper(QC_MODEL)
    ensure_voices()
    try:
        bootstrap_default()
    except Exception as e:  # noqa: BLE001
        print("DEFAULT_VOICE_URL failed:", e, flush=True)
    with _lock:
        _condition(get_voice(None))
    print("ready", flush=True)


def tmpfile(suffix: str) -> Path:
    return Path(tempfile.mkstemp(suffix=suffix)[1])
