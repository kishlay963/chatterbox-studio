# Chatterbox Studio

A ready-to-run GPU container for **voice-clone narration** ([Chatterbox](https://github.com/resemble-ai/chatterbox))
and **word-level transcription** (faster-whisper large-v3). All models are baked into the image, so a container
starts without downloading anything. Bring your own voice: upload a sample or a Chatterbox `.pt` file through the API.

```
ghcr.io/kishlay963/chatterbox-studio:latest
```

## Run it

```bash
# any NVIDIA GPU box with ~10 GB VRAM (Vast, RunPod, local)
docker run -d --gpus all -p 8000:8000 -v studio-data:/data \
  -e API_KEY=change-me \
  -e DEFAULT_VOICE_URL="https://…/my-voice.wav" \
  ghcr.io/kishlay963/chatterbox-studio:latest
# docs: http://HOST:8000/docs
```

| Env | Default | Meaning |
|---|---|---|
| `API_KEY` | none (open) | require header `X-API-Key: <key>` on every endpoint except `/health` |
| `DEFAULT_VOICE_URL` | none | audio sample **or** `.pt` to load at start-up and use as the default voice (a private blob/S3 link works well) |
| `DEFAULT_VOICE_NAME` | `narrator` | the name it's stored under |
| `WHISPER_MODEL` | `large-v3` | model for `/transcribe` (`large-v3-turbo`, `medium.en` … are downloaded on first use) |
| `WARMUP` | `1` | load the models at start-up (~1 min); `GET /health` shows when they're ready |

Voices and jobs live in `/data`: mount a volume to keep them between runs. With no voice configured, the default
is Chatterbox's stock voice (`chatterbox`).

**Vast.ai:** use image `ghcr.io/kishlay963/chatterbox-studio:latest`, expose port `8000`, set the env vars above, and
set the on-start command to `uvicorn app.server:app --host 0.0.0.0 --port 8000`.

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | GPU, loaded models, default voice, queue length |
| GET | `/voices` | List voices (the default is marked) |
| POST | `/voices` | **Add a voice**: multipart `name` + `file` = an audio/video sample (auto-trimmed to the cleanest ~15 s) **or** a Chatterbox `.pt` |
| GET | `/voices/{name}/conds` | **Export** a voice as a `.pt` to keep and re-upload to any fresh container |
| PUT | `/voices/default` | `{"name": "..."}` sets the default voice |
| GET | `/voices/{name}/audio` | Reference clip of an audio-cloned voice |
| DELETE | `/voices/{name}` | Remove a voice |
| POST | `/tts` | JSON: text to audio (wav/mp3/flac/ogg), or `format: "json"` for base64 audio + word timings + QC report |
| POST | `/tts/clone` | Multipart: `text` + `file` (sample or `.pt`): clone and speak in one call |
| POST | `/narrate` | **Sections** to per-section files + combined file + `.srt` + word timings (background job) |
| GET | `/jobs/{id}` · `/jobs/{id}/result` · `/jobs/{id}/files/{name}` | Job progress, timings JSON, files |
| POST | `/transcribe` | **Word-level transcription** as JSON (optionally aligned to a known script) |

`.pt` uploads are loaded with `torch.load(weights_only=True)`, which accepts tensors only and no pickled code.

### Voice controls (on `/tts`, `/tts/clone` and `/narrate`; overridable per section)

| Field | Default | Meaning |
|---|---|---|
| `voice` | server default | which voice |
| `exaggeration` | 0.65 | expressiveness (0.5 neutral, 0.65 lively, 0.9+ dramatic) |
| `cfg_weight` | 0.4 | pacing; lower = slower, more deliberate |
| `temperature` | 0.8 | variation |
| `takes` | 3 | best-of-N per sentence; each take is scored on Whisper word accuracy + similarity to the voice. `1` = fastest |
| `min_accuracy` | 0.9 | a take below this is retried |
| `sentence_pause` | 0.32 | silence between sentences (s) |
| `seed` | auto | reproducible takes |

Tip: write numbers and acronyms the way they should be *said* ("fourteen fifty", "V-Lookup"). All-caps words like
"NEW" may be spelled out letter by letter, so write them in lowercase.

## Examples

```bash
H='X-API-Key: change-me'
# add your voice once, make it the default, and download it as a .pt for next time
curl -s -H "$H" -X POST localhost:8000/voices -F name=me -F file=@sample.mp3 -F make_default=true
curl -s -H "$H" localhost:8000/voices/me/conds -o me.pt
# …on a fresh container: curl -s -H "$H" -X POST localhost:8000/voices -F name=me -F file=@me.pt -F make_default=true

# a quick line in the default voice
curl -s -H "$H" -X POST localhost:8000/tts -H 'Content-Type: application/json' \
  -d '{"text":"Five thousand rows. One slipped decimal point. Would you spot it?","format":"mp3"}' -o line.mp3

# one-shot clone + speak
curl -s -H "$H" -X POST localhost:8000/tts/clone -F text="Hello there." -F file=@someone.wav -F format=mp3 -o hello.mp3

# full narration in sections
curl -s -H "$H" -X POST localhost:8000/narrate -H 'Content-Type: application/json' -d '{
  "sections": [
    {"id":"intro","text":"Your supplier just emailed the new price list."},
    {"id":"hook","text":"Somewhere in here, a price went from fourteen fifty to one hundred and forty-five dollars.","exaggeration":0.8}
  ],
  "takes": 3, "format": "wav"}'
# -> {"job_id": "...", "status_url": "/jobs/..."}; poll it, then GET /jobs/<id>/result

# word-level transcription (large-v3), aligned to the script you expected
curl -s -H "$H" -X POST localhost:8000/transcribe -F file=@take.wav -F language=en -F script="$(cat script.txt)"
```

`/transcribe` returns:
```json
{"model":"large-v3","language":"en","duration":12.4,"text":"...",
 "segments":[{"id":0,"start":0.0,"end":3.1,"text":"...","words":[{"word":"Your","start":0.12,"end":0.3,"probability":0.98}]}],
 "words":[...], "aligned":[{"word":"Your","start":0.12,"end":0.3}], "script_coverage":0.98}
```

## Build

GitHub Actions builds and pushes `ghcr.io/<owner>/chatterbox-studio` on every push to `main`. To build by hand:
`docker build -t chatterbox-studio .` (needs ~25 GB of free disk). To bake voices into a private build, drop
`<name>.wav` / `<name>.pt` files into `voices/` before building.
