"""Download models into the image so containers start with no downloads.  usage: python bake_models.py chatterbox|<whisper-model>"""
import sys
what = sys.argv[1]
if what == "chatterbox":
    from chatterbox.tts import ChatterboxTTS
    ChatterboxTTS.from_pretrained(device="cpu")   # downloads and proves the weights load
else:
    from faster_whisper.utils import download_model
    print(download_model(what))
print("baked", what)
