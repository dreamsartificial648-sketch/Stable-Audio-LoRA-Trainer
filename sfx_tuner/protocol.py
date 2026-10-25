"""Small JSON event protocol shared by CPU and GPU workers."""
import json
from pathlib import Path
import time

PREFIX = "SFX_EVENT:"


def emit(**payload):
    print(PREFIX + json.dumps(payload, allow_nan=False), flush=True)


def stopped(request):
    return Path(request["stop_file"]).exists()


def wait_if_paused(request):
    # Cooperative pause at batch/sampling boundaries keeps CUDA contexts intact.
    while Path(request["pause_file"]).exists() and not stopped(request):
        time.sleep(.1)
