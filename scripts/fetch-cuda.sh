#!/bin/sh
# Download the CUDA 13 + cuDNN 9 runtime libraries that ONNX Runtime's CUDA provider needs (~1.5 GB unpacked),
# for a boardcast built with `--features cuda`. Taken from NVIDIA's PyPI wheels, so no CUDA toolkit install is needed;
# you do need an NVIDIA driver new enough for CUDA 13 (r580 or later: check `nvidia-smi`).
# Saved to ${XDG_DATA_HOME:-~/.local/share}/boardcast/cuda, where boardcast looks for it.
set -e
dir="${XDG_DATA_HOME:-$HOME/.local/share}/boardcast/cuda"
mkdir -p "$dir"
python3 - "$dir" <<'EOF'
import hashlib, json, os, sys, tempfile, urllib.request, zipfile

dest = sys.argv[1]
# CUDA 13.0 runs on any CUDA 13 driver; cuDNN 9 for CUDA 13.
wheels = [
    ("nvidia-cuda-runtime", "13.0.96"),
    ("nvidia-cublas", "13.0.2.14"),
    ("nvidia-curand", "10.4.0.35"),
    ("nvidia-cufft", "12.0.0.61"),
    ("nvidia-cuda-nvrtc", "13.0.88"),
    ("nvidia-cudnn-cu13", "9.19.1.2"),
]
for name, ver in wheels:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{ver}/json") as r:
        files = json.load(r)["urls"]
    f = next(f for f in files if f["filename"].endswith(".whl") and "x86_64" in f["filename"])
    print(f"{f['filename']} ({f['size'] >> 20} MB)", flush=True)
    with tempfile.TemporaryFile() as tmp:
        h = hashlib.sha256()
        with urllib.request.urlopen(f["url"]) as r:
            while chunk := r.read(1 << 20):
                h.update(chunk)
                tmp.write(chunk)
        if h.hexdigest() != f["digests"]["sha256"]:
            sys.exit(f"checksum mismatch for {f['filename']}")
        with zipfile.ZipFile(tmp) as z:
            for m in z.infolist():
                base = os.path.basename(m.filename)
                if "/lib/" in m.filename and ".so" in base:
                    with z.open(m) as src, open(os.path.join(dest, base), "wb") as out:
                        while chunk := src.read(1 << 20):
                            out.write(chunk)
EOF
echo "ok: $dir"
