#!/usr/bin/env python3
"""Post-registered compatibility-only adapter: Python 3.10 lacks hashlib.file_digest.

Does not modify frozen v084_evidence.py, trial protocol, controller, output, or thresholds.
Preserve the initial console failure as an honest negative.
"""
import hashlib
from pathlib import Path
import v084_evidence as frozen
def sha_compatible(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def main():
    frozen.sha=sha_compatible
    print("POSTREGISTERED_PY310_HASH_COMPAT_ONLY; frozen analysis functions reused",flush=True)
    return frozen.summarize()
if __name__=="__main__":main()
