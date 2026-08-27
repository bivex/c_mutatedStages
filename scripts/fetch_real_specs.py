#!/usr/bin/env python3
"""
Real Specification Downloader via GitHub CLI (gh).
Downloads actual upstream specification files from:
- qemu/qemu (decodetree instruction tables)
- LuaJIT/LuaJIT (bytecode & stage tables)
- rems-project/sail-riscv (formal Sail models)
- NationalSecurityAgency/ghidra (Sleigh processor definitions)
- iovisor/ubpf (eBPF VM opcodes)
- WebAssembly/spec (Wasm binary decoder)
"""

import subprocess
import json
import base64
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = BASE_DIR / "upstream_specs"

UPSTREAM_TARGETS = [
    {
        "category": "qemu",
        "repo": "qemu/qemu",
        "path": "target/riscv/insn32.decode",
        "outfile": "qemu/riscv_insn32.decode"
    },
    {
        "category": "qemu",
        "repo": "qemu/qemu",
        "path": "target/riscv/insn16.decode",
        "outfile": "qemu/riscv_insn16.decode"
    },
    {
        "category": "qemu",
        "repo": "qemu/qemu",
        "path": "target/arm/tcg/a64.decode",
        "outfile": "qemu/arm_a64.decode"
    },
    {
        "category": "luajit",
        "repo": "LuaJIT/LuaJIT",
        "path": "src/lj_bc.h",
        "outfile": "luajit/lj_bc.h"
    },
    {
        "category": "ebpf",
        "repo": "iovisor/ubpf",
        "path": "vm/ubpf_int.h",
        "outfile": "ebpf/ubpf_int.h"
    },
    {
        "category": "sail",
        "repo": "rems-project/sail-riscv",
        "path": "model/core/arithmetic.sail",
        "outfile": "sail/arithmetic.sail"
    },
    {
        "category": "ghidra",
        "repo": "NationalSecurityAgency/ghidra",
        "path": "Ghidra/Processors/BPF/data/languages/BPF.sinc",
        "outfile": "ghidra/BPF.sinc"
    },
    {
        "category": "wasm",
        "repo": "WebAssembly/spec",
        "path": "interpreter/binary/decode.ml",
        "outfile": "wasm/decode.ml"
    }
]

def fetch_file_via_gh(repo, path, outpath):
    print(f"[*] Fetching {repo}:{path} via gh CLI...")
    cmd = ["gh", "api", f"repos/{repo}/contents/{path}"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [-] Failed to fetch {repo}:{path}: {res.stderr.strip()}")
        return False
    try:
        data = json.loads(res.stdout)
        if "content" in data:
            raw_bytes = base64.b64decode(data["content"])
            outpath.parent.mkdir(parents=True, exist_ok=True)
            with open(outpath, "wb") as f:
                f.write(raw_bytes)
            print(f"  [+] Saved: {outpath.relative_to(BASE_DIR)} ({len(raw_bytes):,} bytes)")
            return True
        elif "download_url" in data and data["download_url"]:
            # For large files where base64 content is omitted by GitHub API
            curl_cmd = ["curl", "-s", "-L", "-o", str(outpath), data["download_url"]]
            outpath.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(curl_cmd, check=True)
            print(f"  [+] Saved via direct download: {outpath.relative_to(BASE_DIR)}")
            return True
    except Exception as e:
        print(f"  [-] Error decoding {repo}:{path}: {e}")
        return False
    return False

def main():
    UPSTREAM_DIR.mkdir(parents=True, exist_ok=True)
    success = 0
    total = len(UPSTREAM_TARGETS)
    
    print(f"=== Downloading Real Architecture Specs via GitHub CLI (gh) ===")
    for target in UPSTREAM_TARGETS:
        outpath = UPSTREAM_DIR / target["outfile"]
        if fetch_file_via_gh(target["repo"], target["path"], outpath):
            success += 1
            
    print(f"\n[✓] Completed: {success}/{total} real architecture files downloaded to 'upstream_specs/'")

if __name__ == "__main__":
    main()
