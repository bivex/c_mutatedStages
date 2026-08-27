# c_mutatedStages

> 🧪 Formal TLA+ specification, stage catalog, and Apple MLX AI Transformer for synthesizing unique 100+ multi-stage Virtual Machines.

## ⚡ Quickstart

### 1. Generate 100+ Stage VM with Apple MLX AI Model
Train/run the MLX Causal Transformer on Apple Silicon GPU to synthesize and compile a 100+ stage VM:
```bash
# Generate and run executable 108-stage C VM
python3 mlx_model/generate_100_stage_vm.py
```

### 2. Verify Formal Model (TLA+)
Check formal invariants and deadlock-freedom with TLC:
```bash
java -cp tla/tla2tools.jar tlc2.TLC tla/MutatedStageVM.tla -config tla/MutatedStageVM.cfg
```

### 3. Generate & Verify 3 Random Unique Architectures
Synthesize and verify 3 unique architectures with Clang and TLA+:
```bash
python3 scripts/verify_architectures.py
```

---

## 📂 Project Structure

- **`mlx_model/`** — Neural network generative engine powered by Apple MLX (`mlx.nn`, `mlx.core`):
  - `vm_transformer.py` — Causal Transformer architecture with 135+ stage vocabulary.
  - `train_generator.py` — Training pipeline compiled on Apple Silicon GPU.
  - `generate_100_stage_vm.py` — Autoregressive generator producing full 100+ stage C engines.
- **`spec/stages_db.json`** — Comprehensive database (NSA Ghidra P-Code, W3C WASM, Linux eBPF, QEMU, RISC-V).
- **`tla/MutatedStageVM.tla`** — Formal TLA+ model proving data hazard freedom and deadlock-free execution.
- **`generated_architectures/`** — Generated JSON profiles and executable C engines.
