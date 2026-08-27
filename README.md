# c_mutatedStages

> 🧪 Formal TLA+ specification, stage catalog, and architecture generator for polymorphic multi-stage Virtual Machines.

## ⚡ Quickstart

### 1. Verify Formal Model (TLA+)
Check formal invariants and deadlock-freedom with TLC:
```bash
java -cp tla/tla2tools.jar tlc2.TLC tla/MutatedStageVM.tla -config tla/MutatedStageVM.cfg
```

### 2. Generate Unique VM Architectures
Synthesize verified VM configurations from the stages database:
```bash
python3 scripts/generate_arch.py 3
```

---

## 📂 Project Structure

- **`spec/stages_db.json`** — Catalog of execution paradigms (Stack, Register, Acc), dispatch mechanisms (Computed goto, Switch, Coroutine), opcode formats, and pipeline stages.
- **`tla/MutatedStageVM.tla`** — Formal TLA+ model verifying data hazard freedom, stage causality, and state commit integrity.
- **`scripts/generate_arch.py`** — Generator synthesizing verified unique VM architecture profiles.
- **`generated_architectures/`** — Generated JSON profiles ready for C VM code emission.
