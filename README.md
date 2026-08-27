# c_mutatedStages

> 🧪 Rigorous TLA+ formal specification, stage catalog, and Sail ISA synthesizer for deeply staged (100+ micro-stages) virtual machines.

## ⚡ Quickstart

### 1. Verify Deep Staged Model in TLA+
Formally prove bit-sliced ALU mathematical equivalence, MBA polynomial invariance, Feistel deobfuscation, and deadlock-freedom:
```bash
java -cp tla/tla2tools.jar tlc2.TLC tla/DeepStagedVM.tla -config tla/DeepStagedVM.cfg
```

### 2. Synthesize 100+ Stage Architecture in Sail Language
Generate a genuine 112-stage formal architecture specification in Cambridge Sail format (`.sail`):
```bash
python3 scripts/generate_100_stage_vm.py
```

---

## 🔬 Formal Verification Guarantees (TLA+ / TLC)

The TLA+ specification in **`tla/DeepStagedVM.tla`** mathematically proves:
1. **`BitSliceCorrectness`**: Induction proof that 64-bit Full-Adder stages across latches compute $(A + B) \pmod{2^N}$ with zero bit corruption.
2. **`MBAEquivalence`**: Mixed Boolean-Arithmetic polynomial transformations retain exact algebraic equivalence.
3. **`DataHazardFreedom`**: Operand latches are verified before any execution stage touches state.
4. **`CommitIntegrity`**: Architectural state (GPRs, PC) is updated strictly at commit.
5. **`Liveness`**: Full absence of deadlocks across all micro-stages.

---

## 📂 Project Structure

- **`tla/DeepStagedVM.tla`** — Formal TLA+ specification for deep micro-staged VMs.
- **`scripts/generate_100_stage_vm.py`** — Formal synthesizer emitting complete Sail language specifications (`.sail`).
- **`spec/stages_db.json`** — Upstream formal catalog (Ghidra P-Code, WASM, eBPF, QEMU, RISC-V).
- **`generated_architectures/`** — Generated `.sail` architecture files and JSON metadata.
