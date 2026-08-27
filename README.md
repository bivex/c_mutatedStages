# c_mutatedStages

> 🧪 Rigorous TLA+ formal specification, Sail ISA synthesizer, and Apple MLX Differentiable Neural Virtual Machine (DNC RAM + Continuous Stack + Neural ALU).

## ⚡ Quickstart

### 1. Run Differentiable Neural Virtual Machine (Apple MLX Metal GPU)
Execute and train programs directly inside the continuous neural virtual machine via Backprop Through Time (BPTT):
```bash
python3 mlx_model/differentiable_neural_vm.py
```

### 2. Verify Deep Staged Model in TLA+
Formally prove bit-sliced ALU mathematical equivalence, MBA polynomial invariance, Feistel deobfuscation, and deadlock-freedom:
```bash
java -cp tla/tla2tools.jar tlc2.TLC tla/DeepStagedVM.tla -config tla/DeepStagedVM.cfg
```

### 3. Synthesize 100+ Stage Architecture in Sail Language
Generate a genuine 112-stage formal architecture specification in Cambridge Sail format (`.sail`):
```bash
python3 scripts/generate_100_stage_vm.py
```

---

## 🧠 Differentiable Neural VM Architecture (Apple MLX)

The neural virtual machine implemented in **`mlx_model/differentiable_neural_vm.py`** is an end-to-end differentiable execution engine:
1. **Differentiable DNC Memory (`DifferentiableRAM`)**: Cosine content-based addressing, dynamic allocation, and soft read/erase/write vectors.
2. **Continuous Stack (`DifferentiableStack`)**: Recurrent continuous push/pop depth distribution $V_t$.
3. **Neural ALU (`DifferentiableALU`)**: Continuous relaxed arithmetic, bitwise XNOR/AND/OR, and MBA polynomial units.
4. **Dynamic Type Signature Routing**: $\alpha_{t,u} = \text{Softmax}\left(\frac{q_t \cdot s_u}{\tau}\right)$ dynamically routing execution to functional units.
5. **Program Induction via Gradient Descent**: Programs can be trained and synthesized directly through the VM via Backpropagation Through Time (BPTT).

---

## 🔬 Formal Verification Guarantees (TLA+ / TLC)

The TLA+ specification in **`tla/DeepStagedVM.tla`** mathematically proves:
1. **`BitSliceCorrectness`**: Induction proof that 64-bit Full-Adder stages across latches compute $(A + B) \pmod{2^N}$ with zero bit corruption.
2. **`MBAEquivalence`**: Mixed Boolean-Arithmetic polynomial transformations retain exact algebraic equivalence.
3. **`DataHazardFreedom`**: Operand latches are verified before any execution stage touches state.
4. **`CommitIntegrity`**: Architectural state (GPRs, PC) is updated strictly at commit.
5. **`Liveness`**: Full absence of deadlocks across all micro-stages.
