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
1. **Differentiable DNC Memory (`DifferentiableRAM`)**: Cosine content-based addressing with a learnable, sigmoid-bounded addressing strength $\beta \in (0, 20)$, and soft read/erase/write vectors — all gated by live routing weights (gradients flow every cycle).
2. **Continuous Stack (`DifferentiableStack`)**: Recurrent continuous push/pop depth distribution $V_t \in [0,1]^V$ (clamped, so activations and gradients stay bounded over long horizons).
3. **Neural ALU (`DifferentiableALU`)**: Continuous relaxed arithmetic, bitwise XNOR/AND/OR, MBA polynomial units, and a learned affine unit.
4. **Dynamic Type Signature Routing**: $\alpha_{t,u} = \text{Softmax}\left(\frac{W_q I_t \cdot s_u}{\tau}\right)$ dynamically routing execution across 13 signatures (9 ALU units + MEM_READ/MEM_WRITE/STACK_PUSH/STACK_POP).
5. **Program Induction via Gradient Descent**: Programs are trained directly through the VM via BPTT with global-norm gradient clipping, curriculum, and $\tau$-annealing ($\tau \to 0$ at test time). The test asserts deterministic convergence, live gradients for **every** parameter and memory subsystem, and an 8-seed robustness sweep.

---

## 🔬 Formal Verification Guarantees (TLA+ / TLC)

The TLA+ specification in **`tla/DeepStagedVM.tla`** is model-checked **exhaustively**: `Init` is nondeterministic over the operand registers, so TLC enumerates *every* operand pair (256 initial states, 10 496 distinct states) rather than a single scripted trace:
1. **`BitSliceCorrectness`**: Proved for **all** operand pairs — the Full-Adder stages compute $(A + B) \bmod 2^N$ with zero bit corruption across the complete input space.
2. **`MBAEquivalence`**: Non-vacuous — the MBA stages apply the genuine identities $(a \oplus b) + 2(a \wedge b) = a + b$ and $(a \vee b) + (a \wedge b) = a + b$, and the invariant verifies they preserve exact arithmetic for every operand pair.
3. **`DataHazardFreedom`**: Operand latches are verified against the register file before any execution stage touches state.
4. **`CommitIntegrity`**: Architectural state (GPRs, PC) is updated strictly at commit.
5. **`Termination` (temporal liveness)**: Under weak fairness (`WF_vars(Next)`), TLC checks the temporal property $\Diamond\,\texttt{vmHalted}$ — no livelock and no deadlock along any fair behavior.

The Sail synthesizer (**`scripts/generate_100_stage_vm.py`**) emits a 112-stage specification using constructs attested in the upstream sail-riscv model (effectful `function ... : unit -> unit = { ... }`, two-argument `vector(n, bits(w))` registers, indexed register writes, `to_bits_truncate`, `while ... do` loops) plus an **instance-parameterized** matching TLA+ model: two synthesized architectures with different cryptographic seeds produce different verification modules, and TLC proves each over all operand pairs (~22 800 states).
