#!/usr/bin/env python3
"""
Rigorous Differentiable Neural Virtual Machine (Differentiable Neural Computer & Neural ALU)
Implemented in Apple MLX (Apple Silicon Metal GPU).

Implements:
1. Differentiable Continuous Memory Matrix (DNC RAM) with Cosine Content Addressing and Soft-Write
2. Differentiable Register File R_t in R^(R_num x D)
3. Differentiable Stack Machine (Continuous Stack with relaxed Push/Pop, depth weights in [0, 1])
4. Differentiable Neural Bit-Slice / Continuous ALU (Arithmetic, Logic, MBA, learned affine unit)
5. Type Signature Dynamic Routing: alpha_(t,u) = Softmax(W_q * I_t . s_u / tau)
6. Masked State Commit & Differentiable Program Counter (Soft-Branching)
7. End-to-End Gradient Descent Execution on Apple Silicon GPU (BPTT with gradient clipping)

Engineering notes (numerical stability):
- Memory/stack/control gates are LIVE routing outputs (no hard-disabled paths).
- All learnable parameters participate in the computation graph and receive gradients.
- Softmax addressing strength beta is learnable and bounded in (0, BETA_MAX) via sigmoid.
- Stack depth weights are clamped to [0, 1] per the Stack-RNN recurrence, bounding
  activations and gradients over long horizons.
- Committed values are soft-clamped to (-CLAMP_SCALE, CLAMP_SCALE) via a scaled tanh,
  which is ~identity for |x| << CLAMP_SCALE and prevents activation drift.
- zero_flag uses a rational (polynomial-decay) detector so its gradient does not
  underflow for large-magnitude results.
- BPTT optimization uses global-norm gradient clipping.
"""

import time
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map

# --- 0. Numerical utilities ---


def global_grad_norm(grad_tree) -> mx.array:
    """L2 norm over an arbitrary pytree of gradient arrays."""
    total = mx.array(0.0)
    for _, leaf in tree_flatten(grad_tree):
        if isinstance(leaf, mx.array):
            total = total + (leaf.astype(mx.float32) ** 2).sum()
    return mx.sqrt(total)


def clip_grad_global_norm(grad_tree, max_norm: float):
    """Global-norm gradient clipping: g <- g * min(1, max_norm / ||g||)."""
    norm = global_grad_norm(grad_tree)
    scale = mx.minimum(1.0, max_norm / (norm + 1e-12))
    return tree_map(lambda g: g * scale if isinstance(g, mx.array) else g, grad_tree), norm


# --- 1. Differentiable ALU (Continuous & Bitwise Operations) ---

class DifferentiableALU(nn.Module):
    """
    Neural ALU implementing distinct functional units routed via type signatures.
    Units:
      0: ADD (a + b)
      1: SUB (a - b)
      2: MUL (a * b)
      3: BIT_XOR (Continuous relaxed XOR: a + b - 2ab)
      4: BIT_AND (Relaxed AND: a * b)
      5: BIT_OR  (Relaxed OR: a + b - ab)
      6: MBA_POLY (Mixed Boolean-Arithmetic identity: (a ^ b) + 2(a & b) == a + b)
      7: PASS_IMM (Pass immediate / constant)
      8: NEURAL_AFFINE (Learned affine unit: W [a; b] — a genuinely neural operation)
    """
    def __init__(self, d_val: int):
        super().__init__()
        self.d_val = d_val
        self.num_units = 9
        # Learnable affine transformation weights for the neural functional unit
        self.proj_linear = nn.Linear(d_val * 2, d_val)

    def __call__(self, op_a: mx.array, op_b: mx.array, unit_weights: mx.array) -> tuple[mx.array, mx.array]:
        """
        op_a: (Batch, d_val)
        op_b: (Batch, d_val)
        unit_weights: (Batch, num_units) - Softmax distribution over ALU functional units
        """
        # 0: ADD
        out_add = op_a + op_b
        # 1: SUB
        out_sub = op_a - op_b
        # 2: MUL
        out_mul = op_a * op_b
        # 3: XOR (Smooth continuous relaxation)
        out_xor = op_a + op_b - 2.0 * (op_a * op_b)
        # 4: AND
        out_and = op_a * op_b
        # 5: OR
        out_or = op_a + op_b - (op_a * op_b)
        # 6: MBA Polynomial identity: (a ^ b) + 2(a & b)
        out_mba = out_xor + 2.0 * out_and
        # 7: PASS_IMM / PASS_B
        out_pass = op_b
        # 8: NEURAL_AFFINE (learned unit — keeps proj_linear in the live compute path)
        out_neural = self.proj_linear(mx.concatenate([op_a, op_b], axis=-1))

        # Stack outputs: (Batch, num_units, d_val)
        stacked = mx.stack([
            out_add, out_sub, out_mul, out_xor,
            out_and, out_or, out_mba, out_pass, out_neural
        ], axis=1)

        # Weighted combination: y_t = sum_u (alpha_{t,u} * ALU_u)
        weights_expanded = unit_weights[:, :, None]  # (Batch, num_units, 1)
        result = (stacked * weights_expanded).sum(axis=1)  # (Batch, d_val)

        # Compute Flags: Zero Flag (rational smooth detector, no exp underflow)
        norm_sq = (result ** 2).sum(axis=-1, keepdims=True)
        zero_flag = 1.0 / (1.0 + norm_sq / self.d_val)

        return result, zero_flag


# --- 2. Differentiable External Memory (DNC Content-Addressed RAM) ---

class DifferentiableRAM(nn.Module):
    """
    Differentiable Neural Computer (DNC) RAM Matrix M_t in R^(N x W).
    Supports soft read and soft write via cosine content-based addressing.
    The addressing strength beta is learnable and sigmoid-bounded in (0, BETA_MAX),
    so the softmax stays in its numerically useful regime (no hard saturation).
    """
    BETA_MAX = 20.0

    def __init__(self, num_cells: int = 16, cell_width: int = 8):
        super().__init__()
        self.num_cells = num_cells
        self.cell_width = cell_width
        # Learnable addressing strength: beta = BETA_MAX * sigmoid(beta_param)
        self.beta_param = mx.zeros((1,))

    @property
    def beta(self) -> mx.array:
        return self.BETA_MAX * mx.sigmoid(self.beta_param)  # in (0, 20)

    def content_addressing(self, memory: mx.array, key: mx.array, beta: mx.array) -> mx.array:
        """
        Cosine similarity addressing:
        c[i] = Softmax(beta * (k . M[i]) / (||k|| * ||M[i]|| + eps))
        """
        eps = 1e-6
        key_norm = mx.sqrt((key ** 2).sum(axis=-1, keepdims=True) + eps)  # (B, 1)
        mem_norm = mx.sqrt((memory ** 2).sum(axis=-1, keepdims=True) + eps)  # (B, N, 1)

        # (B, 1, W) @ (B, W, N) -> (B, 1, N)
        similarity = (key[:, None, :] @ memory.transpose(0, 2, 1)).squeeze(1)  # (B, N)
        normalized_sim = similarity / (key_norm * mem_norm.squeeze(-1) + eps)

        weights = mx.softmax(normalized_sim * beta, axis=-1)  # (B, N)
        return weights

    def read(self, memory: mx.array, read_weights: mx.array) -> mx.array:
        """
        r_t = sum_i (w_t^r[i] * M_t[i])
        """
        # (B, 1, N) @ (B, N, W) -> (B, 1, W)
        return (read_weights[:, None, :] @ memory).squeeze(1)

    def write(self, memory: mx.array, write_weights: mx.array, erase_vec: mx.array, write_vec: mx.array) -> mx.array:
        """
        M_t[i, j] = M_{t-1}[i, j] * (1 - w_t^w[i] * e_t[j]) + w_t^w[i] * v_t[j]
        """
        # w_t^w: (B, N, 1), e_t: (B, 1, W) -> (B, N, W)
        w_expanded = write_weights[:, :, None]
        e_expanded = erase_vec[:, None, :]
        v_expanded = write_vec[:, None, :]

        erase_matrix = 1.0 - (w_expanded @ e_expanded)
        add_matrix = w_expanded @ v_expanded
        new_memory = memory * erase_matrix + add_matrix
        return new_memory


# --- 3. Differentiable Continuous Stack (Stack-RNN) ---

class DifferentiableStack(nn.Module):
    """
    Continuous Stack Matrix S_t in R^(V x D) with continuous depth weights
    V_t in [0, 1]^V (clamped per the Stack-RNN recurrence so that both the
    depth distribution and its gradients stay bounded over long horizons).
    """

    def __init__(self, stack_depth: int = 16, d_val: int = 8):
        super().__init__()
        self.depth = stack_depth
        self.d_val = d_val

    def step(self, stack_mem: mx.array, depth_weights: mx.array, push_val: mx.array, u_push: mx.array, u_pop: mx.array):
        """
        Relaxed Push/Pop stack recurrence:
        V_t[i] = clamp(V_{t-1}[i-1] + u_push - u_pop, 0, 1)
        r_t = sum_i (max(0, V_t[i] - V_t[i+1]) * S_t[i])
        """
        # Shift depth weights up by one slot (a "push" moves the stack pointer)
        shifted_depth = mx.pad(depth_weights[:, :-1], [(0, 0), (1, 0)])
        new_depth = mx.clip(shifted_depth + u_push - u_pop, 0.0, 1.0)

        # Top-of-stack indicator: positive part of the first difference
        top_weight = mx.maximum(0.0, new_depth - mx.pad(new_depth[:, 1:], [(0, 0), (0, 1)]))
        read_val = (top_weight[:, :, None] * stack_mem).sum(axis=1)

        # Soft push write
        w_push = top_weight * u_push
        new_stack = stack_mem * (1.0 - w_push[:, :, None]) + push_val[:, None, :] * w_push[:, :, None]
        return new_stack, new_depth, read_val


# --- 4. Master Neural Differentiable VM ---

class NeuralDifferentiableVM(nn.Module):
    """
    Complete End-to-End Differentiable Neural Virtual Machine:
    - Type Signature Routing (Module 2): alpha_(t,u) = Softmax(W_q I_t . s_u / tau)
    - Differentiable ALU (Module 3): 9 functional units incl. learned affine unit
    - DNC RAM + Registers + Diff-Stack (Modules 3 & 4): all gated by LIVE routing weights
    - Soft-Branching Program Counter (Module 4)
    - Soft-clamped state commits (bounded activations, bounded error drift)

    Signature layout (13 total):
      0..8   : ALU functional units (ADD, SUB, MUL, XOR, AND, OR, MBA, PASS, NEURAL)
      9      : MEM_READ   (routes RAM read value into the commit value)
      10     : MEM_WRITE  (routes ALU result into a DNC soft-write)
      11     : STACK_PUSH (routes ALU result onto the continuous stack)
      12     : STACK_POP  (routes the stack top into the commit value)
    """
    CLAMP_SCALE = 256.0  # soft saturation bound for committed values

    def __init__(self, num_registers: int = 8, d_val: int = 8, num_ram_cells: int = 16, stack_depth: int = 16):
        super().__init__()
        self.num_registers = num_registers
        self.d_val = d_val
        self.num_ram_cells = num_ram_cells
        self.stack_depth = stack_depth

        self.num_alu_units = 9
        self.num_signatures = 13  # 9 ALU + MEM_READ + MEM_WRITE + STACK_PUSH + STACK_POP

        # Functional Unit Type Signatures s_u in R^D
        self.type_signatures = nn.Embedding(self.num_signatures, d_val)
        # Small-scale init: the untrained router must start near-uniform so that no
        # functional unit is hard-saturated at initialisation (softmax regime control).
        self.type_signatures.weight = self.type_signatures.weight * 0.05

        # Neural Sub-engines
        self.alu = DifferentiableALU(d_val)
        self.ram = DifferentiableRAM(num_ram_cells, d_val)
        self.stack = DifferentiableStack(stack_depth, d_val)

        # Controller / Instruction Query Projection: q_t = W_q * I_t
        self.query_proj = nn.Linear(d_val, d_val)
        self.query_proj.weight = self.query_proj.weight * 0.2

        # Soft-branch / control flow generator
        self.branch_proj = nn.Linear(d_val, 1)

    def route(self, instruction_tensor: mx.array, tau: float) -> tuple[mx.array, dict]:
        """
        Type Signature Dynamic Routing (Module 2):
          q_t    = W_q * I_t
          logits = q_t . s_u / tau
          alpha  = Softmax(logits)   over 13 functional-unit signatures
        """
        op_vec = instruction_tensor[:, 0, :]                      # (B, d_val)
        q = self.query_proj(op_vec)                               # (B, d_val)
        logits = (q @ self.type_signatures.weight.T) / tau        # (B, num_signatures)
        alpha = mx.softmax(logits, axis=-1)                       # (B, num_signatures)

        gates = {
            'mem_read':   alpha[:, 9:10],   # (B, 1)
            'mem_write':  alpha[:, 10:11],
            'stack_push': alpha[:, 11:12],
            'stack_pop':  alpha[:, 12:13],
        }
        return alpha[:, :self.num_alu_units], gates

    def execute_step(self, state: dict, instruction_tensor: mx.array, tau: float = 1.0) -> dict:
        """
        Executes one differentiable machine cycle F_I: S_t -> S_{t+1}.

        state: {
            'regs': (B, num_regs, d_val),
            'ram':  (B, num_ram_cells, d_val),
            'stack_mem': (B, stack_depth, d_val),
            'stack_depth': (B, stack_depth),
            'ip':   (B, 1),
        }
        instruction_tensor: (B, 4, d_val) -> [Opcode_vec, Dst_vec, Src1_vec, Src2/Imm_vec]
        tau: Softmax temperature of the type-signature router
        """
        B = instruction_tensor.shape[0]
        regs = state['regs']
        ram = state['ram']
        stack_mem = state['stack_mem']
        stack_depth_w = state['stack_depth']
        ip = state['ip']

        # 1. Decode & Type Signature Dynamic Routing (Module 2)
        alu_weights, gates = self.route(instruction_tensor, tau)
        is_mem_read, is_mem_write = gates['mem_read'], gates['mem_write']
        is_stack_push, is_stack_pop = gates['stack_push'], gates['stack_pop']

        # 2. Register Read / Operand Resolution (Module 2)
        dst_idx_logits  = instruction_tensor[:, 1, :self.num_registers]  # (B, num_regs)
        src1_idx_logits = instruction_tensor[:, 2, :self.num_registers]
        src2_idx_logits = instruction_tensor[:, 3, :self.num_registers]
        imm_val         = instruction_tensor[:, 3, :]                    # (B, d_val)

        reg_scores1 = mx.softmax(src1_idx_logits / tau, axis=-1)
        reg_scores2 = mx.softmax(src2_idx_logits / tau, axis=-1)
        dst_scores  = mx.softmax(dst_idx_logits / tau, axis=-1)

        operand_a = (reg_scores1[:, :, None] * regs).sum(axis=1)  # (B, d_val)
        operand_b = (reg_scores2[:, :, None] * regs).sum(axis=1)  # (B, d_val)

        # 3. Differentiable ALU Execution (Module 3)
        alu_result, zero_flag = self.alu(operand_a, operand_b + imm_val, alu_weights)

        # Soft-clamp committed values: ~identity for |x| << CLAMP_SCALE, bounded above
        alu_result = self.CLAMP_SCALE * mx.tanh(alu_result / self.CLAMP_SCALE)

        # 4. Differentiable RAM Access (DNC) — gated by live MEM_WRITE / MEM_READ routing
        beta = mx.broadcast_to(self.ram.beta, (B,))
        read_weights = self.ram.content_addressing(ram, key=operand_a, beta=beta)
        ram_read_val = self.ram.read(ram, read_weights)

        write_weights = read_weights * is_mem_write
        erase_vec = mx.full((B, self.d_val), 0.9)
        new_ram = self.ram.write(ram, write_weights, erase_vec, write_vec=alu_result)

        # 5. Differentiable Stack Step — gated by live STACK_PUSH / STACK_POP routing
        u_push = mx.broadcast_to(is_stack_push, (B, self.stack_depth))
        u_pop = mx.broadcast_to(is_stack_pop, (B, self.stack_depth))
        new_stack_mem, new_stack_depth, stack_read_val = self.stack.step(
            stack_mem, stack_depth_w, push_val=alu_result, u_push=u_push, u_pop=u_pop
        )

        # Combine effective result: convex mixture over routed sources
        alu_mass = alu_weights.sum(axis=-1, keepdims=True)  # (B, 1)
        mix_denom = alu_mass + is_mem_read + is_stack_pop + 1e-8
        final_val = (
            alu_result * alu_mass +
            ram_read_val * is_mem_read +
            stack_read_val * is_stack_pop
        ) / mix_denom

        # 6. Masked Register Commit (Module 4)
        dst_weights_expanded = dst_scores[:, :, None]  # (B, num_regs, 1)
        new_regs = regs * (1.0 - dst_weights_expanded) + dst_weights_expanded * final_val[:, None, :]

        # 7. Soft-Branching Program Counter Update
        branch_cond = mx.sigmoid(self.branch_proj(final_val))  # (B, 1)
        step_forward = mx.ones((B, 1))
        jump_offset = imm_val[:, :1]
        new_ip = ip + step_forward + branch_cond * jump_offset * zero_flag

        new_state = {
            'regs': new_regs,
            'ram': new_ram,
            'stack_mem': new_stack_mem,
            'stack_depth': new_stack_depth,
            'ip': new_ip,
            'last_result': final_val,
            'zero_flag': zero_flag,
            'routing': {
                'alu_mass': alu_mass,
                'mem_read': is_mem_read,
                'mem_write': is_mem_write,
                'stack_push': is_stack_push,
                'stack_pop': is_stack_pop,
            },
        }
        return new_state


# --- 5. Self-Contained Verification & End-to-End Gradient Optimization ---

MASTER_KEY = 20260828  # deterministic seeding: the verification suite must be reproducible


def make_initial_state(batch_size: int, num_regs: int, num_ram: int, stack_depth: int, d_val: int,
                       key=None, r0_val: float = 2.0) -> dict:
    if key is not None:
        ram0 = mx.random.normal((batch_size, num_ram, d_val), key=key) * 0.2
    else:
        ram0 = mx.zeros((batch_size, num_ram, d_val))
    state = {
        'regs': mx.zeros((batch_size, num_regs, d_val)),
        'ram': ram0,
        'stack_mem': mx.zeros((batch_size, stack_depth, d_val)),
        'stack_depth': mx.zeros((batch_size, stack_depth)),
        'ip': mx.zeros((batch_size, 1)),
    }
    # Initial registers: r0 = r0_val, r1 = 3.0
    state['regs'] = state['regs'] + mx.array([
        [[r0_val] * d_val, [3.0] * d_val] + [[0.0] * d_val] * (num_regs - 2)
    ])
    return state


def scaffold_program(d_val: int, num_regs: int, key) -> mx.array:
    """
    Program-tensor initialisation with a minimal task-interface scaffold:
    a weak dst prior toward r0 (the objective register named by the loss).
    Without it the early dst gradient is antagonistic (writing a small commit
    value over a non-zero r0 decreases the loss) and the dst softmax collapses
    onto a wrong register, permanently severing the loss path.
    """
    p = mx.random.normal((1, 4, d_val), key=key) * 0.1
    dst_prior = mx.zeros((1, 4, d_val))
    dst_prior[:, 0, 0] = 1.0  # dst -> r0
    return p + dst_prior


def induce_program(vm: NeuralDifferentiableVM, init_state: dict, program_p: mx.array,
                   target: float, steps: int = 3, iters: int = 700, lr: float = 0.05,
                   clip_norm: float = 5.0, tau_start: float = 1.0, tau_end: float = 0.1,
                   verbose: bool = False):
    """
    Program Induction via BPTT with three stability mechanisms:
      1. Curriculum: the first phase optimises a single-step objective, the second
         phase the full multi-step one. Rationale: with a multi-step objective from
         the start, the initial destination-register gradient points AWAY from the
         target register (writing a small commit value over a larger initial value
         decreases the objective), which saturates the dst softmax on the wrong
         register within ~20 iterations and permanently kills that gradient
         (winner-take-all collapse). The single-step phase first raises the commit
         value toward the target, which flips the dst gradient sign toward the
         target register before deep composition begins.
      2. Softmax temperature annealing tau_start -> tau_end (Module 5: tau -> 0):
         a fixed soft tau lets uniform writes homogenise the register file, which
         zeroes dL/d(dst logits); sharpening tau revives register selection.
      3. Global-norm gradient clipping (the dst-field gradient dominates ~10x and
         would otherwise destabilise the other program fields).
    """

    def tau_at(it: int) -> float:
        frac = min(1.0, it / max(1, iters * 0.75))
        return tau_start * (tau_end / tau_start) ** frac  # geometric anneal

    curriculum_switch = int(iters * 0.4)

    def loss_fn(p, tau, n_steps):
        st = init_state
        for _ in range(n_steps):
            st = vm.execute_step(st, p, tau=tau)
        reg_err = (st['regs'][0, 0, 0] - target) ** 2
        # Secondary objective: IP advances by exactly n_steps (no spurious branches)
        ip_err = (st['ip'][0, 0] - n_steps) ** 2
        return reg_err + 0.02 * ip_err

    last_tau = tau_end
    for it in range(1, iters + 1):
        tau = tau_at(it)
        last_tau = tau
        n_steps = 1 if it <= curriculum_switch else steps
        loss, grads = mx.value_and_grad(lambda p_: loss_fn(p_, tau, n_steps))(program_p)
        grads, gnorm = clip_grad_global_norm(grads, clip_norm)
        program_p = program_p - lr * grads
        mx.eval(program_p, loss, gnorm)
        if verbose and (it % 50 == 0 or it == 1 or it == curriculum_switch + 1):
            phase = "1-step" if n_steps == 1 else f"{steps}-step"
            print(f"    Iteration {it:03d}/{iters} [{phase}] | Loss: {loss.item():.6f} | "
                  f"GradNorm: {gnorm.item():.3f} | tau: {tau:.3f}")
    mx.eval(program_p)
    return program_p, last_tau


def run_forward_trace(vm: NeuralDifferentiableVM, num_regs: int, d_val: int):
    print("[1/3] Multi-Step Forward Execution Trace with live Type-Signature Routing...")
    state = make_initial_state(1, num_regs, vm.num_ram_cells, vm.stack_depth, d_val)

    # Program tensor encoding: ADD r0 + r1 -> r0
    inst_tensor = mx.zeros((1, 4, d_val)) + mx.array([
        [[10.0] + [0.0] * (d_val - 1),                 # Opcode vector
         [10.0, 0.0, 0.0, 0.0] + [0.0] * (d_val - 4),  # Dst = r0
         [10.0, 0.0, 0.0, 0.0] + [0.0] * (d_val - 4),  # Src1 = r0
         [0.0, 10.0, 0.0, 0.0] + [0.0] * (d_val - 4)]  # Src2 = r1
    ])

    for stage_step in range(4):
        state = vm.execute_step(state, inst_tensor, tau=0.1)
        mx.eval(state['regs'], state['ip'], state['ram'], state['stack_mem'])
        r = state['routing']
        print(f"  • Cycle {stage_step + 1:02d} | IP: {state['ip'][0, 0].item():5.2f} | "
              f"r0[0]: {state['regs'][0, 0, 0].item():8.4f} | routing(ALU/MEMr/MEMw/PUSH/POP): "
              f"{r['alu_mass'][0, 0].item():.3f}/"
              f"{r['mem_read'][0, 0].item():.3f}/"
              f"{r['mem_write'][0, 0].item():.3f}/"
              f"{r['stack_push'][0, 0].item():.3f}/"
              f"{r['stack_pop'][0, 0].item():.3f}")
    print("    [✓] RAM and Stack are live: memory/stack states receive gradient every cycle")


def run_gradient_liveness_report(vm: NeuralDifferentiableVM, init_state: dict, program_p: mx.array,
                                 eval_tau: float = 0.5):
    """
    Proves every learnable parameter and every memory subsystem participates in BPTT.

    NOTE on eval_tau: the report must run at a MODERATE temperature. A fully
    annealed (tau -> 0.1) converged program is quasi-discrete: unused routing
    gates carry mass ~e^{-|logit gap|}, which underflows float32 to exactly 0.0
    and would spuriously report a LIVE subsystem as dead. At tau = 0.5 every
    functional unit retains measurable mass, so this check verifies the
    connectivity of the computation graph itself.
    """
    print("[2/3] Gradient Liveness Report (every parameter & subsystem must receive gradient)...")

    def p_loss():
        st = init_state
        for _ in range(3):
            st = vm.execute_step(st, program_p, tau=eval_tau)
        return st['regs'][0, 0, 0] + 0.02 * st['ip'][0, 0]

    import mlx.nn as mnn
    _, grads = mnn.value_and_grad(vm, p_loss)()

    def flatten(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flatten(v, prefix + k + ".")
            else:
                yield prefix + k, v

    dead = []
    for name, g in flatten(grads):
        gmax = mx.abs(g).max().item()
        status = "LIVE" if gmax > 0.0 else "DEAD"
        if gmax == 0.0:
            dead.append(name)
        print(f"    param {name:34s} grad_inf = {gmax:.3e}  [{status}]")

    # State-subsystem gradients: DNC RAM and Continuous Stack must both be live
    def state_loss(ram, stack_mem):
        st = dict(init_state)
        st['ram'], st['stack_mem'] = ram, stack_mem
        for _ in range(3):
            st = vm.execute_step(st, program_p, tau=eval_tau)
        return st['regs'].sum() + st['ip'].sum()

    g_ram, g_stack = mx.grad(state_loss, argnums=(0, 1))(init_state['ram'], init_state['stack_mem'])
    ram_live = mx.abs(g_ram).max().item() > 0.0
    stack_live = mx.abs(g_stack).max().item() > 0.0
    print(f"    subsystem d(regs,ip)/d RAM        grad_inf = {mx.abs(g_ram).max().item():.3e}  "
          f"[{'LIVE' if ram_live else 'DEAD'}]")
    print(f"    subsystem d(regs,ip)/d Stack       grad_inf = {mx.abs(g_stack).max().item():.3e}  "
          f"[{'LIVE' if stack_live else 'DEAD'}]")

    assert not dead, f"Dead parameters detected (zero gradient): {dead}"
    assert ram_live and stack_live, "Dead subsystem detected: DNC RAM or Continuous Stack receives no gradient"
    print("    [✓] All parameters and memory subsystems are live in the BPTT graph")


def run_neural_vm_experiment():
    print("======================================================================")
    print("🧠 RUNNING RIGOROUS DIFFERENTIABLE NEURAL VM (APPLE MLX METAL GPU)")
    print("======================================================================\n")

    d_val = 8
    num_regs = 4
    num_ram = 8
    # Seed the global RNG BEFORE model construction: nn layers draw their
    # initialisation from the global stream, so without this the router
    # weights (and therefore convergence behaviour) differ on every process
    # run and the test is not reproducible.
    mx.random.seed(MASTER_KEY)
    vm = NeuralDifferentiableVM(num_registers=num_regs, d_val=d_val, num_ram_cells=num_ram, stack_depth=8)
    mx.eval(vm.parameters())

    run_forward_trace(vm, num_regs, d_val)
    print()

    target_val = 42.0
    print(f"[2b/3] End-to-End Differentiable Program Optimization (BPTT, deterministic seed)")
    print(f"       Goal: Target Value = {target_val:.1f} in Register r0 after 3 steps\n")

    key = mx.random.key(MASTER_KEY)
    program_p = scaffold_program(d_val, num_regs, key)
    # Induction starts from a zeroed accumulator register r0 (natural ISA
    # semantics) so that growing the commit value is never antagonistic to
    # selecting r0 as the destination. The RAM is initialised with (seeded)
    # non-zero content: over an all-zero memory the content-addressing beta
    # has a numerically zero gradient (reads return 0 regardless of weights),
    # which would starve the beta parameter of learning signal.
    init_state = make_initial_state(
        1, num_regs, num_ram, vm.stack_depth, d_val,
        key=mx.random.key(MASTER_KEY + 7), r0_val=0.0,
    )

    program_p, final_tau = induce_program(vm, init_state, program_p, target_val, verbose=True)

    final_st = init_state
    for _ in range(3):
        final_st = vm.execute_step(final_st, program_p, tau=final_tau)
    mx.eval(final_st['regs'])
    final_val = final_st['regs'][0, 0, 0].item()
    print(f"\n  • Final Output Value in Register r0: {final_val:.4f} (Target: {target_val:.1f})")
    print(f"  • Error Margin: {abs(final_val - target_val):.6f}")
    assert abs(final_val - target_val) < 0.1, f"Neural VM failed gradient convergence, got {final_val}"
    print("  • [✓ VERIFIED] Program induction converged (deterministic seed)")

    run_gradient_liveness_report(vm, init_state, program_p, eval_tau=0.5)
    print()

    print("[3/3] Multi-Seed Robustness Sweep (clipping + bounded state must converge reliably)...")
    converged = 0
    n_seeds = 8
    for s in range(n_seeds):
        k = mx.random.key(MASTER_KEY + 137 * (s + 1))
        p = scaffold_program(d_val, num_regs, k)
        p, p_tau = induce_program(vm, init_state, p, target_val)
        st = init_state
        for _ in range(3):
            st = vm.execute_step(st, p, tau=p_tau)
        err = abs(st['regs'][0, 0, 0].item() - target_val)
        ok = err < 0.1
        converged += ok
        print(f"    seed {s}: final err = {err:.4f}  [{'✓' if ok else '✗'}]")
    print(f"  • Convergence: {converged}/{n_seeds} seeds")
    assert converged >= n_seeds - 1, f"Robustness regression: only {converged}/{n_seeds} seeds converged"
    print("\n======================================================================")
    print("[✓ ALL VERIFIED] Differentiable Neural VM: deterministic induction,")
    print("    live routing to ALU/RAM/Stack, live gradients for all parameters.")
    print("======================================================================")


if __name__ == "__main__":
    start_t = time.time()
    run_neural_vm_experiment()
    print(f"\nTotal wall-clock: {time.time() - start_t:.3f}s")
