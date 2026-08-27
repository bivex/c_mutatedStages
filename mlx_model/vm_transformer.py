#!/usr/bin/env python3
"""
Apple MLX Neural Network for Multi-Stage VM Architecture Generation.
Implements a Causal Transformer Decoder with Dynamic Topological Masking
to autoregressively synthesize 100+ stage virtual machine architectures.
"""

import math
import mlx.core as mx
import mlx.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x


class VMPipelineTransformer(nn.Module):
    """
    Autoregressive AI Model for Synthesizing 100+ Stage VM Pipelines.
    """
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        B, L = tokens.shape
        pos = mx.arange(L)
        
        x = self.token_emb(tokens) + self.pos_emb(pos)
        
        # Causal triangular attention mask
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class VMVocabulary:
    """
    Vocab mapping of over 100+ micro-stage primitives, Ghidra P-Code ops,
    obfuscation hooks, and metamorphic mutator layers.
    """
    SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>"]
    
    CORE_STAGES = [
        "STAGE_FETCH_DIRECT",
        "STAGE_FETCH_PREFETCH_4B",
        "STAGE_FETCH_XOR_DECRYPT",
        "STAGE_DECRYPT_AFFINE",
        "STAGE_DECRYPT_LOOKUP_PERM",
        "STAGE_DECRYPT_KEY_ROTATION",
        "STAGE_DECODE_BITMASK_RV",
        "STAGE_DECODE_HUFFMAN",
        "STAGE_DECODE_VARBYTE_LEB",
        "STAGE_DECODE_VLIW_SLOT",
        "STAGE_OPERAND_REG_READ_RS1",
        "STAGE_OPERAND_REG_READ_RS2",
        "STAGE_OPERAND_STACK_POP_A",
        "STAGE_OPERAND_STACK_POP_B",
        "STAGE_OPERAND_IMM_SIGN_EXTEND",
        "STAGE_OPERAND_MEM_INDIRECT",
        "STAGE_EXEC_ALU_ADD",
        "STAGE_EXEC_ALU_SUB",
        "STAGE_EXEC_ALU_XOR",
        "STAGE_EXEC_ALU_AND",
        "STAGE_EXEC_ALU_OR",
        "STAGE_EXEC_ALU_SHL",
        "STAGE_EXEC_ALU_SHR",
        "STAGE_EXEC_MBA_LINEAR",
        "STAGE_EXEC_MBA_NONLINEAR",
        "STAGE_EXEC_BRANCH_COND",
        "STAGE_EXEC_BRANCH_INDIR",
        "STAGE_COMMIT_REG_WRITE_RD",
        "STAGE_COMMIT_STACK_PUSH",
        "STAGE_COMMIT_PC_ADVANCE",
        "STAGE_COMMIT_MEM_STORE",
        "STAGE_COMMIT_FLUSH_LATCHES"
    ]
    
    # Ghidra P-Code micro-stages
    PCODE_MICRO_STAGES = [
        "STAGE_PCODE_INT_CARRY",
        "STAGE_PCODE_INT_SCARRY",
        "STAGE_PCODE_INT_SBORROW",
        "STAGE_PCODE_INT_2COMP",
        "STAGE_PCODE_INT_NEGATE",
        "STAGE_PCODE_INT_SRIGHT",
        "STAGE_PCODE_INT_MULT_HI",
        "STAGE_PCODE_INT_MULT_LO",
        "STAGE_PCODE_INT_DIV",
        "STAGE_PCODE_INT_REM",
        "STAGE_PCODE_PIECE_MERGE",
        "STAGE_PCODE_SUBPIECE_SPLIT",
        "STAGE_PCODE_INT_ZEXT32",
        "STAGE_PCODE_INT_SEXT64",
        "STAGE_PCODE_BOOL_EQUAL",
        "STAGE_PCODE_BOOL_SLESS",
        "STAGE_PCODE_CALL_SUB_DISPATCH"
    ]
    
    # Metamorphic & Obfuscation Stages
    METAMORPHIC_STAGES = [
        "STAGE_MUTATOR_PERMUTE_TABLE",
        "STAGE_MUTATOR_DYNAMIC_KEY_XOR",
        "STAGE_MUTATOR_SWAP_SHADOW_REGS",
        "STAGE_MUTATOR_JIT_STUB_PATCH",
        "STAGE_MUTATOR_ENTROPY_CLOCK",
        "STAGE_MUTATOR_SPLIT_PIPELINE"
    ]
    
    # Latency, Inter-stage Latches, Hardware forwarding & Decoy junk units
    FORWARDING_AND_JUNK_STAGES = [
        f"STAGE_FORWARDING_LATCH_L{i}" for i in range(1, 25)
    ] + [
        f"STAGE_JUNK_ALU_PADDING_{i}" for i in range(1, 30)
    ] + [
        f"STAGE_DECOY_ENTROPY_CYCLE_{i}" for i in range(1, 25)
    ]

    def __init__(self):
        self.all_stages = (
            self.SPECIAL_TOKENS +
            self.CORE_STAGES +
            self.PCODE_MICRO_STAGES +
            self.METAMORPHIC_STAGES +
            self.FORWARDING_AND_JUNK_STAGES
        )
        self.token2id = {t: i for i, t in enumerate(self.all_stages)}
        self.id2token = {i: t for i, t in enumerate(self.all_stages)}

    @property
    def vocab_size(self):
        return len(self.all_stages)

    @property
    def pad_id(self):
        return self.token2id["<PAD>"]

    @property
    def bos_id(self):
        return self.token2id["<BOS>"]

    @property
    def eos_id(self):
        return self.token2id["<EOS>"]
