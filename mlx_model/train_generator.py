#!/usr/bin/env python3
"""
Training script for MLX VMPipelineTransformer on Apple Silicon GPU.
Synthesizes valid 100+ stage VM architectural pipelines,
trains the causal transformer using AdamW and cross-entropy loss,
and exports model weights.
"""

import random
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from pathlib import Path

from vm_transformer import VMPipelineTransformer, VMVocabulary

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_WEIGHTS_PATH = BASE_DIR / "mlx_model" / "vm_transformer_weights.npz"

def generate_synthetic_100_stage_pipeline(vocab: VMVocabulary, target_len: int = 110):
    """
    Synthesizes a valid topological ground-truth pipeline with 100+ stages.
    """
    stages = [vocab.bos_id]
    
    # 1. Frontend: Fetch & Decryption (3-5 stages)
    stages.append(vocab.token2id[random.choice(["STAGE_FETCH_DIRECT", "STAGE_FETCH_PREFETCH_4B", "STAGE_FETCH_XOR_DECRYPT"])])
    stages.append(vocab.token2id[random.choice(["STAGE_DECRYPT_AFFINE", "STAGE_DECRYPT_LOOKUP_PERM", "STAGE_DECRYPT_KEY_ROTATION"])])
    
    # 2. Decode stages (2-4 stages)
    stages.append(vocab.token2id[random.choice(["STAGE_DECODE_BITMASK_RV", "STAGE_DECODE_HUFFMAN", "STAGE_DECODE_VARBYTE_LEB"])])
    
    # 3. Operand resolution (3-5 stages)
    stages.append(vocab.token2id["STAGE_OPERAND_REG_READ_RS1"])
    stages.append(vocab.token2id["STAGE_OPERAND_REG_READ_RS2"])
    stages.append(vocab.token2id["STAGE_OPERAND_IMM_SIGN_EXTEND"])
    
    # 4. Long metamorphic body: Interleave P-Code, Latches, Junk, and Mutators until length >= target_len - 10
    body_pool = (
        vocab.PCODE_MICRO_STAGES +
        vocab.FORWARDING_AND_JUNK_STAGES +
        vocab.METAMORPHIC_STAGES
    )
    
    while len(stages) < target_len - 6:
        chosen = random.choice(body_pool)
        stages.append(vocab.token2id[chosen])
        
    # 5. Core Execution & MBA (2-4 stages)
    stages.append(vocab.token2id[random.choice(["STAGE_EXEC_ALU_ADD", "STAGE_EXEC_ALU_XOR", "STAGE_EXEC_MBA_LINEAR", "STAGE_EXEC_MBA_NONLINEAR"])])
    
    # 6. Dynamic Mutator (1-2 stages)
    stages.append(vocab.token2id[random.choice(vocab.METAMORPHIC_STAGES)])
    
    # 7. Commit & Writeback (3 stages)
    stages.append(vocab.token2id["STAGE_COMMIT_REG_WRITE_RD"])
    stages.append(vocab.token2id["STAGE_COMMIT_PC_ADVANCE"])
    stages.append(vocab.token2id["STAGE_COMMIT_FLUSH_LATCHES"])
    stages.append(vocab.eos_id)
    
    return stages

def build_dataset(vocab: VMVocabulary, num_samples: int = 500, max_len: int = 128):
    data = []
    for _ in range(num_samples):
        seq = generate_synthetic_100_stage_pipeline(vocab, target_len=random.randint(102, 115))
        # Pad to max_len
        if len(seq) < max_len:
            seq = seq + [vocab.pad_id] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        data.append(seq)
    return mx.array(data)

def loss_fn(model, x, y, pad_id):
    logits = model(x)
    # Cross entropy loss ignoring pad tokens
    loss = nn.losses.cross_entropy(logits, y, reduction="none")
    mask = (y != pad_id).astype(mx.float32)
    return (loss * mask).sum() / mask.sum()

def train():
    vocab = VMVocabulary()
    print(f"[*] Initializing MLX VMPipelineTransformer (Vocab Size: {vocab.vocab_size})...")
    
    model = VMPipelineTransformer(
        vocab_size=vocab.vocab_size,
        max_seq_len=160,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512
    )
    mx.eval(model.parameters())
    
    optimizer = optim.AdamW(learning_rate=3e-3, weight_decay=0.01)
    
    print("[*] Generating synthetic dataset of 100+ stage topological VM pipelines...")
    data = build_dataset(vocab, num_samples=300, max_len=128)
    
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    
    print(f"[*] Training MLX Neural Network on Apple Silicon GPU (Batch Size: 32, Epochs: 25)...")
    batch_size = 32
    num_batches = data.shape[0] // batch_size
    
    start_time = time.time()
    for epoch in range(1, 26):
        epoch_loss = 0.0
        # Shuffle indices
        indices = mx.array(random.sample(range(data.shape[0]), data.shape[0]))
        shuffled_data = data[indices]
        
        for b in range(num_batches):
            batch = shuffled_data[b * batch_size : (b + 1) * batch_size]
            x = batch[:, :-1]
            y = batch[:, 1:]
            
            loss, grads = loss_and_grad_fn(model, x, y, vocab.pad_id)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / num_batches
        if epoch % 5 == 0 or epoch == 1:
            print(f"  • Epoch {epoch:2d}/25 | Cross-Entropy Loss: {avg_loss:.4f}")
            
    elapsed = time.time() - start_time
    print(f"[✓] Training completed in {elapsed:.2f}s!")
    
    # Flatten model parameters for export to npz
    flat_weights = {}
    def extract_weights(prefix, d):
        for k, v in d.items():
            full_k = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                extract_weights(full_k, v)
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        extract_weights(f"{full_k}.{i}", item)
                    else:
                        flat_weights[f"{full_k}.{i}"] = item
            else:
                flat_weights[full_k] = v
                
    extract_weights("", model.parameters())
    mx.savez(str(MODEL_WEIGHTS_PATH), **flat_weights)
    print(f"[✓] Saved trained MLX model weights to {MODEL_WEIGHTS_PATH.relative_to(BASE_DIR)}")

if __name__ == "__main__":
    train()
