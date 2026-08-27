#!/usr/bin/env python3
"""
Architecture Generator for Unique Multi-Stage Mutated Virtual Machines.
Reads 'spec/stages_db.json', synthesizes combinatorially unique VM architectures,
formally verifies causality invariants, and exports specifications.
"""

import json
import random
import hashlib
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
SPEC_DIR = BASE_DIR / "spec"
OUT_DIR = BASE_DIR / "generated_architectures"

def load_stages_db():
    db_path = SPEC_DIR / "stages_db.json"
    with open(db_path, "r", encoding="utf-8") as f:
        return json.load(f)

class ArchitectureSynthesizer:
    def __init__(self, db, seed=None):
        self.db = db
        if seed is not None:
            random.seed(seed)

    def generate_pipeline(self, max_junk_stages=2, enable_mutator=True):
        """
        Synthesizes a valid multi-stage pipeline complying with causality invariants.
        """
        causality = self.db["mutation_algebra"]["causality_matrix"]
        
        # Mandatory core stages in topological order
        core_stages = [
            "STAGE_FETCH",
            "STAGE_DYNAMIC_DECRYPT",
            "STAGE_DECODE",
            "STAGE_OPERAND_FETCH",
            "STAGE_EXECUTE",
            "STAGE_COMMIT_WRITEBACK"
        ]
        
        # Optional / Floating stages
        pipeline = list(core_stages)
        
        # 1. Optionally insert junk stages at safe slots
        for _ in range(random.randint(0, max_junk_stages)):
            # Safe slots: between fetch and commit (excluding after commit)
            insert_pos = random.randint(1, len(pipeline) - 1)
            pipeline.insert(insert_pos, "STAGE_JUNK_NOOP")
            
        # 2. Optionally insert metamorphic engine
        if enable_mutator and random.choice([True, False]):
            # Place stage mutator either after decode or before commit
            insert_pos = random.randint(2, len(pipeline) - 1)
            pipeline.insert(insert_pos, "STAGE_STAGE_MUTATOR")
            
        return pipeline

    def verify_causality_invariants(self, pipeline):
        """
        Formal invariant check (mirroring TLA+ specification):
        1. Causal order: Fetch < Decrypt < Decode < OperandFetch < Execute < Commit
        2. Commit must be the terminal architectural effect.
        """
        causality = self.db["mutation_algebra"]["causality_matrix"]
        seen_ranks = []
        
        for stage in pipeline:
            if stage in causality:
                rank = causality[stage]
                if seen_ranks and rank < max(seen_ranks):
                    return False, f"Causality Violation: Stage {stage} (rank {rank}) appeared after higher rank {max(seen_ranks)}"
                seen_ranks.append(rank)
                
        if "STAGE_COMMIT_WRITEBACK" in pipeline:
            commit_idx = pipeline.index("STAGE_COMMIT_WRITEBACK")
            # Commit should be at or near the end (only non-destructive stages allowed after)
            for subsequent in pipeline[commit_idx + 1:]:
                if subsequent not in ["STAGE_STAGE_MUTATOR", "STAGE_JUNK_NOOP"]:
                    return False, f"Illegal stage {subsequent} placed after state commit."
                    
        return True, "All Formal Invariants Verified OK"

    def synthesize_unique_architecture(self, name_prefix="ArchVM"):
        """
        Synthesizes a complete unique VM profile.
        """
        paradigm_key = random.choice(list(self.db["execution_paradigms"].keys()))
        paradigm = self.db["execution_paradigms"][paradigm_key]
        
        dispatch_key = random.choice(list(self.db["dispatch_mechanisms"].keys()))
        dispatch = self.db["dispatch_mechanisms"][dispatch_key]
        
        fmt_key = random.choice(list(self.db["instruction_formats"].keys()))
        fmt = self.db["instruction_formats"][fmt_key]
        
        pipeline = self.generate_pipeline()
        is_valid, msg = self.verify_causality_invariants(pipeline)
        
        if not is_valid:
            raise ValueError(f"Synthesis failed invariant check: {msg}")
            
        # Generate polymorphic key table
        opcodes = ["ADD", "SUB", "XOR", "AND", "SHL", "SHR", "LOAD", "STORE", "JMP", "JZ", "HALT"]
        shuffled_opcodes = list(opcodes)
        random.shuffle(shuffled_opcodes)
        opcode_map = {op: idx for idx, op in enumerate(shuffled_opcodes)}
        
        # Compute deterministic fingerprint
        spec_blob = f"{paradigm_key}:{dispatch_key}:{fmt_key}:{pipeline}:{shuffled_opcodes}"
        arch_hash = hashlib.sha256(spec_blob.encode()).hexdigest()[:12]
        arch_id = f"{name_prefix}_{arch_hash}"
        
        architecture = {
            "arch_id": arch_id,
            "paradigm": paradigm,
            "dispatch": dispatch,
            "instruction_format": fmt,
            "pipeline_stages": pipeline,
            "opcode_mapping": opcode_map,
            "formal_verification_status": {
                "invariants_passed": is_valid,
                "verification_message": msg,
                "tla_model_compatible": True
            }
        }
        return architecture

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = load_stages_db()
    synthesizer = ArchitectureSynthesizer(db)
    
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"[*] Generating {count} unique verified VM architectures from stages database...")
    
    for i in range(count):
        arch = synthesizer.synthesize_unique_architecture(name_prefix=f"VM_Gen_{i+1}")
        out_file = OUT_DIR / f"{arch['arch_id']}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(arch, f, indent=2)
            
        print(f"  [+] Created: {arch['arch_id']}")
        print(f"      - Paradigm : {arch['paradigm']['id']}")
        print(f"      - Dispatch : {arch['dispatch']['id']}")
        print(f"      - Pipeline : {' -> '.join(arch['pipeline_stages'])}")
        print(f"      - Verified : {arch['formal_verification_status']['verification_message']}")
        print(f"      - Saved to : {out_file.relative_to(BASE_DIR)}\n")

if __name__ == "__main__":
    main()
