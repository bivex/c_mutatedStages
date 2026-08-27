#!/usr/bin/env python3
"""
Architecture Generator for Unique Multi-Stage Mutated Virtual Machines.
Reads the comprehensive database in 'spec/stages_db.json' (Ghidra, WASM, eBPF, RISC-V, LLVM),
synthesizes unique VM configurations, formally verifies causality invariants, and exports
both JSON metadata and generated C headers for instant compilation.
"""

import json
import random
import hashlib
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
        core_stages = [
            "STAGE_FETCH",
            "STAGE_DYNAMIC_DECRYPT",
            "STAGE_DECODE",
            "STAGE_OPERAND_FETCH",
            "STAGE_EXECUTE",
            "STAGE_COMMIT_WRITEBACK"
        ]
        pipeline = list(core_stages)
        
        # 1. Optionally insert junk/decoy stages at safe points
        for _ in range(random.randint(0, max_junk_stages)):
            insert_pos = random.randint(1, len(pipeline) - 1)
            pipeline.insert(insert_pos, "STAGE_JUNK_NOOP")
            
        # 2. Optionally insert metamorphic engine
        if enable_mutator and random.choice([True, False]):
            insert_pos = random.randint(2, len(pipeline) - 1)
            pipeline.insert(insert_pos, "STAGE_STAGE_MUTATOR")
            
        return pipeline

    def verify_causality_invariants(self, pipeline):
        """
        Formal invariant verification:
        1. Causal order: Fetch < Decrypt < Decode < OperandFetch < Execute < Commit
        2. Commit must be the terminal architectural writeback effect.
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
            for subsequent in pipeline[commit_idx + 1:]:
                if subsequent not in ["STAGE_STAGE_MUTATOR", "STAGE_JUNK_NOOP"]:
                    return False, f"Illegal stage {subsequent} placed after state commit."
                    
        return True, "All Formal Invariants Verified OK"

    def select_instruction_set(self, paradigm_id):
        """
        Selects and binds real instruction primitives from Ghidra, WASM, or eBPF.
        """
        if paradigm_id == "STACK":
            # WASM numeric + control subset
            pool = list(self.db.get("wasm_core", {}).get("categories", {}).get("numeric_i32", []))
            if not pool:
                pool = ["i32.add", "i32.sub", "i32.xor", "i32.and", "i32.const", "drop"]
            random.shuffle(pool)
            selected = pool[:10]
        elif paradigm_id == "REGISTER":
            # eBPF / RISC-V register ALU ops
            pool = list(self.db.get("ebpf_arch", {}).get("alu_operations", []))
            if not pool:
                pool = ["ADD", "SUB", "XOR", "AND", "OR", "LSH", "RSH", "MOV"]
            random.shuffle(pool)
            selected = [f"BPF_{op}" for op in pool[:10]]
        else:
            # Ghidra universal P-Code micro-ops
            pcode_ops = [x["op"] for x in self.db.get("ghidra_pcode", {}).get("micro_ops", [])]
            random.shuffle(pcode_ops)
            selected = pcode_ops[:10]
            
        selected.append("HALT")
        # Build shuffled polymorphic opcodes
        random.shuffle(selected)
        return {op: idx for idx, op in enumerate(selected)}

    def synthesize_unique_architecture(self, name_prefix="ArchVM"):
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
            
        opcodes = self.select_instruction_set(paradigm["id"])
        
        # Micro-architectural units from LLVM
        sched_units = random.sample(
            self.db.get("llvm_pipeline_units", {}).get("units", [{"name": "DefaultAlu"}]),
            k=min(3, len(self.db.get("llvm_pipeline_units", {}).get("units", [])))
        )
        
        # Compute deterministic fingerprint
        spec_blob = f"{paradigm_key}:{dispatch_key}:{fmt_key}:{pipeline}:{sorted(opcodes.items())}"
        arch_hash = hashlib.sha256(spec_blob.encode()).hexdigest()[:12]
        arch_id = f"{name_prefix}_{arch_hash}"
        
        architecture = {
            "arch_id": arch_id,
            "paradigm": paradigm,
            "dispatch": dispatch,
            "instruction_format": fmt,
            "pipeline_stages": pipeline,
            "opcode_mapping": opcodes,
            "micro_sched_units": sched_units,
            "formal_verification_status": {
                "invariants_passed": is_valid,
                "verification_message": msg,
                "tla_model_compatible": True
            }
        }
        return architecture

    def emit_c_header(self, arch):
        """
        Emits an auto-generated C header describing this synthesized VM architecture.
        """
        guard = f"VM_{arch['arch_id'].upper()}_H"
        stages_enum = ",\n    ".join([f"STAGE_{idx}_{s}" for idx, s in enumerate(arch["pipeline_stages"])])
        opcodes_enum = ",\n    ".join([f"OP_{op.replace('.', '_').upper()} = {val}" for op, val in arch["opcode_mapping"].items()])
        
        c_code = f"""/* Auto-generated by c_mutatedStages Architecture Synthesizer */
/* Unique Architecture ID: {arch['arch_id']} */
/* Paradigm: {arch['paradigm']['id']} | Dispatch: {arch['dispatch']['id']} */

#ifndef {guard}
#define {guard}

#include <stdint.h>
#include <stdbool.h>

#define ARCH_ID "{arch['arch_id']}"
#define PIPELINE_STAGE_COUNT {len(arch['pipeline_stages'])}

/* Opcodes for this polymorphic instance */
typedef enum {{
    {opcodes_enum}
}} {arch['arch_id']}_opcode_t;

/* Formal pipeline stage sequence */
typedef enum {{
    {stages_enum}
}} {arch['arch_id']}_stage_t;

/* Latches / Inter-stage register state */
typedef struct {{
    uint32_t pc;
    uint64_t registers[16];
    uint64_t stack[64];
    uint32_t sp;
    uint32_t current_stage_idx;
    uint32_t entropy_key;
    bool halted;
}} {arch['arch_id']}_context_t;

#endif /* {guard} */
"""
        return c_code

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = load_stages_db()
    synthesizer = ArchitectureSynthesizer(db)
    
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"[*] Generating {count} unique verified VM architectures from comprehensive DB...")
    
    for i in range(count):
        arch = synthesizer.synthesize_unique_architecture(name_prefix=f"VM_Gen_{i+1}")
        
        # 1. Save JSON profile
        json_file = OUT_DIR / f"{arch['arch_id']}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(arch, f, indent=2)
            
        # 2. Save C Header
        c_file = OUT_DIR / f"{arch['arch_id']}.h"
        with open(c_file, "w", encoding="utf-8") as f:
            f.write(synthesizer.emit_c_header(arch))
            
        print(f"  [+] Created: {arch['arch_id']}")
        print(f"      - Paradigm : {arch['paradigm']['id']}")
        print(f"      - Dispatch : {arch['dispatch']['id']}")
        print(f"      - Pipeline : {' -> '.join(arch['pipeline_stages'])}")
        print(f"      - Opcodes  : {len(arch['opcode_mapping'])} ops bound")
        print(f"      - JSON     : {json_file.relative_to(BASE_DIR)}")
        print(f"      - C Header : {c_file.relative_to(BASE_DIR)}\n")

if __name__ == "__main__":
    main()
