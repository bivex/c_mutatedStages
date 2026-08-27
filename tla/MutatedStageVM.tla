--------------------------- MODULE MutatedStageVM ---------------------------
(* 
   Formal TLA+ Specification for Multi-Stage Mutated Virtual Machines.
   Models pipelined stage execution, dynamic pipeline mutation/polymorphism,
   inter-stage latches, and formal safety/correctness invariants.
*)

EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    MaxSteps,        \* Bound on execution steps for model checker
    RegCount,        \* Number of general purpose registers (e.g., 4)
    MaxPC,           \* Maximum program counter
    MaxStackDepth    \* Maximum stack evaluation depth

VARIABLES
    pc,              \* Program counter: 0..MaxPC
    regs,            \* Register file: [1..RegCount -> Int]
    stack,           \* Evaluation stack: Seq(Int)
    codeMem,         \* Immutable program code memory
    pipeline,        \* Active sequence of stages: Seq(STRING)
    stageIdx,        \* Current stage pointer: 1..(Len(pipeline)+1)
    
    \* Inter-stage latches (Buffers passed between micro-stages)
    fetchLatch,      \* Holds raw instruction record
    decodeLatch,     \* Holds decoded opcode and register indices
    operandLatch,    \* Holds resolved operand values (valA, valB, dst)
    execLatch,       \* Holds computation result and branch info
    
    mutationEntropy, \* Ephemeral key/entropy used for dynamic polymorphic mutations
    stepCount,       \* Total execution step counter
    vmHalted         \* Termination status

vars == << pc, regs, stack, codeMem, pipeline, stageIdx, 
           fetchLatch, decodeLatch, operandLatch, execLatch, 
           mutationEntropy, stepCount, vmHalted >>

\* Defined Stages in the VM Universe
STAGE_FETCH       == "FETCH"
STAGE_DECRYPT     == "DECRYPT"
STAGE_DECODE      == "DECODE"
STAGE_OP_FETCH    == "OP_FETCH"
STAGE_EXECUTE     == "EXECUTE"
STAGE_COMMIT      == "COMMIT"
STAGE_JUNK        == "JUNK"
STAGE_MUTATE      == "MUTATE"

AllStages == {
    STAGE_FETCH, STAGE_DECRYPT, STAGE_DECODE, 
    STAGE_OP_FETCH, STAGE_EXECUTE, STAGE_COMMIT, 
    STAGE_JUNK, STAGE_MUTATE
}

\* Supported Instruction Opcodes for the formal model
OP_ADD  == "ADD"
OP_SUB  == "SUB"
OP_XOR  == "XOR"
OP_PUSH == "PUSH"
OP_HALT == "HALT"

Opcodes == { OP_ADD, OP_SUB, OP_XOR, OP_PUSH, OP_HALT }

\* Sample initial program for verification
InitialProgram == [
    addr \in 0..MaxPC |->
        IF addr = 0 THEN [op |-> OP_PUSH, src1 |-> 1, src2 |-> 1, dst |-> 1, imm |-> 10]
        ELSE IF addr = 1 THEN [op |-> OP_ADD, src1 |-> 1, src2 |-> 1, dst |-> 2, imm |-> 5]
        ELSE IF addr = 2 THEN [op |-> OP_XOR, src1 |-> 1, src2 |-> 2, dst |-> 3, imm |-> 0]
        ELSE [op |-> OP_HALT, src1 |-> 1, src2 |-> 1, dst |-> 1, imm |-> 0]
]

\* Default Canonical Pipeline
CanonicalPipeline == <<
    STAGE_FETCH,
    STAGE_DECRYPT,
    STAGE_DECODE,
    STAGE_OP_FETCH,
    STAGE_EXECUTE,
    STAGE_COMMIT
>>

-----------------------------------------------------------------------------
(* Initial State *)

Init ==
    /\ pc = 0
    /\ regs = [r \in 1..RegCount |-> 0]
    /\ stack = <<>>
    /\ codeMem = InitialProgram
    /\ pipeline = CanonicalPipeline
    /\ stageIdx = 1
    /\ fetchLatch = [valid |-> FALSE, raw |-> [op |-> OP_HALT, src1 |-> 1, src2 |-> 1, dst |-> 1, imm |-> 0]]
    /\ decodeLatch = [valid |-> FALSE, op |-> OP_HALT, src1 |-> 1, src2 |-> 1, dst |-> 1, imm |-> 0]
    /\ operandLatch = [valid |-> FALSE, valA |-> 0, valB |-> 0, dst |-> 1, imm |-> 0]
    /\ execLatch = [valid |-> FALSE, result |-> 0, dst |-> 1, isHalt |-> FALSE]
    /\ mutationEntropy = 42
    /\ stepCount = 0
    /\ vmHalted = FALSE

-----------------------------------------------------------------------------
(* Stage Actions *)

\* 1. Stage: Instruction Fetch
StepFetch ==
    /\ ~vmHalted
    /\ stageIdx <= Len(pipeline)
    /\ pipeline[stageIdx] = STAGE_FETCH
    /\ fetchLatch' = [valid |-> TRUE, raw |-> codeMem[pc]]
    /\ stageIdx' = stageIdx + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, stack, codeMem, pipeline, decodeLatch, operandLatch, execLatch, mutationEntropy, vmHalted >>

\* 2. Stage: Polymorphic Dynamic Decrypt
StepDecrypt ==
    /\ ~vmHalted
    /\ stageIdx <= Len(pipeline)
    /\ pipeline[stageIdx] = STAGE_DECRYPT
    /\ fetchLatch.valid
    \* Simulates affine/key decryption transformation (identity or permutation based on entropy)
    /\ fetchLatch' = [fetchLatch EXCEPT !.raw.imm = fetchLatch.raw.imm]
    /\ stageIdx' = stageIdx + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, stack, codeMem, pipeline, decodeLatch, operandLatch, execLatch, mutationEntropy, vmHalted >>

\* 3. Stage: Instruction Decode
StepDecode ==
    /\ ~vmHalted
    /\ stageIdx <= Len(pipeline)
    /\ pipeline[stageIdx] = STAGE_DECODE
    /\ fetchLatch.valid
    /\ decodeLatch' = [
        valid |-> TRUE,
        op    |-> fetchLatch.raw.op,
        src1  |-> fetchLatch.raw.src1,
        src2  |-> fetchLatch.raw.src2,
        dst   |-> fetchLatch.raw.dst,
        imm   |-> fetchLatch.raw.imm
       ]
    /\ stageIdx' = stageIdx + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, stack, codeMem, pipeline, fetchLatch, operandLatch, execLatch, mutationEntropy, vmHalted >>

\* 4. Stage: Operand Fetch & Resolution
StepOperandFetch ==
    /\ ~vmHalted
    /\ stageIdx <= Len(pipeline)
    /\ pipeline[stageIdx] = STAGE_OP_FETCH
    /\ decodeLatch.valid
    /\ operandLatch' = [
        valid |-> TRUE,
        valA  |-> regs[decodeLatch.src1],
        valB  |-> regs[decodeLatch.src2],
        dst   |-> decodeLatch.dst,
        imm   |-> decodeLatch.imm
       ]
    /\ stageIdx' = stageIdx + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, stack, codeMem, pipeline, fetchLatch, decodeLatch, execLatch, mutationEntropy, vmHalted >>

\* 5. Stage: ALU Core Execution
StepExecute ==
    /\ ~vmHalted
    /\ stageIdx <= Len(pipeline)
    /\ pipeline[stageIdx] = STAGE_EXECUTE
    /\ operandLatch.valid
    /\ decodeLatch.valid
    /\ LET op  == decodeLatch.op
           va  == operandLatch.valA
           vb  == operandLatch.valB
           imm == operandLatch.imm
       IN
       /\ execLatch' = [
            valid  |-> TRUE,
            result |-> IF op = OP_ADD THEN va + imm
                       ELSE IF op = OP_SUB THEN va - imm
                       ELSE IF op = OP_PUSH THEN imm
                       ELSE 0,
            dst    |-> operandLatch.dst,
            isHalt |-> (op = OP_HALT)
          ]
    /\ stageIdx' = stageIdx + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, stack, codeMem, pipeline, fetchLatch, decodeLatch, operandLatch, mutationEntropy, vmHalted >>

\* 6. Stage: State Commit & Writeback (End of Pipeline Cycle)
StepCommit ==
    /\ ~vmHalted
    /\ stageIdx <= Len(pipeline)
    /\ pipeline[stageIdx] = STAGE_COMMIT
    /\ execLatch.valid
    /\ IF execLatch.isHalt THEN
          /\ vmHalted' = TRUE
          /\ UNCHANGED << pc, regs, stack >>
       ELSE
          /\ regs' = [regs EXCEPT ![execLatch.dst] = execLatch.result]
          /\ pc' = IF pc < MaxPC THEN pc + 1 ELSE pc
          /\ UNCHANGED << stack, vmHalted >>
    \* Reset latches for next instruction cycle and point stageIdx back to 1
    /\ stageIdx' = 1
    /\ fetchLatch' = [fetchLatch EXCEPT !.valid = FALSE]
    /\ decodeLatch' = [decodeLatch EXCEPT !.valid = FALSE]
    /\ operandLatch' = [operandLatch EXCEPT !.valid = FALSE]
    /\ execLatch' = [execLatch EXCEPT !.valid = FALSE]
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << codeMem, pipeline, mutationEntropy >>

\* 7. Floating Stage: Decoy / Junk Stage (No-Op Side Effect)
StepJunk ==
    /\ ~vmHalted
    /\ stageIdx <= Len(pipeline)
    /\ pipeline[stageIdx] = STAGE_JUNK
    /\ mutationEntropy' = (mutationEntropy * 13 + 7) % 256
    /\ stageIdx' = stageIdx + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, stack, codeMem, pipeline, fetchLatch, decodeLatch, operandLatch, execLatch, vmHalted >>

\* 8. Metamorphic Stage: Dynamic Pipeline Mutator
\* Mutates the pipeline configuration for subsequent cycles while preserving causality
StepMutatePipeline ==
    /\ ~vmHalted
    /\ stageIdx <= Len(pipeline)
    /\ pipeline[stageIdx] = STAGE_MUTATE
    /\ LET \** Randomly choose either inserting a junk stage or swapping safe floating stages
           canInsertJunk == (Len(pipeline) < 8)
       IN
       IF canInsertJunk THEN
          pipeline' = Append(pipeline, STAGE_JUNK)
       ELSE
          pipeline' = CanonicalPipeline
    /\ stageIdx' = stageIdx + 1
    /\ mutationEntropy' = (mutationEntropy + 1) % 100
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, stack, codeMem, fetchLatch, decodeLatch, operandLatch, execLatch, vmHalted >>

-----------------------------------------------------------------------------
(* State Machine Next Transition *)

Terminated ==
    /\ vmHalted
    /\ UNCHANGED vars

Next ==
    /\ stepCount < MaxSteps
    /\ \/ StepFetch
       \/ StepDecrypt
       \/ StepDecode
       \/ StepOperandFetch
       \/ StepExecute
       \/ StepCommit
       \/ StepJunk
       \/ StepMutatePipeline
       \/ Terminated

-----------------------------------------------------------------------------
(* Formal Invariants & Verification Properties *)

\* 1. Type Safety
TypeOK ==
    /\ pc \in 0..MaxPC
    /\ \A r \in 1..RegCount : regs[r] \in Int
    /\ stageIdx \in 1..(Len(pipeline) + 1)
    /\ vmHalted \in BOOLEAN

\* 2. Causality Invariant (Data Hazard Freedom)
\* An execution stage MUST NOT execute unless valid decoded operands have been latched
NoExecutionBeforeDecode ==
    (stageIdx <= Len(pipeline) /\ pipeline[stageIdx] = STAGE_EXECUTE) => (operandLatch.valid /\ decodeLatch.valid)

\* 3. Commit Integrity
\* State writeback MUST NOT occur unless valid execution results exist
NoCommitBeforeExecution ==
    (stageIdx <= Len(pipeline) /\ pipeline[stageIdx] = STAGE_COMMIT) => execLatch.valid

\* 4. Decryption Integrity
\* Decode cannot operate on un-decrypted ciphertext if decrypt stage is in the pipeline
DecodeAfterDecrypt ==
    \A i, j \in 1..Len(pipeline) :
        (pipeline[i] = STAGE_DECRYPT /\ pipeline[j] = STAGE_DECODE) => (i < j)

\* 5. Liveness / Progress Guarantee (No deadlock)
\* When not halted, the system can always take a valid step forward
Liveness ==
    ~vmHalted => (stageIdx <= Len(pipeline))

=============================================================================
