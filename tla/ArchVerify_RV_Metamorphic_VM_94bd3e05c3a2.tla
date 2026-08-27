--------------------------- MODULE ArchVerify_RV_Metamorphic_VM_94bd3e05c3a2 ---------------------------
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS MaxSteps, RegCount, MaxPC

VARIABLES pc, regs, pipeline, stageIdx, fetchLatch, decodeLatch, operandLatch, execLatch, mutationEntropy, stepCount, vmHalted

vars == << pc, regs, pipeline, stageIdx, fetchLatch, decodeLatch, operandLatch, execLatch, mutationEntropy, stepCount, vmHalted >>

PipelineDef == << "FETCH", "DYNAMIC_DECRYPT", "DECODE", "OPERAND_FETCH", "EXECUTE", "MUTATOR", "COMMIT_WRITEBACK" >>

Init ==
    /\ pc = 0
    /\ regs = [r \in 1..RegCount |-> 0]
    /\ pipeline = PipelineDef
    /\ stageIdx = 1
    /\ fetchLatch = [valid |-> FALSE, raw_imm |-> 10]
    /\ decodeLatch = [valid |-> FALSE, imm |-> 10, dst |-> 1]
    /\ operandLatch = [valid |-> FALSE, valA |-> 0, valB |-> 0, dst |-> 1, imm |-> 10]
    /\ execLatch = [valid |-> FALSE, result |-> 10, dst |-> 1, isHalt |-> FALSE]
    /\ mutationEntropy = 42
    /\ stepCount = 0
    /\ vmHalted = FALSE

StepGeneric ==
    /\ ~vmHalted
    /\ stageIdx <= Len(pipeline)
    /\ LET currentStage == pipeline[stageIdx] IN
       /\ IF currentStage = "FETCH" THEN
             /\ fetchLatch' = [fetchLatch EXCEPT !.valid = TRUE]
             /\ UNCHANGED << decodeLatch, operandLatch, execLatch >>
          ELSE IF currentStage = "DYNAMIC_DECRYPT" THEN
             /\ fetchLatch' = [fetchLatch EXCEPT !.raw_imm = fetchLatch.raw_imm]
             /\ UNCHANGED << decodeLatch, operandLatch, execLatch >>
          ELSE IF currentStage = "DECODE" THEN
             /\ decodeLatch' = [decodeLatch EXCEPT !.valid = TRUE]
             /\ UNCHANGED << fetchLatch, operandLatch, execLatch >>
          ELSE IF currentStage = "OPERAND_FETCH" THEN
             /\ operandLatch' = [operandLatch EXCEPT !.valid = TRUE]
             /\ UNCHANGED << fetchLatch, decodeLatch, execLatch >>
          ELSE IF currentStage = "EXECUTE" THEN
             /\ execLatch' = [execLatch EXCEPT !.valid = TRUE]
             /\ UNCHANGED << fetchLatch, decodeLatch, operandLatch >>
          ELSE IF currentStage = "COMMIT_WRITEBACK" THEN
             /\ fetchLatch' = [fetchLatch EXCEPT !.valid = FALSE]
             /\ decodeLatch' = [decodeLatch EXCEPT !.valid = FALSE]
             /\ operandLatch' = [operandLatch EXCEPT !.valid = FALSE]
             /\ execLatch' = [execLatch EXCEPT !.valid = FALSE]
          ELSE
             UNCHANGED << fetchLatch, decodeLatch, operandLatch, execLatch >>
       /\ IF currentStage = "COMMIT_WRITEBACK" THEN
             /\ regs' = [regs EXCEPT ![execLatch.dst] = execLatch.result]
             /\ pc' = pc + 1
             /\ IF pc + 1 >= MaxPC THEN vmHalted' = TRUE ELSE UNCHANGED vmHalted
             /\ stageIdx' = 1
          ELSE
             /\ UNCHANGED << regs, pc, vmHalted >>
             /\ stageIdx' = stageIdx + 1
       /\ stepCount' = stepCount + 1
       /\ UNCHANGED << pipeline, mutationEntropy >>

Terminated ==
    /\ vmHalted
    /\ UNCHANGED vars

Next ==
    /\ stepCount < MaxSteps
    /\ (StepGeneric \/ Terminated)

TypeOK ==
    /\ pc \in 0..(MaxPC + 1)
    /\ stageIdx \in 1..(Len(pipeline) + 1)
    /\ vmHalted \in BOOLEAN

DataHazardFree ==
    (stageIdx <= Len(pipeline) /\ pipeline[stageIdx] = "EXECUTE") => (operandLatch.valid /\ decodeLatch.valid)

CommitIntegrity ==
    (stageIdx <= Len(pipeline) /\ pipeline[stageIdx] = "COMMIT_WRITEBACK") => execLatch.valid

Liveness ==
    ~vmHalted => (stageIdx <= Len(pipeline))

=============================================================================
