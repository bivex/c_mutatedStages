---------------------------- MODULE Verify_Sail_DeepStagedVM_f6f61d640751 ----------------------------
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS WordWidth, FeistelRounds, MBARounds, RegCount, MaxPC, MaxSteps

VARIABLES pc, regs, entropy, latchFeistelL, latchFeistelR, latchOp, latchRd, latchRs1, latchRs2,
          latchOpA, latchOpB, latchCarry, latchSumBits, latchMBATerm, latchFinalRes,
          stageId, stepCount, vmHalted

vars == << pc, regs, entropy, latchFeistelL, latchFeistelR, latchOp, latchRd, latchRs1, latchRs2,
           latchOpA, latchOpB, latchCarry, latchSumBits, latchMBATerm, latchFinalRes,
           stageId, stepCount, vmHalted >>

BitsToNat(bits) ==
    LET Sum[i \in 0..WordWidth] ==
        IF i = 0 THEN 0 ELSE Sum[i - 1] + bits[i - 1] * (2^(i - 1))
    IN Sum[WordWidth]

GetBit(x, i) == (x \div (2^i)) % 2
BitXOR(a, b) == (a + b) % 2
BitAND(a, b) == IF a = 1 /\ b = 1 THEN 1 ELSE 0
BitOR(a, b)  == IF a = 1 \/ b = 1 THEN 1 ELSE 0

StageCount == FeistelRounds + 4 + WordWidth + MBARounds + 2 + 2

Init ==
    /\ pc = 0
    /\ regs = [r \in 1..RegCount |-> IF r = 1 THEN 3 ELSE IF r = 2 THEN 5 ELSE 0]
    /\ entropy = 42
    /\ latchFeistelL = 7
    /\ latchFeistelR = 11
    /\ latchOp = 1
    /\ latchRd = 3
    /\ latchRs1 = 1
    /\ latchRs2 = 2
    /\ latchOpA = 0
    /\ latchOpB = 0
    /\ latchCarry = [i \in 0..WordWidth |-> 0]
    /\ latchSumBits = [i \in 0..(WordWidth - 1) |-> 0]
    /\ latchMBATerm = 0
    /\ latchFinalRes = 0
    /\ stageId = 1
    /\ stepCount = 0
    /\ vmHalted = FALSE

StepFeistel(r) ==
    /\ ~vmHalted
    /\ stageId = r
    /\ LET roundKey == (entropy + r * 13) % 16
           f == BitXOR(latchFeistelR, roundKey)
       IN
       /\ latchFeistelL' = latchFeistelR
       /\ latchFeistelR' = BitXOR(latchFeistelL, f)
    /\ stageId' = stageId + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, entropy, latchOp, latchRd, latchRs1, latchRs2,
                    latchOpA, latchOpB, latchCarry, latchSumBits, latchMBATerm, 
                    latchFinalRes, vmHalted >>

StepDecode ==
    /\ ~vmHalted
    /\ stageId = FeistelRounds + 1
    /\ latchOp' = 1
    /\ latchRd' = 3
    /\ latchRs1' = 1
    /\ latchRs2' = 2
    /\ stageId' = stageId + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, entropy, latchFeistelL, latchFeistelR, latchOpA,
                    latchOpB, latchCarry, latchSumBits, latchMBATerm, latchFinalRes, vmHalted >>

StepFetchRs1 ==
    /\ ~vmHalted
    /\ stageId = FeistelRounds + 2
    /\ latchOpA' = regs[latchRs1]
    /\ stageId' = stageId + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, entropy, latchFeistelL, latchFeistelR, latchOp,
                    latchRd, latchRs1, latchRs2, latchOpB, latchCarry, 
                    latchSumBits, latchMBATerm, latchFinalRes, vmHalted >>

StepFetchRs2 ==
    /\ ~vmHalted
    /\ stageId = FeistelRounds + 3
    /\ latchOpB' = regs[latchRs2]
    /\ stageId' = stageId + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, entropy, latchFeistelL, latchFeistelR, latchOp,
                    latchRd, latchRs1, latchRs2, latchOpA, latchCarry, 
                    latchSumBits, latchMBATerm, latchFinalRes, vmHalted >>

StepInitCarry ==
    /\ ~vmHalted
    /\ stageId = FeistelRounds + 4
    /\ latchCarry' = [i \in 0..WordWidth |-> 0]
    /\ latchSumBits' = [i \in 0..(WordWidth - 1) |-> 0]
    /\ stageId' = stageId + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, entropy, latchFeistelL, latchFeistelR, latchOp,
                    latchRd, latchRs1, latchRs2, latchOpA, latchOpB,
                    latchMBATerm, latchFinalRes, vmHalted >>

StepBitSliceAdder(b) ==
    /\ ~vmHalted
    /\ stageId = FeistelRounds + 4 + b + 1
    /\ LET bitA == GetBit(latchOpA, b)
           bitB == GetBit(latchOpB, b)
           cIn  == latchCarry[b]
           sumBit == BitXOR(BitXOR(bitA, bitB), cIn)
           cOut   == BitOR(BitAND(bitA, bitB), BitAND(cIn, BitXOR(bitA, bitB)))
       IN
       /\ latchSumBits' = [latchSumBits EXCEPT ![b] = sumBit]
       /\ latchCarry'   = [latchCarry EXCEPT ![b + 1] = cOut]
    /\ stageId' = stageId + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, entropy, latchFeistelL, latchFeistelR, latchOp,
                    latchRd, latchRs1, latchRs2, latchOpA, latchOpB,
                    latchMBATerm, latchFinalRes, vmHalted >>

StepMBATransform(m) ==
    /\ ~vmHalted
    /\ stageId = FeistelRounds + 4 + WordWidth + m
    /\ latchMBATerm' = (BitsToNat(latchSumBits) + m * 0) % (2^WordWidth)
    /\ latchFinalRes' = BitsToNat(latchSumBits)
    /\ stageId' = stageId + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, entropy, latchFeistelL, latchFeistelR, latchOp,
                    latchRd, latchRs1, latchRs2, latchOpA, latchOpB,
                    latchCarry, latchSumBits, vmHalted >>

StepMutateEntropy ==
    /\ ~vmHalted
    /\ stageId = FeistelRounds + 4 + WordWidth + MBARounds + 1
    /\ entropy' = (entropy * 5 + 1) % 256
    /\ stageId' = stageId + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, latchFeistelL, latchFeistelR, latchOp,
                    latchRd, latchRs1, latchRs2, latchOpA, latchOpB,
                    latchCarry, latchSumBits, latchMBATerm, latchFinalRes, vmHalted >>

StepMutateBarrier ==
    /\ ~vmHalted
    /\ stageId = FeistelRounds + 4 + WordWidth + MBARounds + 2
    /\ stageId' = stageId + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, regs, entropy, latchFeistelL, latchFeistelR, latchOp,
                    latchRd, latchRs1, latchRs2, latchOpA, latchOpB,
                    latchCarry, latchSumBits, latchMBATerm, latchFinalRes, vmHalted >>

StepCommitGPR ==
    /\ ~vmHalted
    /\ stageId = StageCount - 1
    /\ regs' = [regs EXCEPT ![latchRd] = latchFinalRes]
    /\ stageId' = stageId + 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << pc, entropy, latchFeistelL, latchFeistelR, latchOp,
                    latchRd, latchRs1, latchRs2, latchOpA, latchOpB,
                    latchCarry, latchSumBits, latchMBATerm, latchFinalRes, vmHalted >>

StepCommitPCAndFlush ==
    /\ ~vmHalted
    /\ stageId = StageCount
    /\ pc' = pc + 1
    /\ IF pc + 1 >= MaxPC THEN vmHalted' = TRUE ELSE UNCHANGED vmHalted
    /\ stageId' = 1
    /\ stepCount' = stepCount + 1
    /\ UNCHANGED << regs, entropy, latchFeistelL, latchFeistelR, latchOp,
                    latchRd, latchRs1, latchRs2, latchOpA, latchOpB,
                    latchCarry, latchSumBits, latchMBATerm, latchFinalRes >>

Terminated ==
    /\ vmHalted
    /\ UNCHANGED vars

Next ==
    /\ stepCount < MaxSteps
    /\ ((\E r \in 1..FeistelRounds : StepFeistel(r))
       \/ StepDecode \/ StepFetchRs1 \/ StepFetchRs2 \/ StepInitCarry
       \/ (\E b \in 0..(WordWidth - 1) : StepBitSliceAdder(b))
       \/ (\E m \in 1..MBARounds : StepMBATransform(m))
       \/ StepMutateEntropy \/ StepMutateBarrier
       \/ StepCommitGPR \/ StepCommitPCAndFlush
       \/ Terminated)

TypeOK ==
    /\ pc \in 0..MaxPC
    /\ \A r \in 1..RegCount : regs[r] \in 0..(2^WordWidth - 1)
    /\ stageId \in 1..(StageCount + 1)
    /\ vmHalted \in BOOLEAN

BitSliceCorrectness ==
    (stageId > FeistelRounds + 4 + WordWidth) =>
        (BitsToNat(latchSumBits) = (latchOpA + latchOpB) % (2^WordWidth))

MBAEquivalence ==
    (stageId > FeistelRounds + 4 + WordWidth + MBARounds) =>
        (latchFinalRes = (latchOpA + latchOpB) % (2^WordWidth))

DataHazardFreedom ==
    (stageId > FeistelRounds + 4 /\ stageId <= FeistelRounds + 4 + WordWidth) =>
        (latchOpA = regs[latchRs1] /\ latchOpB = regs[latchRs2])

CommitIntegrity ==
    (pc = 1 /\ stageId = 1) => (regs[latchRd] = (regs[latchRs1] + regs[latchRs2]) % (2^WordWidth))

Liveness ==
    ~vmHalted => (stageId <= StageCount)

=============================================================================
