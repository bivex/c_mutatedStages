#include <stdio.h>
#include <assert.h>
#include "RV_Metamorphic_VM_92b91413cfc5.h"

int main(void) {
    RV_Metamorphic_VM_92b91413cfc5_context_t ctx = {0};
    ctx.pc = 0;
    ctx.sp = 0;
    ctx.current_stage_idx = 0;
    ctx.entropy_key = 0x1337;
    ctx.halted = false;

    printf("[C Test] Initialized VM: %s\n", ARCH_ID);
    printf("[C Test] Pipeline stage count: %d\n", PIPELINE_STAGE_COUNT);
    assert(PIPELINE_STAGE_COUNT == 6);
    return 0;
}
