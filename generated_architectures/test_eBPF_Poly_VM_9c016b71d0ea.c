#include <stdio.h>
#include <assert.h>
#include "eBPF_Poly_VM_9c016b71d0ea.h"

int main(void) {
    eBPF_Poly_VM_9c016b71d0ea_context_t ctx = {0};
    ctx.pc = 0;
    ctx.sp = 0;
    ctx.current_stage_idx = 0;
    ctx.entropy_key = 0x1337;
    ctx.halted = false;

    printf("[C Test] Initialized VM: %s\n", ARCH_ID);
    printf("[C Test] Pipeline stage count: %d\n", PIPELINE_STAGE_COUNT);
    assert(PIPELINE_STAGE_COUNT == 7);
    return 0;
}
