/*
 * QEMU GPGPU - RISC-V SIMT Core Implementation
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "gpgpu.h"
#include "gpgpu_core.h"
#include "math.h"
/**
 * gpgpu_core_init_warp - 初始化一个 warp 中所有活跃 lane 的状态
 *
 * 为每个活跃 lane 设置:
 *   - PC 指向内核入口
 *   - mhartid 编码: [block_id_linear:19 | warp_id:8 | lane_id:5]
 *   - 寄存器清零 (x0 硬连线为 0)
 *   - 浮点状态初始化
 */
void gpgpu_core_init_warp(GPGPUWarp *warp, uint32_t pc,
                          uint32_t thread_id_base, const uint32_t block_id[3],
                          uint32_t num_threads,
                          uint32_t warp_id, uint32_t block_id_linear)
{
    int i;

    if (num_threads > GPGPU_WARP_SIZE)
    {
        num_threads = GPGPU_WARP_SIZE;
    }

    memset(warp, 0, sizeof(*warp));

    /* 设置 warp 级上下文 */
    warp->active_mask = (num_threads == GPGPU_WARP_SIZE)
                            ? 0xFFFFFFFF
                            : (1u << num_threads) - 1;
    warp->thread_id_base = thread_id_base;
    warp->warp_id = warp_id;
    warp->block_id[0] = block_id[0];
    warp->block_id[1] = block_id[1];
    warp->block_id[2] = block_id[2];

    /* 初始化每个活跃 lane */
    for (i = 0; i < num_threads; i++)
    {
        GPGPULane *lane = &warp->lanes[i];

        lane->pc = pc;
        /* mhartid = [block(19b) | warp(8b) | thread(5b)] */
        lane->mhartid = MHARTID_ENCODE(block_id_linear, warp_id, i);
        lane->gpr[0] = 0; /* x0 硬连线为 0 */
        lane->fcsr = 0;
        lane->active = true;
    }
}
/* 在 gpgpu_core_exec_warp 函数之前，添加两个 C 语言辅助函数 */
static uint8_t float_to_e4m3_c(float f)
{
    if (f == 0.0f)
        return 0x00;
    uint32_t bits;
    memcpy(&bits, &f, sizeof(uint32_t));
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = (bits >> 16) & 0x7F;
    if (exp < -6)
        return (sign << 7) | 0x00;
    if (exp > 7)
        return (sign << 7) | 0x7E;
    uint8_t e4m3_exp = (uint8_t)(exp + 7);
    uint8_t e4m3_mant = (mant >> 4) & 0x7;
    return (sign << 7) | (e4m3_exp << 3) | e4m3_mant;
}

static float e4m3_to_float_c(uint8_t e4m3)
{
    if (e4m3 == 0)
        return 0.0f;
    uint32_t sign = (e4m3 >> 7) & 1;
    int32_t exp = ((e4m3 >> 3) & 0xF) - 7;
    uint32_t mant = (e4m3 & 0x7) << 4;
    uint32_t fp32_bits = (sign << 31) | ((exp + 127) << 23) | (mant << 16);
    float f;
    memcpy(&f, &fp32_bits, sizeof(float));
    return f;
}

static uint8_t float_to_e5m2_c(float f)
{
    if (f == 0.0f)
        return 0x00;
    uint32_t bits;
    memcpy(&bits, &f, sizeof(uint32_t));
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = (bits >> 16) & 0x7F;
    if (exp < -6)
        return (sign << 7) | 0x00;
    if (exp > 7)
        return (sign << 7) | 0x7F;

    uint8_t e5m2_exp = (uint8_t)(exp + 15);
    uint8_t e5m2_mant = (mant >> 5) & 0x3;
    return (sign << 7) | (e5m2_exp << 2) | e5m2_mant;
}

static float e5m2_to_float_c(uint8_t e5m2)
{
    if (e5m2 == 0)
        return 0.0f;
    uint32_t sign = (e5m2 >> 7) & 1;
    int32_t exp = ((e5m2 >> 2) & 0x1F) - 15;
    uint32_t mant = (e5m2 & 0X3) << 5;
    uint32_t fp32_bits = (sign << 31) | ((exp + 127) << 23) | (mant << 16);
    float f;
    memcpy(&f, &fp32_bits, sizeof(float));
    return f;
}

static uint8_t float_to_e2m1_c(float f)
{
    if (f == 0.0f)
        return 0x00;
    uint32_t bits;
    memcpy(&bits, &f, sizeof(uint32_t));
    uint32_t sign = (bits >> 31) & 1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t mant = (bits >> 16) & 0x7F;
    int32_t e2m1_exp = exp + 1;
    if (e2m1_exp < 0)
        return (sign << 3) | 0x00;
    if (e2m1_exp > 3)
        return (sign << 3) | 0x7;
    uint8_t e2m1_mant = (mant >> 5) & 0x1;
    return (sign << 3) | (e2m1_exp << 1) | e2m1_mant;
}

static float e2m1_to_float_c(uint8_t e2m1)
{
    if (e2m1 == 0)
        return 0.0f;
    uint32_t sign = (e2m1 >> 3) & 1;
    int32_t exp = ((e2m1 >> 1) & 0X3) - 1;
    uint32_t mant = (e2m1 & 1) << 6;
    uint32_t fp32_bits = (sign << 31) | ((exp + 127) << 23) | (mant << 16);
    float f;
    memcpy(&f, &fp32_bits, sizeof(float));
    return f;
}

/**
 * gpgpu_core_exec_warp - 执行单个 warp 直到完成或达到周期上限
 *
 * TODO: 实现 RV32I + RV32F 指令解释器
 *   - 取指: 从 s->vram_ptr + lane->pc 读取指令
 *   - 解码: RV32I/RV32F 指令集
 *   - 锁步执行: 所有活跃 lane 执行同一条指令
 *   - 访存: 拦截 0x80000000 范围的 CTRL 地址返回 simt 上下文
 *   - 分支分歧: 通过 active_mask 管理
 *   - Barrier: 写 GPGPU_REG_BARRIER 触发同步
 */
int gpgpu_core_exec_warp(GPGPUState *s, GPGPUWarp *warp, uint32_t max_cycles)
{
    uint32_t cycle = 0;
    GPGPULane *lane0 = &warp->lanes[0];
    if (warp->active_mask == 0)
        return 0;
    while (cycle < max_cycles && warp->active_mask != 0)
    {
        uint32_t pc = lane0->pc;

        // 1.fetch instruction
        uint32_t fetch_addr = pc;
        if (fetch_addr + 4 > s->vram_size)
        {
            printf("kernel address out of range!\n");
            return -1;
        }

        uint32_t ins = *(uint32_t *)(s->vram_ptr + fetch_addr);

        // 2.decode instruction
        uint32_t opcode = ins & 0x7f;
        uint32_t rd = (ins >> 7) & 0x1F;
        uint32_t funct3 = (ins >> 12) & 0x7;
        uint32_t rs1 = (ins >> 15) & 0x1F;
        uint32_t rs2 = (ins >> 20) & 0x1F;
        uint32_t funct7 = (ins >> 25) & 0x7F;

        int32_t imm_i = (int32_t)(ins & 0xFFF00000) >> 20; // sign-extend
        uint32_t imm_u = ins & 0xFFFFF000;                 // lui
        uint32_t imm_s = (((ins >> 25) & 0x7F) << 5)

                         | ((ins >> 7) & 0x1F); // sw offset
        imm_s = (int32_t)(imm_s << 20) >> 20;   // sign-extend 12-bit
        printf("ins %08x  opcode %08x  rd %08x  funct3 %08x  rs1 %08x  rs2 %08x  funct7 %08x\r\n",
               ins, opcode, rd, funct3, rs1, rs2, funct7);

        switch (opcode)
        {
        case 0x73: // system
        {
            uint32_t csr_addr = (ins >> 20) & 0xFFF;
            for (uint32_t t = 0; t < GPGPU_WARP_SIZE; t++)
            {
                GPGPULane *lane = &warp->lanes[t];
                switch (funct3)
                {
                case 0:
                    if (csr_addr == 0x1)
                    {
                        // printf("ebreak: deactivating warp\n"); // 新增
                        warp->active_mask = 0;
                    }
                    break;
                case 1:
                    /* code */
                    printf("csrrw x%d, 0x%03x, x%d (not fully implemented)\n", rd, csr_addr, rs1); // 新增
                    break;
                case 2: // mhartid
                    if (csr_addr == 0xF14)
                    {
                        lane->gpr[rd] = lane->mhartid;
                    }
                    break;
                default:
                    printf("system: funct3=%d, csr=0x%03x\n", funct3, csr_addr); // 新增
                    break;
                }
            }
            break;
        }
        case 0x13: // imm op
        {
            for (uint32_t t = 0; t < GPGPU_WARP_SIZE; t++)
            {
                if (!(warp->active_mask & (1u << t)))
                    continue;
                GPGPULane *lane = &warp->lanes[t];
                uint32_t src = lane->gpr[rs1];
                uint32_t result = 0;

                switch (funct3)
                {
                case 0: // addi rd, rs1, imm
                    result = src + imm_i;
                    printf("addi src %08x result %08x\r\n", src, result);
                    break;
                case 1: // slli rd, rs1, shamt
                    result = src << (imm_i & 0x1F);
                    printf("slli src %08x result %08x\r\n", src, result);
                    break;
                case 7: // andi rd, rs1, imm
                    result = src & (uint32_t)imm_i;
                    printf("andi src %08x result %08x\r\n", src, result);
                    break;
                default:
                    printf("Unsupport OP-IMM: funct3=%d, rs1=%d, imm=%d\n", funct3, rs1, imm_i); // 新增
                    break;
                }
                if (rd != 0)
                    lane->gpr[rd] = result;
            }
            break;
        }
        case 0x37: // lui
        {
            for (uint32_t t = 0; t < GPGPU_WARP_SIZE; t++)
            {
                if (!(warp->active_mask & (1u << t)))
                    continue;
                GPGPULane *lane = &warp->lanes[t];
                if (rd != 0)
                {
                    lane->gpr[rd] = imm_u;
                    printf("lui x%d, 0x%08x\n", rd, imm_u); // 新增
                }
            }
            break;
        }
        case 0x33: // reg op
        {
            // ADD
            for (uint32_t t = 0; t < GPGPU_WARP_SIZE; t++)
            {
                if (!(warp->active_mask & (1u << t)))
                    continue;
                GPGPULane *lane = &warp->lanes[t];
                if (rd != 0)
                {
                    uint32_t src1 = lane->gpr[rs1];
                    uint32_t src2 = lane->gpr[rs2];
                    lane->gpr[rd] = src1 + src2;
                    printf("add x%d, x%d, x%d: 0x%08x + 0x%08x = 0x%08x\n",
                           rd, rs1, rs2, src1, src2, lane->gpr[rd]); // 新增
                }
            }
            break;
        }
        /* ===== OP-FP (0x53): fcvt.s.w, fmul.s, fadd.s, fcvt.w.s ===== */
        case 0x53:
        {
            uint8_t rm = (ins >> 12) & 0x7;
            for (uint32_t t = 0; t < GPGPU_WARP_SIZE; t++)
            {
                if (!(warp->active_mask & (1u << t)))
                    continue;
                GPGPULane *lane = &warp->lanes[t];

                if (rm == 7)
                    printf("need support lane float rounding mode\n");

                switch (funct7)
                {
                case 0x00: // FADD.S
                {
                    float fsrc1, fsrc2, fdst;
                    memcpy(&fsrc1, &lane->fpr[rs1], sizeof(float));
                    memcpy(&fsrc2, &lane->fpr[rs2], sizeof(float));
                    fdst = fsrc1 + fsrc2;
                    memcpy(&lane->fpr[rd], &fdst, sizeof(float));
                    printf("fadd.s f%d, f%d, f%d: %f + %f = %f\n",
                           rd, rs1, rs2, fsrc1, fsrc2, fdst);
                    break;
                }
                case 0x08: // FMUL.S
                {
                    float fsrc1, fsrc2, fdst;
                    memcpy(&fsrc1, &lane->fpr[rs1], sizeof(float));
                    memcpy(&fsrc2, &lane->fpr[rs2], sizeof(float));
                    fdst = fsrc1 * fsrc2;
                    memcpy(&lane->fpr[rd], &fdst, sizeof(float));
                    printf("fmul.s f%d, f%d, f%d: %f * %f = %f\n",
                           rd, rs1, rs2, fsrc1, fsrc2, fdst);
                    break;
                }
                case 0x68: // FCVT.S.W
                {
                    float fsrc = (float)((int32_t)lane->gpr[rs1]);
                    memcpy(&lane->fpr[rd], &fsrc, sizeof(float));
                    printf("fcvt.s.w f%d, x%d: %d -> %f\n", rd, rs1, lane->gpr[rs1], fsrc);
                    break;
                }
                case 0x60: // FCVT.W.S
                {
                    float fsrc;
                    int32_t result;
                    memcpy(&fsrc, &lane->fpr[rs1], sizeof(float));
                    switch (rm)
                    {
                    case 0:
                        result = (int32_t)roundf(fsrc);
                        break;
                    case 1:
                        result = (int32_t)fsrc;
                        break;
                    case 2:
                        result = (int32_t)floorf(fsrc);
                        break;
                    case 3:
                        result = (int32_t)ceilf(fsrc);
                        break;
                    default:
                        result = (int32_t)fsrc;
                        break;
                    }
                    lane->gpr[rd] = (uint32_t)result;
                    printf("fcvt.w.s x%d, f%d, rm=%d: %f -> %d (0x%08x)\n",
                           rd, rs1, rm, fsrc, result, (uint32_t)result); // 新增打印
                    break;
                }
                case 0x22: /* e2m1 ↔ float */
                {
                    if (rs2 == 0) /* fcvt.s.bf16: bf16 → float */
                    {
                        uint32_t bf16_bits = lane->fpr[rs1] & 0xFFFF;
                        uint32_t fp32_bits = bf16_bits << 16;
                        float fresult;
                        memcpy(&fresult, &fp32_bits, sizeof(float));
                        memcpy(&lane->fpr[rd], &fresult, sizeof(float));
                        printf("fcvt.s.bf16 f%d, f%d: bf16=0x%04x -> float=%f (0x%08x)\n",
                               rd, rs1, bf16_bits, fresult, fp32_bits);
                    }
                    else /* fcvt.bf16.s: float → bf16 */
                    {
                        float fsrc;
                        memcpy(&fsrc, &lane->fpr[rs1], sizeof(float));
                        uint32_t fp32_bits;
                        memcpy(&fp32_bits, &fsrc, sizeof(uint32_t));
                        uint16_t bf16_bits = (uint16_t)(fp32_bits >> 16);
                        uint32_t boxed = 0xFFFF0000 | bf16_bits;
                        lane->fpr[rd] = boxed;
                        printf("fcvt.bf16.s f%d, f%d: float=%f (0x%08x) -> bf16=0x%04x, boxed=0x%08x\n",
                               rd, rs1, fsrc, fp32_bits, bf16_bits, boxed);
                    }
                    break;
                }

                case 0x24: /* E4M3 ↔ float */
                {
                    if (rs2 == 0) /* fcvt.s.e4m3: e4m3 → float */
                    {
                        uint8_t e4m3_val = (uint8_t)(lane->fpr[rs1] & 0xFF);
                        float fresult = e4m3_to_float_c(e4m3_val);
                        memcpy(&lane->fpr[rd], &fresult, sizeof(float));
                        printf("fcvt.s.e4m3 f%d, f%d: e4m3=0x%02x -> float=%f\n",
                               rd, rs1, e4m3_val, fresult);
                    }
                    else if (rs2 == 1) /* fcvt.e4m3.s: float → e4m3 */
                    {
                        float fsrc;
                        memcpy(&fsrc, &lane->fpr[rs1], sizeof(float));
                        uint8_t e4m3_val = float_to_e4m3_c(fsrc);
                        uint32_t boxed = 0xFFFFFF00 | e4m3_val;
                        lane->fpr[rd] = boxed;
                        printf("fcvt.e4m3.s f%d, f%d: float=%f -> e4m3=0x%02x, boxed=0x%08x\n",
                               rd, rs1, fsrc, e4m3_val, boxed);
                    }
                    else if (rs2 == 2) /* fcvt.s.e5m2: e5m2 → fp32 */
                    {
                        uint8_t e5m2_val = (uint8_t)(lane->fpr[rs1] & 0xFF);
                        float fresult = e5m2_to_float_c(e5m2_val);
                        memcpy(&lane->fpr[rd], &fresult, sizeof(float));
                        printf("fcvt.s.e5m2 f%d, f%d: e4m3=0x%02x -> float=%f\n",
                               rd, rs1, e5m2_val, fresult);
                    }
                    else if (rs2 == 3) /* fcvt.s.e5m2: float → e5m2 */
                    {
                        float fsrc;
                        memcpy(&fsrc, &lane->fpr[rs1], sizeof(float));
                        uint8_t e5m2_val = float_to_e5m2_c(fsrc);
                        uint32_t boxed = 0xFFFFFF00 | e5m2_val;
                        lane->fpr[rd] = boxed;
                        printf("fcvt.e4m3.s f%d, f%d: float=%f -> e4m3=0x%02x, boxed=0x%08x\n",
                               rd, rs1, fsrc, e5m2_val, boxed);
                    }
                    else
                    {
                        printf("Unsupport rs2%d \r\n", rs2);
                    }
                    break;
                }
                case 0x26: /* E2M1 ↔ float */
                {
                    if (rs2 == 0) /* fcvt.s.e2m1: e2m1 → float */
                    {
                        uint8_t e2m1_val = (uint8_t)(lane->fpr[rs1] & 0xF);
                        float fresult = e2m1_to_float_c(e2m1_val);
                        memcpy(&lane->fpr[rd], &fresult, sizeof(float));
                        printf("fcvt.s.e2m1 f%d, f%d: e2m1=0x%02x -> float=%f\n",
                               rd, rs1, e2m1_val, fresult);
                    }
                    else /* fcvt.e2m1.s: float → e2m1 */
                    {
                        float fsrc;
                        memcpy(&fsrc, &lane->fpr[rs1], sizeof(float));
                        uint8_t e2m1_val = float_to_e2m1_c(fsrc);
                        uint32_t boxed = 0xFFFFFF00 | e2m1_val;
                        lane->fpr[rd] = boxed;
                        printf("fcvt.e4m3.s f%d, f%d: float=%f -> e4m3=0x%02x, boxed=0x%08x\n",
                               rd, rs1, fsrc, e2m1_val, boxed);
                    }
                    break;
                }
                default:
                    printf("unhandled OP-FP funct7=0x%02x\n", funct7);
                    break;
                }
            }
        }
        break;
        case 0x23: // memory
        {
            for (uint32_t t = 0; t < GPGPU_WARP_SIZE; t++)
            {
                if (!(warp->active_mask & (1u << t)))
                    continue;
                GPGPULane *lane = &warp->lanes[t];
                // store 处理
                if (funct3 == 2)
                {
                    uint32_t addr = lane->gpr[rs1] + imm_s;
                    printf("sw x%d, %d(x%d): store 0x%08x to addr 0x%08x\n",
                           rs2, imm_s, rs1, lane->gpr[rs2], addr); // 新增更详细打印
                    if (addr + 4 < s->vram_size)
                    {
                        memcpy(s->vram_ptr + addr, &lane->gpr[rs2], sizeof(lane->gpr[rs2]));
                    }
                }
                else
                {
                    printf("store: unhandled funct3=%d\n", funct3); // 新增
                }
            }
            break;
        }
        default:
            printf("Unsupported opcode 0x%02x at PC=0x%08x\n", opcode, pc); // 新增 PC 信息
            break;
        }
        lane0->pc += 4;
        cycle++;
    }
    return 0;
}

/**
 * gpgpu_core_exec_kernel - 完整的 kernel 调度与执行
 *
 * 调用链: gpgpu_ctrl_write(GPGPU_REG_DISPATCH) → gpgpu_core_exec_kernel()
 *
 * 执行流程:
 *   1. 从 s->kernel 读取 grid_dim / block_dim
 *   2. 三重循环遍历所有 block
 *   3. 每个 block 内按 warp 粒度切分线程
 *   4. gpgpu_core_init_warp() → gpgpu_core_exec_warp()
 */
int gpgpu_core_exec_kernel(GPGPUState *s)
{

    GPGPUWarp warp;
    uint32_t block_id[3];
    uint32_t threads_per_block;
    uint32_t warps_per_block;
    uint32_t num_threads;
    uint32_t block_id_linear = 0;
    uint32_t gx, gy, gz, w;
    int ret;

    /* 计算每个 block 的总线程数和 warp 数 */
    threads_per_block = s->kernel.block_dim[0] *
                        s->kernel.block_dim[1] *
                        s->kernel.block_dim[2];
    if (threads_per_block == 0)
    {
        return 0; /* 未配置 block 维度, 无任务执行 */
    }
    warps_per_block = (threads_per_block + GPGPU_WARP_SIZE - 1) / GPGPU_WARP_SIZE;

    /* 三重循环: 遍历 grid 中所有 block */
    for (gz = 0; gz < s->kernel.grid_dim[2]; gz++)
    {
        block_id[2] = gz;
        for (gy = 0; gy < s->kernel.grid_dim[1]; gy++)
        {
            block_id[1] = gy;
            for (gx = 0; gx < s->kernel.grid_dim[0]; gx++)
            {
                block_id[0] = gx;

                /* 每个 block 内的所有 warp */
                for (w = 0; w < warps_per_block; w++)
                {
                    uint32_t thread_id_base = w * GPGPU_WARP_SIZE;
                    uint32_t remaining = threads_per_block - thread_id_base;
                    num_threads = remaining < GPGPU_WARP_SIZE
                                      ? remaining
                                      : GPGPU_WARP_SIZE;

                    /* 更新 SIMT 上下文: 当前执行的 block/warp 信息 */
                    s->simt.block_id[0] = block_id[0];
                    s->simt.block_id[1] = block_id[1];
                    s->simt.block_id[2] = block_id[2];
                    s->simt.warp_id = w;
                    s->simt.thread_id[0] = thread_id_base;
                    s->simt.thread_id[1] = 0;
                    s->simt.thread_id[2] = 0;

                    /* 初始化 warp → 执行 warp */
                    gpgpu_core_init_warp(&warp,
                                         (uint32_t)s->kernel.kernel_addr,
                                         thread_id_base,
                                         block_id,
                                         num_threads,
                                         w,
                                         block_id_linear);

                    ret = gpgpu_core_exec_warp(s, &warp, 1000000);
                    if (ret != 0)
                    {
                        qemu_log_mask(LOG_GUEST_ERROR,
                                      "gpgpu: warp exec error "
                                      "block(%u,%u,%u) warp=%u ret=%d\n",
                                      gx, gy, gz, w, ret);
                        return ret;
                    }
                }

                block_id_linear++;
            }
        }
    }

    return 0;
}
