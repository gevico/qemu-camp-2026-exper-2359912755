/*
 * QEMU Educational GPGPU Device
 *
 * Copyright (c) 2024-2025
 *
 * This work is licensed under the terms of the GNU GPL, version 2 or later.
 * See the COPYING file in the top-level directory.
 */

#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/units.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "qapi/error.h"
#include "hw/pci/pci.h"
#include "hw/pci/msi.h"
#include "hw/pci/msix.h"
#include "hw/core/qdev-properties.h"
#include "migration/vmstate.h"

#include "gpgpu.h"
#include "gpgpu_core.h"

// static void dma_rw(GPGPUState *gpgpuStatus, bool write, dma_addr_t *val, dma_addr_t *dma,
//                 bool timer)
// {
//     // if (write && (edu->dma.cmd & EDU_DMA_RUN)) {
//     //     return;
//     // }

//     // if (write) {
//     //     *dma = *val;
//     // } else {
//     //     *val = *dma;
//     // }

//     if (timer) {
//         timer_mod(&gpgpuStatus->dma_timer, qemu_clock_get_ms(QEMU_CLOCK_VIRTUAL) + 100);
//     }
// }

/* TODO: Implement MMIO control register read */
static uint64_t gpgpu_ctrl_read(void *opaque, hwaddr addr, unsigned size)
{
    uint64_t val = ~0ULL;
    GPGPUState *gpgpuStatus = opaque;

    if (size != 4)
        return val;

    switch (addr)
    {
    case GPGPU_REG_DEV_ID:
        val = GPGPU_DEV_ID_VALUE; // device id
        break;
    case GPGPU_REG_DEV_VERSION:
        val = GPGPU_DEV_VERSION_VALUE; // version number
        break;
    case GPGPU_REG_DEV_CAPS:
        val = GPGPU_DEV_VERSION_VALUE;
        break;
    case GPGPU_REG_VRAM_SIZE_LO:
        val = GPGPU_DEFAULT_VRAM_SIZE & 0xffffffff;
        break;
    case GPGPU_REG_VRAM_SIZE_HI:
        val = ((uint64_t)GPGPU_DEFAULT_VRAM_SIZE >> 32) & 0xffffffff;
        break;
    /* 全局控制寄存器组 (0x0100 - 0x01FF): 设备使能、复位、状态查询 */
    case GPGPU_REG_GLOBAL_CTRL:
        val = gpgpuStatus->global_ctrl & GPGPU_CTRL_ENABLE;
        break;
    case GPGPU_REG_GLOBAL_STATUS:
        val = gpgpuStatus->global_status;
        break;
    case GPGPU_REG_ERROR_STATUS:
        val = gpgpuStatus->error_status;
        break;
    /* 中断控制寄存器组 (0x0200 - 0x02FF): 中断使能和状态管理 */
    case GPGPU_REG_IRQ_ENABLE:
        val = gpgpuStatus->irq_enable;
        break;
    case GPGPU_REG_IRQ_STATUS:
        val = gpgpuStatus->irq_status;
        break;
    /* 内核分发寄存器组 (0x0300 - 0x03FF): 配置和启动 GPU 计算任务 */
    case GPGPU_REG_KERNEL_ADDR_LO:
        val = gpgpuStatus->kernel.kernel_addr & 0xffffffff;
        break;
    case GPGPU_REG_KERNEL_ADDR_HI:
        val = (gpgpuStatus->kernel.kernel_addr >> 32) & 0xffffffff;
        break;
    case GPGPU_REG_GRID_DIM_X:
        val = gpgpuStatus->kernel.grid_dim[0];
        break;
    case GPGPU_REG_GRID_DIM_Y:
        val = gpgpuStatus->kernel.grid_dim[1];
        break;
    case GPGPU_REG_GRID_DIM_Z:
        val = gpgpuStatus->kernel.grid_dim[2];
        break;
    case GPGPU_REG_BLOCK_DIM_X:
        val = gpgpuStatus->kernel.block_dim[0];
        break;
    case GPGPU_REG_BLOCK_DIM_Y:
        val = gpgpuStatus->kernel.block_dim[1];
        break;
    case GPGPU_REG_BLOCK_DIM_Z:
        val = gpgpuStatus->kernel.block_dim[2];
        break;
    case GPGPU_REG_SHARED_MEM_SIZE:
        val = gpgpuStatus->kernel.shared_mem_size;
        break;
    case GPGPU_REG_DISPATCH:
        val = (gpgpuStatus->global_status & GPGPU_STATUS_BUSY) ? 1 : 0;
        break;
    // DMA engine reg
    case GPGPU_REG_DMA_SRC_LO:
        val = gpgpuStatus->dma.src_addr & 0xffffffff;
        break;
    case GPGPU_REG_DMA_SRC_HI:
        val = (gpgpuStatus->dma.src_addr >> 32) & 0xffffffff;
        break;
    case GPGPU_REG_DMA_DST_LO:
        val = gpgpuStatus->dma.dst_addr & 0xffffffff;
        break;
    case GPGPU_REG_DMA_DST_HI:
        val = (gpgpuStatus->dma.dst_addr >> 32) & 0xffffffff;
        break;
    case GPGPU_REG_DMA_SIZE:
        val = gpgpuStatus->dma.size;
        break;
    case GPGPU_REG_DMA_CTRL:
        val = gpgpuStatus->dma.ctrl;
        break;
    case GPGPU_REG_DMA_STATUS:
        val = gpgpuStatus->dma.status;
        break;
    /* 线程上下文寄存器组 (0x1000 - 0x1FFF): GPU 线程读取自身 ID */
    case GPGPU_REG_THREAD_ID_X:
        val = gpgpuStatus->simt.thread_id[0];
        break;
    case GPGPU_REG_THREAD_ID_Y:
        val = gpgpuStatus->simt.thread_id[1];
        break;
    case GPGPU_REG_THREAD_ID_Z:
        val = gpgpuStatus->simt.thread_id[2];
        break;
    case GPGPU_REG_BLOCK_ID_X:
        val = gpgpuStatus->simt.block_id[0];
        break;
    case GPGPU_REG_BLOCK_ID_Y:
        val = gpgpuStatus->simt.block_id[1];
        break;
    case GPGPU_REG_BLOCK_ID_Z:
        val = gpgpuStatus->simt.block_id[2];
        break;
    case GPGPU_REG_WARP_ID:
        val = gpgpuStatus->simt.warp_id;
        break;
    case GPGPU_REG_LANE_ID:
        val = gpgpuStatus->simt.lane_id;
        break;
    case GPGPU_REG_BARRIER:
        val = gpgpuStatus->simt.barrier_active ? gpgpuStatus->simt.barrier_count : 0;
        break;
    case GPGPU_REG_THREAD_MASK:
        val = gpgpuStatus->simt.thread_mask;
        break;
    }
    return val;
}

/* TODO: Implement MMIO control register write */
static void gpgpu_ctrl_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *gpgpuStatus = opaque;
    printf("enter\r\n");
    if (size != 4)
        return;

    switch (addr)
    {
    case GPGPU_REG_GLOBAL_CTRL:
        /* Bit 0: 设备使能 */
        if (val & GPGPU_CTRL_ENABLE)
        {
            gpgpuStatus->global_ctrl |= GPGPU_CTRL_ENABLE;
        }
        else
        {
            gpgpuStatus->global_ctrl &= ~GPGPU_CTRL_ENABLE;
        }
        /* Bit 1: 软复位, 写 1 触发, 自动清除 SIMT 上下文 */
        if (val & GPGPU_CTRL_RESET)
        {
            memset(&gpgpuStatus->simt, 0, sizeof(gpgpuStatus->simt));
            gpgpuStatus->global_ctrl &= ~GPGPU_CTRL_RESET;
        }
        break;
    /* 中断控制寄存器组 (0x0200 - 0x02FF): 中断使能和状态管理 */
    case GPGPU_REG_IRQ_ENABLE:
        gpgpuStatus->irq_enable = val;
        break;
    case GPGPU_REG_IRQ_ACK:
        break;
    /* 内核分发寄存器组 (0x0300 - 0x03FF): 配置和启动 GPU 计算任务 */
    case GPGPU_REG_KERNEL_ADDR_LO:
        gpgpuStatus->kernel.kernel_addr |= val;
        break;
    case GPGPU_REG_KERNEL_ADDR_HI:
        gpgpuStatus->kernel.kernel_addr |= val << 32;
        break;
    case GPGPU_REG_GRID_DIM_X:
        gpgpuStatus->kernel.grid_dim[0] = val;
        break;
    case GPGPU_REG_GRID_DIM_Y:
        gpgpuStatus->kernel.grid_dim[1] = val;
        break;
    case GPGPU_REG_GRID_DIM_Z:
        gpgpuStatus->kernel.grid_dim[2] = val;
        break;
    case GPGPU_REG_BLOCK_DIM_X:
        gpgpuStatus->kernel.block_dim[0] = val;
        break;
    case GPGPU_REG_BLOCK_DIM_Y:
        gpgpuStatus->kernel.block_dim[1] = val;
        break;
    case GPGPU_REG_BLOCK_DIM_Z:
        gpgpuStatus->kernel.block_dim[2] = val;
        break;
    case GPGPU_REG_SHARED_MEM_SIZE:
        gpgpuStatus->kernel.shared_mem_size = val;
        break;
    case GPGPU_REG_DISPATCH:
        /* 写任意值启动内核执行: 设备需处于使能状态 */
        if (gpgpuStatus->global_ctrl & GPGPU_CTRL_ENABLE)
        {
            gpgpuStatus->global_status = GPGPU_STATUS_BUSY;
            gpgpu_core_exec_kernel(gpgpuStatus);
            gpgpuStatus->global_status = GPGPU_STATUS_READY;
            /* 内核执行完成, 触发中断 */
            if (gpgpuStatus->irq_enable & GPGPU_IRQ_KERNEL_DONE)
            {
                gpgpuStatus->irq_status |= GPGPU_IRQ_KERNEL_DONE;
                msix_notify(PCI_DEVICE(gpgpuStatus), GPGPU_MSIX_VEC_KERNEL);
            }
        }
        break;
    // DMA engine reg
    case GPGPU_REG_DMA_SRC_LO:
        gpgpuStatus->dma.src_addr |= val;
        break;
    case GPGPU_REG_DMA_SRC_HI:
        gpgpuStatus->dma.src_addr |= val << 32;
        break;
    case GPGPU_REG_DMA_DST_LO:
        gpgpuStatus->dma.dst_addr |= val;
        break;
    case GPGPU_REG_DMA_DST_HI:
        gpgpuStatus->dma.dst_addr |= val << 32;
        break;
    case GPGPU_REG_DMA_SIZE:
        gpgpuStatus->dma.size = val;
        break;
    case GPGPU_REG_DMA_CTRL:
        gpgpuStatus->dma.ctrl = val;
        break;
    /* 线程上下文寄存器组 (0x1000 - 0x1FFF): GPU 线程读取自身 ID */
    case GPGPU_REG_THREAD_ID_X:
        gpgpuStatus->simt.thread_id[0] = val;
        break;
    case GPGPU_REG_THREAD_ID_Y:
        gpgpuStatus->simt.thread_id[1] = val;
        break;
    case GPGPU_REG_THREAD_ID_Z:
        gpgpuStatus->simt.thread_id[2] = val;
        break;
    case GPGPU_REG_BLOCK_ID_X:
        gpgpuStatus->simt.block_id[0] = val;
        break;
    case GPGPU_REG_BLOCK_ID_Y:
        gpgpuStatus->simt.block_id[1] = val;
        break;
    case GPGPU_REG_BLOCK_ID_Z:
        gpgpuStatus->simt.block_id[2] = val;
        break;
    case GPGPU_REG_WARP_ID:
        gpgpuStatus->simt.warp_id = val;
        break;
    case GPGPU_REG_LANE_ID:
        gpgpuStatus->simt.lane_id = val;
        break;
    /* 同步寄存器组 (0x2000 - 0x2FFF): 线程同步原语 */
    case GPGPU_REG_BARRIER:
        /* 写任意值触发 barrier: 当前线程到达屏障 */
        gpgpuStatus->simt.barrier_active = true;
        gpgpuStatus->simt.barrier_count++;
        break;
    case GPGPU_REG_THREAD_MASK:
        gpgpuStatus->simt.thread_mask = val;
        break;
    }
}

static const MemoryRegionOps gpgpu_ctrl_ops = {
    .read = gpgpu_ctrl_read,
    .write = gpgpu_ctrl_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* VRAM read: 根据 access size 选择正确的宽度 */
static uint64_t gpgpu_vram_read(void *opaque, hwaddr addr, unsigned size)
{
    GPGPUState *gpgpuStatus = opaque;
    uint64_t val = ~0ULL;

    if (addr + size > gpgpuStatus->vram_size) {
        return val;
    }

    switch (size) {
    case 1:
        val = *(uint8_t *)(gpgpuStatus->vram_ptr + addr);
        break;
    case 2:
        val = *(uint16_t *)(gpgpuStatus->vram_ptr + addr);
        break;
    case 4:
        val = *(uint32_t *)(gpgpuStatus->vram_ptr + addr);
        break;
    case 8:
        val = *(uint64_t *)(gpgpuStatus->vram_ptr + addr);
        break;
    default:
        break;
    }
    return val;
}

/* VRAM write: 根据 access size 选择正确的宽度 */
static void gpgpu_vram_write(void *opaque, hwaddr addr, uint64_t val,
                             unsigned size)
{
    GPGPUState *gpgpuStatus = opaque;

    if (addr + size > gpgpuStatus->vram_size) {
        return;
    }

    switch (size) {
    case 1:
        *(uint8_t *)(gpgpuStatus->vram_ptr + addr) = (uint8_t)val;
        break;
    case 2:
        *(uint16_t *)(gpgpuStatus->vram_ptr + addr) = (uint16_t)val;
        break;
    case 4:
        *(uint32_t *)(gpgpuStatus->vram_ptr + addr) = (uint32_t)val;
        break;
    case 8:
        *(uint64_t *)(gpgpuStatus->vram_ptr + addr) = val;
        break;
    default:
        break;
    }
}

static const MemoryRegionOps gpgpu_vram_ops = {
    .read = gpgpu_vram_read,
    .write = gpgpu_vram_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

static uint64_t gpgpu_doorbell_read(void *opaque, hwaddr addr, unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)size;
    return 0;
}

static void gpgpu_doorbell_write(void *opaque, hwaddr addr, uint64_t val,
                                 unsigned size)
{
    (void)opaque;
    (void)addr;
    (void)val;
    (void)size;
}

static const MemoryRegionOps gpgpu_doorbell_ops = {
    .read = gpgpu_doorbell_read,
    .write = gpgpu_doorbell_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* TODO: Implement DMA completion handler */
static void gpgpu_dma_complete(void *opaque)
{
    (void)opaque;
}

/* TODO: Implement kernel completion handler */
static void gpgpu_kernel_complete(void *opaque)
{
    (void)opaque;
}

static void gpgpu_realize(PCIDevice *pdev, Error **errp)
{
    GPGPUState *s = GPGPU(pdev);
    uint8_t *pci_conf = pdev->config;

    pci_config_set_interrupt_pin(pci_conf, 1);

    s->vram_ptr = g_malloc0(s->vram_size);
    if (!s->vram_ptr)
    {
        error_setg(errp, "GPGPU: failed to allocate VRAM");
        return;
    }

    /* BAR 0: control registers */
    memory_region_init_io(&s->ctrl_mmio, OBJECT(s), &gpgpu_ctrl_ops, s,
                          "gpgpu-ctrl", GPGPU_CTRL_BAR_SIZE);
    pci_register_bar(pdev, 0,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                         PCI_BASE_ADDRESS_MEM_TYPE_64,
                     &s->ctrl_mmio);

    /* BAR 2: VRAM */
    memory_region_init_io(&s->vram, OBJECT(s), &gpgpu_vram_ops, s,
                          "gpgpu-vram", s->vram_size);
    pci_register_bar(pdev, 2,
                     PCI_BASE_ADDRESS_SPACE_MEMORY |
                         PCI_BASE_ADDRESS_MEM_TYPE_64 |
                         PCI_BASE_ADDRESS_MEM_PREFETCH,
                     &s->vram);

    /* BAR 4: doorbell registers */
    memory_region_init_io(&s->doorbell_mmio, OBJECT(s), &gpgpu_doorbell_ops, s,
                          "gpgpu-doorbell", GPGPU_DOORBELL_BAR_SIZE);
    pci_register_bar(pdev, 4,
                     PCI_BASE_ADDRESS_SPACE_MEMORY,
                     &s->doorbell_mmio);

    if (msix_init(pdev, GPGPU_MSIX_VECTORS,
                  &s->ctrl_mmio, 0, 0xFE000,
                  &s->ctrl_mmio, 0, 0xFF000,
                  0, errp))
    {
        g_free(s->vram_ptr);
        return;
    }

    msi_init(pdev, 0, 1, true, false, errp);

    s->dma_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL, gpgpu_dma_complete, s);
    s->kernel_timer = timer_new_ms(QEMU_CLOCK_VIRTUAL,
                                   gpgpu_kernel_complete, s);

    s->global_status = GPGPU_STATUS_READY;
}

static void gpgpu_exit(PCIDevice *pdev)
{
    GPGPUState *s = GPGPU(pdev);

    timer_free(s->dma_timer);
    timer_free(s->kernel_timer);
    g_free(s->vram_ptr);
    msix_uninit(pdev, &s->ctrl_mmio, &s->ctrl_mmio);
    msi_uninit(pdev);
}

static void gpgpu_reset(DeviceState *dev)
{
    GPGPUState *s = GPGPU(dev);

    s->global_ctrl = 0;
    s->global_status = GPGPU_STATUS_READY;
    s->error_status = 0;
    s->irq_enable = 0;
    s->irq_status = 0;
    memset(&s->kernel, 0, sizeof(s->kernel));
    memset(&s->dma, 0, sizeof(s->dma));
    memset(&s->simt, 0, sizeof(s->simt));
    timer_del(s->dma_timer);
    timer_del(s->kernel_timer);
    if (s->vram_ptr)
    {
        memset(s->vram_ptr, 0, s->vram_size);
    }
}

static const Property gpgpu_properties[] = {
    DEFINE_PROP_UINT32("num_cus", GPGPUState, num_cus,
                       GPGPU_DEFAULT_NUM_CUS),
    DEFINE_PROP_UINT32("warps_per_cu", GPGPUState, warps_per_cu,
                       GPGPU_DEFAULT_WARPS_PER_CU),
    DEFINE_PROP_UINT32("warp_size", GPGPUState, warp_size,
                       GPGPU_DEFAULT_WARP_SIZE),
    DEFINE_PROP_UINT64("vram_size", GPGPUState, vram_size,
                       GPGPU_DEFAULT_VRAM_SIZE),
};

static const VMStateDescription vmstate_gpgpu = {
    .name = "gpgpu",
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (const VMStateField[]){
        VMSTATE_PCI_DEVICE(parent_obj, GPGPUState),
        VMSTATE_UINT32(global_ctrl, GPGPUState),
        VMSTATE_UINT32(global_status, GPGPUState),
        VMSTATE_UINT32(error_status, GPGPUState),
        VMSTATE_UINT32(irq_enable, GPGPUState),
        VMSTATE_UINT32(irq_status, GPGPUState),
        VMSTATE_END_OF_LIST()}};

static void gpgpu_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *pc = PCI_DEVICE_CLASS(klass);

    pc->realize = gpgpu_realize;
    pc->exit = gpgpu_exit;
    pc->vendor_id = GPGPU_VENDOR_ID;
    pc->device_id = GPGPU_DEVICE_ID;
    pc->revision = GPGPU_REVISION;
    pc->class_id = GPGPU_CLASS_CODE;

    device_class_set_legacy_reset(dc, gpgpu_reset);
    dc->desc = "Educational GPGPU Device";
    dc->vmsd = &vmstate_gpgpu;
    device_class_set_props(dc, gpgpu_properties);
    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
}

static const TypeInfo gpgpu_type_info = {
    .name = TYPE_GPGPU,
    .parent = TYPE_PCI_DEVICE,
    .instance_size = sizeof(GPGPUState),
    .class_init = gpgpu_class_init,
    .interfaces = (InterfaceInfo[]){
        {INTERFACE_PCIE_DEVICE},
        {}},
};

static void gpgpu_register_types(void)
{
    type_register_static(&gpgpu_type_info);
}

type_init(gpgpu_register_types)
