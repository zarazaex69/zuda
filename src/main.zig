const std = @import("std");
const print = std.debug.print;

const CUdevice = i32;
const CUcontext = *anyopaque;
const CUmodule = *anyopaque;
const CUfunction = *anyopaque;
const CUdeviceptr = u64;

extern fn cuInit(flags: u32) callconv(.c) i32;
extern fn cuDriverGetVersion(version: *i32) callconv(.c) i32;
extern fn cuDeviceGet(device: *CUdevice, ordinal: i32) callconv(.c) i32;
extern fn cuCtxCreate(pctx: *CUcontext, flags: u32, dev: CUdevice) callconv(.c) i32;
extern fn cuCtxDestroy(ctx: CUcontext) callconv(.c) i32;
extern fn cuModuleLoad(module: *CUmodule, fname: [*:0]const u8) callconv(.c) i32;
extern fn cuModuleLoadData(module: *CUmodule, image: *const anyopaque) callconv(.c) i32;
extern fn cuModuleUnload(module: CUmodule) callconv(.c) i32;
extern fn cuModuleGetFunction(hfunc: *CUfunction, hmod: CUmodule, name: [*:0]const u8) callconv(.c) i32;
extern fn cuMemAlloc(dptr: *CUdeviceptr, bytesize: u64) callconv(.c) i32;
extern fn cuMemFree(dptr: CUdeviceptr) callconv(.c) i32;
extern fn cuMemcpyHtoD(dst: CUdeviceptr, src: *const anyopaque, bytesize: u64) callconv(.c) i32;
extern fn cuMemcpyDtoH(dst: *anyopaque, src: CUdeviceptr, bytesize: u64) callconv(.c) i32;
extern fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: u32,
    gridDimY: u32,
    gridDimZ: u32,
    blockDimX: u32,
    blockDimY: u32,
    blockDimZ: u32,
    sharedMemBytes: u32,
    stream: ?*anyopaque,
    kernelParams: ?[*]?*anyopaque,
    extra: ?[*]?*anyopaque,
) callconv(.c) i32;

const ptx_kernel =
    \\ .version 8.0
    \\ .target sm_70
    \\ .address_size 64
    \\ .visible .entry add_kernel(
    \\     .param .u64 result_ptr
    \\ )
    \\ {
    \\     .reg .u32 %r1;
    \\     .reg .u64 %rd1;
    \\     mov.u32 %r1, 2;
    \\     add.u32 %r1, %r1, 2;
    \\     ld.param.u64 %rd1, [result_ptr];
    \\     st.u32 [%rd1], %r1;
    \\     ret;
    \\ }
;

fn initCuda() !void {
    print("[CUDA] Initializing CUDA Driver API ", .{});
    const result = cuInit(0);
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.CUDAInitFailed;
    }
    print("-> SUCCESS\n", .{});
}

fn getDevice() !CUdevice {
    print("[CUDA] Querying device 0 ", .{});
    var device: CUdevice = 0;
    const result = cuDeviceGet(&device, 0);
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.DeviceGetFailed;
    }
    print("-> SUCCESS (device={})\n", .{device});
    return device;
}

fn createContext(device: CUdevice) !CUcontext {
    print("[CUDA] Creating context for device {} ", .{device});
    var ctx: CUcontext = undefined;
    const result = cuCtxCreate(&ctx, 0, device);
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.CtxCreateFailed;
    }
    print("-> SUCCESS (ctx={})\n", .{@intFromPtr(ctx)});
    return ctx;
}

fn loadModule() !CUmodule {
    print("[CUDA] Loading PTX module from memory ", .{});
    var module: CUmodule = undefined;
    const result = cuModuleLoadData(&module, ptx_kernel);
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.ModuleLoadFailed;
    }
    print("-> SUCCESS (module={})\n", .{@intFromPtr(module)});
    return module;
}

fn getKernel(module: CUmodule) !CUfunction {
    print("[CUDA] Resolving kernel function 'add_kernel' ", .{});
    var kernel: CUfunction = undefined;
    const result = cuModuleGetFunction(&kernel, module, "add_kernel");
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.GetFunctionFailed;
    }
    print("-> SUCCESS (func={})\n", .{@intFromPtr(kernel)});
    return kernel;
}

fn allocDeviceMem() !CUdeviceptr {
    print("[CUDA] Allocating {} bytes on device ", .{@sizeOf(i32)});
    var dptr: CUdeviceptr = 0;
    const result = cuMemAlloc(&dptr, @sizeOf(i32));
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.MemAllocFailed;
    }
    print("-> SUCCESS (ptr=0x{X})\n", .{dptr});
    return dptr;
}

fn launchKernel(kernel: CUfunction, dptr: CUdeviceptr) !void {
    print("[CUDA] Launching kernel (grid=1x1x1, block=1x1x1) ", .{});
    var dptr_mut = dptr;
    var kernel_params = [_]?*anyopaque{@ptrCast(@constCast(&dptr_mut))};
    const result = cuLaunchKernel(
        kernel,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        null,
        &kernel_params,
        null,
    );
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.LaunchFailed;
    }
    print("-> SUCCESS\n", .{});
}

fn copyResult(dptr: CUdeviceptr) !i32 {
    print("[CUDA] Copying result from device to host ", .{});
    var host_result: i32 = 0;
    const result = cuMemcpyDtoH(&host_result, dptr, @sizeOf(i32));
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.MemcpyFailed;
    }
    print("-> SUCCESS (value={})\n", .{host_result});
    return host_result;
}

fn compute2Plus2OnGPU() !i32 {
    try initCuda();

    const device = try getDevice();
    const ctx = try createContext(device);
    defer _ = cuCtxDestroy(ctx);

    const module = try loadModule();
    defer _ = cuModuleUnload(module);

    const kernel = try getKernel(module);

    const dptr = try allocDeviceMem();
    defer _ = cuMemFree(dptr);

    try launchKernel(kernel, dptr);

    return try copyResult(dptr);
}

pub fn main() !void {
    print("CUDA on Zig - GPU Compute Demo\n", .{});
    print("[INFO] Computing 2 + 2 using CUDA Driver API\n", .{});
    print("[INFO] PTX target: sm_70 (Volta architecture)\n\n", .{});

    const result = compute2Plus2OnGPU() catch |err| {
        print("\n[FATAL] GPU computation failed: {s}\n", .{@errorName(err)});
        return err;
    };

    print("\n[SUCCESS] GPU computation completed\n", .{});
    print("[RESULT] 2 + 2 = {}\n", .{result});
}
