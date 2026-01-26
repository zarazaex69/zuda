const std = @import("std");
const cuda = @import("../cuda.zig");
const print = std.debug.print;

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
    const result = cuda.cuInit(0);
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.CUDAInitFailed;
    }
    print("-> SUCCESS\n", .{});
}

fn getDevice() !cuda.CUdevice {
    print("[CUDA] Querying device 0 ", .{});
    var device: cuda.CUdevice = 0;
    const result = cuda.cuDeviceGet(&device, 0);
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.DeviceGetFailed;
    }
    print("-> SUCCESS (device={})\n", .{device});
    return device;
}

fn createContext(device: cuda.CUdevice) !cuda.CUcontext {
    print("[CUDA] Creating context for device {} ", .{device});
    var ctx: cuda.CUcontext = undefined;
    const result = cuda.cuCtxCreate(&ctx, 0, device);
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.CtxCreateFailed;
    }
    print("-> SUCCESS (ctx={})\n", .{@intFromPtr(ctx)});
    return ctx;
}

fn loadModule() !cuda.CUmodule {
    print("[CUDA] Loading PTX module from memory ", .{});
    var module: cuda.CUmodule = undefined;
    const result = cuda.cuModuleLoadData(&module, ptx_kernel);
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.ModuleLoadFailed;
    }
    print("-> SUCCESS (module={})\n", .{@intFromPtr(module)});
    return module;
}

fn getKernel(module: cuda.CUmodule) !cuda.CUfunction {
    print("[CUDA] Resolving kernel function 'add_kernel' ", .{});
    var kernel: cuda.CUfunction = undefined;
    const result = cuda.cuModuleGetFunction(&kernel, module, "add_kernel");
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.GetFunctionFailed;
    }
    print("-> SUCCESS (func={})\n", .{@intFromPtr(kernel)});
    return kernel;
}

fn allocDeviceMem() !cuda.CUdeviceptr {
    print("[CUDA] Allocating {} bytes on device ", .{@sizeOf(i32)});
    var dptr: cuda.CUdeviceptr = 0;
    const result = cuda.cuMemAlloc(&dptr, @sizeOf(i32));
    if (result != 0) {
        print("-> FAILED (code={})\n", .{result});
        return error.MemAllocFailed;
    }
    print("-> SUCCESS (ptr=0x{X})\n", .{dptr});
    return dptr;
}

fn launchKernel(kernel: cuda.CUfunction, dptr: cuda.CUdeviceptr) !void {
    print("[CUDA] Launching kernel (grid=1x1x1, block=1x1x1) ", .{});
    var dptr_mut = dptr;
    var kernel_params = [_]?*anyopaque{@ptrCast(@constCast(&dptr_mut))};
    const result = cuda.cuLaunchKernel(
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

fn copyResult(dptr: cuda.CUdeviceptr) !i32 {
    print("[CUDA] Copying result from device to host ", .{});
    var host_result: i32 = 0;
    const result = cuda.cuMemcpyDtoH(&host_result, dptr, @sizeOf(i32));
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
    defer _ = cuda.cuCtxDestroy(ctx);

    const module = try loadModule();
    defer _ = cuda.cuModuleUnload(module);

    const kernel = try getKernel(module);

    const dptr = try allocDeviceMem();
    defer _ = cuda.cuMemFree(dptr);

    try launchKernel(kernel, dptr);

    return try copyResult(dptr);
}

pub fn run() !void {
    print("\n=== Math Experiment: GPU Addition ===\n", .{});
    print("[INFO] Computing 2 + 2 using CUDA Driver API\n", .{});
    print("[INFO] PTX target: sm_70 (Volta architecture)\n\n", .{});

    const result = compute2Plus2OnGPU() catch |err| {
        print("\n[FATAL] GPU computation failed: {s}\n", .{@errorName(err)});
        return err;
    };

    print("\n[SUCCESS] GPU computation completed\n", .{});
    print("[RESULT] 2 + 2 = {}\n", .{result});
}
