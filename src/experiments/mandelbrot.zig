const std = @import("std");
const cuda = @import("../cuda.zig");
const print = std.debug.print;

const ptx_kernel =
    \\ .version 6.0
    \\ .target sm_30
    \\ .address_size 64
    \\ .visible .entry mandelbrot_kernel(
    \\     .param .u64 output_ptr,
    \\     .param .u32 width,
    \\     .param .u32 height
    \\ )
    \\ {
    \\     .reg .u32 %r<12>;
    \\     .reg .u64 %rd<4>;
    \\     .reg .f32 %f<12>;
    \\     .reg .pred %p<4>;
    \\     
    \\     mov.u32 %r0, %ctaid.x;
    \\     mov.u32 %r1, %tid.x;
    \\     mov.u32 %r2, %ntid.x;
    \\     mad.lo.u32 %r3, %r0, %r2, %r1;
    \\     
    \\     ld.param.u32 %r4, [width];
    \\     ld.param.u32 %r5, [height];
    \\     mul.lo.u32 %r6, %r4, %r5;
    \\     setp.ge.u32 %p1, %r3, %r6;
    \\     @%p1 bra DONE;
    \\     
    \\     div.u32 %r7, %r3, %r4;
    \\     rem.u32 %r8, %r3, %r4;
    \\     
    \\     cvt.rn.f32.u32 %f0, %r8;
    \\     cvt.rn.f32.u32 %f1, %r4;
    \\     div.rn.f32 %f2, %f0, %f1;
    \\     mov.f32 %f3, 0f40600000;
    \\     mul.f32 %f2, %f2, %f3;
    \\     mov.f32 %f3, 0f40200000;
    \\     sub.f32 %f2, %f2, %f3;
    \\     
    \\     cvt.rn.f32.u32 %f0, %r7;
    \\     cvt.rn.f32.u32 %f1, %r5;
    \\     div.rn.f32 %f4, %f0, %f1;
    \\     mov.f32 %f5, 0f40000000;
    \\     mul.f32 %f4, %f4, %f5;
    \\     mov.f32 %f5, 0f3f800000;
    \\     sub.f32 %f4, %f4, %f5;
    \\     
    \\     mov.u32 %r9, 0;
    \\     mov.f32 %f6, 0f00000000;
    \\     mov.f32 %f7, 0f00000000;
    \\     
    \\ LOOP:
    \\     setp.ge.u32 %p2, %r9, 256;
    \\     @%p2 bra STORE;
    \\     
    \\     mul.f32 %f8, %f6, %f6;
    \\     mul.f32 %f9, %f7, %f7;
    \\     add.f32 %f10, %f8, %f9;
    \\     mov.f32 %f11, 0f40800000;
    \\     setp.gt.f32 %p3, %f10, %f11;
    \\     @%p3 bra STORE;
    \\     
    \\     sub.f32 %f8, %f8, %f9;
    \\     add.f32 %f8, %f8, %f2;
    \\     mul.f32 %f7, %f6, %f7;
    \\     add.f32 %f7, %f7, %f7;
    \\     add.f32 %f7, %f7, %f4;
    \\     mov.f32 %f6, %f8;
    \\     
    \\     add.u32 %r9, %r9, 1;
    \\     bra LOOP;
    \\     
    \\ STORE:
    \\     ld.param.u64 %rd0, [output_ptr];
    \\     mul.wide.u32 %rd1, %r3, 4;
    \\     add.u64 %rd2, %rd0, %rd1;
    \\     st.u32 [%rd2], %r9;
    \\     
    \\ DONE:
    \\     ret;
    \\ }
;

fn savePPM(filename: []const u8, data: []const u32, width: u32, height: u32) !void {
    const file = try std.fs.cwd().createFile(filename, .{});
    defer file.close();

    var buffer: [8192]u8 = undefined;
    var writer = file.writer(&buffer);

    try writer.interface.print("P3\n{d} {d}\n255\n", .{ width, height });

    for (data) |iter| {
        const val = @min((iter * 255) / 100, 255);
        try writer.interface.print("{d} {d} {d} ", .{ val, val, val });
    }

    try writer.interface.flush();
}

pub fn run() !void {
    print("\n=== Mandelbrot Fractal Experiment ===\n", .{});
    print("[INFO] Generating Mandelbrot set on GPU\n", .{});
    print("[INFO] Resolution: 256x256\n", .{});
    print("[INFO] Max iterations: 100\n\n", .{});

    const width: u32 = 256;
    const height: u32 = 256;
    const total_pixels = width * height;

    print("[CUDA] Initializing CUDA Driver API ", .{});
    const init_result = cuda.cuInit(0);
    if (init_result != 0) {
        print("-> FAILED (code={})\n", .{init_result});
        return error.CUDAInitFailed;
    }
    print("-> SUCCESS\n", .{});

    print("[CUDA] Querying device 0 ", .{});
    var device: cuda.CUdevice = 0;
    const dev_result = cuda.cuDeviceGet(&device, 0);
    if (dev_result != 0) {
        print("-> FAILED (code={})\n", .{dev_result});
        return error.DeviceGetFailed;
    }
    print("-> SUCCESS (device={})\n", .{device});

    print("[CUDA] Creating context ", .{});
    var ctx: cuda.CUcontext = undefined;
    const ctx_result = cuda.cuCtxCreate(&ctx, 0, device);
    if (ctx_result != 0) {
        print("-> FAILED (code={})\n", .{ctx_result});
        return error.CtxCreateFailed;
    }
    print("-> SUCCESS\n", .{});
    defer _ = cuda.cuCtxDestroy(ctx);

    print("[CUDA] Loading PTX module ", .{});
    var module: cuda.CUmodule = undefined;
    const mod_result = cuda.cuModuleLoadData(&module, ptx_kernel);
    if (mod_result != 0) {
        print("-> FAILED (code={})\n", .{mod_result});
        return error.ModuleLoadFailed;
    }
    print("-> SUCCESS\n", .{});
    defer _ = cuda.cuModuleUnload(module);

    print("[CUDA] Resolving kernel function ", .{});
    var kernel: cuda.CUfunction = undefined;
    const func_result = cuda.cuModuleGetFunction(&kernel, module, "mandelbrot_kernel");
    if (func_result != 0) {
        print("-> FAILED (code={})\n", .{func_result});
        return error.GetFunctionFailed;
    }
    print("-> SUCCESS\n", .{});

    print("[CUDA] Allocating device memory ", .{});
    var dptr: cuda.CUdeviceptr = 0;
    const alloc_result = cuda.cuMemAlloc(&dptr, total_pixels * @sizeOf(u32));
    if (alloc_result != 0) {
        print("-> FAILED (code={})\n", .{alloc_result});
        return error.MemAllocFailed;
    }
    print("-> SUCCESS (ptr=0x{X})\n", .{dptr});
    defer _ = cuda.cuMemFree(dptr);

    print("[CUDA] Launching kernel (grid={d}, block=256) ", .{width});
    var dptr_mut = dptr;
    var width_mut = width;
    var height_mut = height;
    var kernel_params = [_]?*anyopaque{
        @ptrCast(@constCast(&dptr_mut)),
        @ptrCast(@constCast(&width_mut)),
        @ptrCast(@constCast(&height_mut)),
    };
    const launch_result = cuda.cuLaunchKernel(
        kernel,
        width,
        1,
        1,
        256,
        1,
        1,
        0,
        null,
        &kernel_params,
        null,
    );
    if (launch_result != 0) {
        print("-> FAILED (code={})\n", .{launch_result});
        return error.LaunchFailed;
    }
    print("-> SUCCESS\n", .{});

    print("[CUDA] Copying result from device ", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const host_data = try allocator.alloc(u32, total_pixels);
    defer allocator.free(host_data);

    const copy_result = cuda.cuMemcpyDtoH(host_data.ptr, dptr, total_pixels * @sizeOf(u32));
    if (copy_result != 0) {
        print("-> FAILED (code={})\n", .{copy_result});
        return error.MemcpyFailed;
    }
    print("-> SUCCESS\n", .{});

    print("[FILE] Saving to mandelbrot.ppm ", .{});
    try savePPM("mandelbrot.ppm", host_data, width, height);
    print("-> SUCCESS\n", .{});

    print("\n[SUCCESS] Mandelbrot generation completed\n", .{});
    print("[INFO] Open mandelbrot.ppm to view the result\n", .{});
}
