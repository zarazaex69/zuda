const std = @import("std");

pub const CUdevice = i32;
pub const CUcontext = *anyopaque;
pub const CUmodule = *anyopaque;
pub const CUfunction = *anyopaque;
pub const CUdeviceptr = u64;

pub extern fn cuInit(flags: u32) callconv(.c) i32;
pub extern fn cuDriverGetVersion(version: *i32) callconv(.c) i32;
pub extern fn cuDeviceGet(device: *CUdevice, ordinal: i32) callconv(.c) i32;
pub extern fn cuCtxCreate(pctx: *CUcontext, flags: u32, dev: CUdevice) callconv(.c) i32;
pub extern fn cuCtxDestroy(ctx: CUcontext) callconv(.c) i32;
pub extern fn cuModuleLoad(module: *CUmodule, fname: [*:0]const u8) callconv(.c) i32;
pub extern fn cuModuleLoadData(module: *CUmodule, image: *const anyopaque) callconv(.c) i32;
pub extern fn cuModuleUnload(module: CUmodule) callconv(.c) i32;
pub extern fn cuModuleGetFunction(hfunc: *CUfunction, hmod: CUmodule, name: [*:0]const u8) callconv(.c) i32;
pub extern fn cuMemAlloc(dptr: *CUdeviceptr, bytesize: u64) callconv(.c) i32;
pub extern fn cuMemFree(dptr: CUdeviceptr) callconv(.c) i32;
pub extern fn cuMemcpyHtoD(dst: CUdeviceptr, src: *const anyopaque, bytesize: u64) callconv(.c) i32;
pub extern fn cuMemcpyDtoH(dst: *anyopaque, src: CUdeviceptr, bytesize: u64) callconv(.c) i32;
pub extern fn cuLaunchKernel(
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
