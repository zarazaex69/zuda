const std = @import("std");
const print = std.debug.print;

const math = @import("experiments/math.zig");
const mandelbrot = @import("experiments/mandelbrot.zig");

const Experiment = struct {
    name: []const u8,
    description: []const u8,
    run_fn: *const fn () anyerror!void,
};

const experiments = [_]Experiment{
    .{
        .name = "math",
        .description = "Simple GPU addition (2 + 2)",
        .run_fn = math.run,
    },
    .{
        .name = "mandelbrot",
        .description = "Mandelbrot fractal generation",
        .run_fn = mandelbrot.run,
    },
};

fn listExperiments() void {
    print("Available experiments:\n\n", .{});
    for (experiments) |exp| {
        print("  {s:<15} {s}\n", .{ exp.name, exp.description });
    }
    print("\nUsage:\n", .{});
    print("  zuda ls              List all experiments\n", .{});
    print("  zuda run <name>      Run specific experiment\n", .{});
}

fn findExperiment(name: []const u8) ?*const Experiment {
    for (&experiments) |*exp| {
        if (std.mem.eql(u8, exp.name, name)) {
            return exp;
        }
    }
    return null;
}

fn runExperiment(name: []const u8) !void {
    const exp = findExperiment(name) orelse {
        print("[ERROR] Experiment '{s}' not found\n", .{name});
        print("Run 'zuda ls' to see available experiments\n", .{});
        return error.ExperimentNotFound;
    };

    try exp.run_fn();
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        print("CUDA Experiments on Zig\n\n", .{});
        listExperiments();
        return;
    }

    const cmd = args[1];

    if (std.mem.eql(u8, cmd, "ls")) {
        listExperiments();
        return;
    }

    if (std.mem.eql(u8, cmd, "run")) {
        if (args.len < 3) {
            print("[ERROR] Missing experiment name\n", .{});
            print("Usage: zuda run <name>\n", .{});
            return error.MissingArgument;
        }
        try runExperiment(args[2]);
        return;
    }

    print("[ERROR] Unknown command '{s}'\n", .{cmd});
    print("Run 'zuda ls' to see available commands\n", .{});
}
