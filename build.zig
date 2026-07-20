// Zig build file, mostly used for cross-compilation
const std = @import("std");

const c_sources = [_][]const u8{
    "src/hqlc.c",   "src/mdct.c",  "src/mdct_tables.c", "src/psy.c",
    "src/quant.c",  "src/entropy.c", "src/entropy_tables.c", "src/tns.c",
    "src/hqlc_cli.c",
};

const c_flags = [_][]const u8{ "-std=c11", "-DNDEBUG" };

const ReleaseTarget = struct {
    name: []const u8,
    query: std.Target.Query,
    static: bool = false,
};

const release_targets = [_]ReleaseTarget{
    .{ .name = "linux-x86_64", .query = .{ .cpu_arch = .x86_64, .os_tag = .linux, .abi = .musl }, .static = true },
    .{ .name = "linux-aarch64", .query = .{ .cpu_arch = .aarch64, .os_tag = .linux, .abi = .musl }, .static = true },
    .{ .name = "windows-x86_64", .query = .{ .cpu_arch = .x86_64, .os_tag = .windows, .abi = .gnu } },
    .{ .name = "macos-x86_64", .query = .{ .cpu_arch = .x86_64, .os_tag = .macos } },
    .{ .name = "macos-arm64", .query = .{ .cpu_arch = .aarch64, .os_tag = .macos } },
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    b.installArtifact(makeExe(b, target, optimize, false));

    // Cross-compile to all targets on release build
    const release = b.step("release", "Cross-compile hqlc for all release targets");
    for (release_targets) |t| {
        const exe = makeExe(b, b.resolveTargetQuery(t.query), .ReleaseFast, t.static);
        const inst = b.addInstallArtifact(exe, .{
            .dest_dir = .{ .override = .{ .custom = t.name } },
        });
        release.dependOn(&inst.step);
    }
}

fn makeExe(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    static: bool,
) *std.Build.Step.Compile {
    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addCSourceFiles(.{ .files = &c_sources, .flags = &c_flags });
    mod.addIncludePath(b.path("include"));
    mod.addIncludePath(b.path("external"));

    const exe = b.addExecutable(.{ .name = "hqlc", .root_module = mod });
    if (static) exe.linkage = .static;
    return exe;
}
