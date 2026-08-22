local version = os.getenv("KD3_VERSION") or "1.3.0"
local major, minor, patch = version:match("^(%d+)%.?(%d*)%.?(%d*)")
set_version(version)
set_license("AGPL-3.0-only")

add_requires("openmp")
add_rules("plugin.compile_commands.autoupdate")
add_rules("mode.debug", "mode.release")
set_languages("cxx23","c23")

target("kd3")
    set_kind("static")
    add_cxxflags("-fno-exceptions","-fno-unwind-tables","-fno-rtti","-nostdlib++")
    --add_ldflags("-nostdlib++")
    add_includedirs("include", {public=true})    
    add_headerfiles("include/(**)", {prefixdir = ""})
    add_files("lib/kd3/**.cpp")
    add_packages("openmp")
    if is_mode("release") then
        set_optimize("fastest")
        -- Instruct the compiler to use AVX and fast-math to ensure auto-vectorization
        add_cxflags("-ffast-math", "-march=native")
    end
    before_build(function (target)
        local ver = version
        local ma, mi, pa = ver:match("^(%d+)%.?(%d*)%.?(%d*)")
        local f = io.open(path.join(os.projectdir(), "include/kd3/version.h"), "w")
        f:write("#pragma once\n\n")
        f:write("#define KD3_VERSION_MAJOR " .. (ma or "0") .. "\n")
        f:write("#define KD3_VERSION_MINOR " .. (mi or "0") .. "\n")
        f:write("#define KD3_VERSION_PATCH " .. (pa or "0") .. "\n")
        f:write("#define KD3_VERSION_STRING \"" .. ver .. "\"\n")
        f:close()
    end)

target("benchmarks.fast")
    set_kind("binary")
    add_files("benchmarks/benchmarks.fast.cpp")
    add_includedirs("include", {public=true})    
    add_packages("openmp")
    if is_mode("release") then
        set_optimize("fastest")
        -- Instruct the compiler to use AVX and fast-math to ensure auto-vectorization
        add_cxflags("-ffast-math", "-march=native")
    end
    set_default(false)

target("c-interface")
    set_kind("binary")
    add_files("examples/c-interface.c")
    add_includedirs("include", {public=true})    
    add_packages("openmp")
    add_deps("kd3")
    if is_mode("release") then
        set_optimize("fastest")
        -- Instruct the compiler to use AVX and fast-math to ensure auto-vectorization
        add_cxflags("-ffast-math", "-march=native")
    end
    set_default(false)

target("plot")
    set_kind("binary")
    add_files("benchmarks/plot.cpp")
    add_includedirs("include", {public=true})    
    add_packages("openmp")
    if is_mode("release") then
        set_optimize("fastest")
        -- Instruct the compiler to use AVX and fast-math to ensure auto-vectorization
        add_cxflags("-ffast-math", "-march=native")
    end
    set_default(false)

target("kd3-glslgen")
    set_kind("binary")
    add_files("tools/glslgen.cpp")
    add_includedirs("include", {public=true})
    if is_mode("release") then
        set_optimize("fastest")
        add_cxflags("-ffast-math", "-march=native")
    end
    set_default(false)

-- Option that allows to build the UI demos, disabled by default to avoid unwanted extra dependencies
option("with_demo")
    set_default(false)
    set_showmenu(true)
    set_category("Build Options")
    set_description("Build the GPU/CPU graphical raymarching demo")
option_end()

-- Option that enables the dataset-driven evaluation suite: fetch-datasets and
-- benchmarks.datasets. Kept behind a flag so heavyweight dependencies (e.g. LASlib/LASzip)
-- never affect regular users.
option("with_evaluation")
    set_default(false)
    set_showmenu(true)
    set_category("Build Options")
    set_description("Build the real-world-dataset benchmarking suite (fetch-datasets, benchmarks.datasets)")
option_end()

if has_config("with_evaluation") then
    add_requires("lastools", { configs = { shared = false } })
end

if has_config("with_demo") then
    add_requires("raylib", { configs = { cxflags = "-DGRAPHICS_API_OPENGL_43" } })
    add_requires("glm")
    add_requires("nanoflann")
end

if has_config("with_evaluation") then
    includes("tools/fetch-datasets.lua")

    target("benchmarks.comparative")
        set_kind("binary")
        add_files("benchmarks/benchmarks.comparative.cpp")
        add_includedirs("include", {public=true})
        add_includedirs("benchmarks")
        add_packages("openmp")
        add_packages("lastools")
        -- Static-linked for full optimization; this evaluation-only binary is not
        -- distributed as part of the library, so we assert license compatibility here.
        set_policy("check.target_package_licenses", false)
        add_defines("KD3_BENCH_LIBLAS")
        if has_config("with_demo") then
            add_packages("nanoflann")
            add_defines("KD3_BENCH_NANOFLANN")
        end
        if is_mode("release") then
            set_optimize("fastest")
            add_cxflags("-ffast-math", "-march=native")
        end
        -- `xmake run` would otherwise start the binary from a build directory,
        -- hiding the datasets/ folder next to the project.
        set_rundir(os.projectdir())
        set_default(false)
end

if has_config("with_demo") then

target("render.raymarch")
    set_kind("binary")
    add_files("./examples/render.raymarch.cpp")
    add_deps("kd3")
    add_packages("raylib")
    add_packages("openmp")
    if is_mode("release") then
        set_optimize("fastest")
        -- Instruct the compiler to use AVX and fast-math to ensure auto-vectorization
        add_cxflags( "-march=native")
    end

target("render.raytrace")
    set_kind("binary")
    add_files("./examples/render.raytrace.cpp")
    add_deps("kd3")
    add_packages("raylib")
    add_packages("openmp")
    if is_mode("release") then
        set_optimize("fastest")
        -- Instruct the compiler to use AVX and fast-math to ensure auto-vectorization
        add_cxflags( "-march=native")
    end

end
