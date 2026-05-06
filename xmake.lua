add_requires("openmp")
add_rules("plugin.compile_commands.autoupdate")
add_rules("mode.debug", "mode.release")
set_languages("cxx26","c23")

target("kd3")
    set_kind("static")
    add_cxxflags("-fno-exceptions","-fno-unwind-tables","-fno-rtti","-nostdlib++")
    --add_ldflags("-nostdlib++")
    add_includedirs("include", {public=true})    
    add_headerfiles("include/kd3/(**)", {prefixdir = ""})
    add_files("lib/kd3/**.cpp")
    add_packages("openmp")
    if is_mode("release") then
        set_optimize("fastest")
        -- Instruct the compiler to use AVX and fast-math to ensure auto-vectorization
        add_cxflags("-ffast-math", "-march=native", {tools = {"gcc", "clang"}})
        add_cxflags("/fp:fast", "/arch:AVX2", {tools = {"msvc"}})
    end

target("benchmarks")
    set_kind("binary")
    add_files("examples/benchmarks.cpp")
    add_includedirs("include", {public=true})    
    add_packages("openmp")
    if is_mode("release") then
        set_optimize("fastest")
        -- Instruct the compiler to use AVX and fast-math to ensure auto-vectorization
        add_cxflags("-ffast-math", "-march=native", {tools = {"gcc", "clang"}})
        add_cxflags("/fp:fast", "/arch:AVX2", {tools = {"msvc"}})
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
        add_cxflags("-ffast-math", "-march=native", {tools = {"gcc", "clang"}})
        add_cxflags("/fp:fast", "/arch:AVX2", {tools = {"msvc"}})
    end
    set_default(false)