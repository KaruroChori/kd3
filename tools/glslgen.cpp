/**
 * @file glslgen.cpp
 * @author karurochari
 * @brief CLI generator for the GLSL shader of kd3.
 * @date 2026-06-28
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <kd3/glsl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void print_usage(const char *prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "Generate GLSL shader code for kd3 tree queries.\n"
        "\n"
        "Options:\n"
        "  --prefix STR         Shader prefix (default: \"kd3\")\n"
        "  -D, --dimensions N   Dimensionality (2-4, default: 3)\n"
        "  --leaf-size N        Points per leaf (default: 32)\n"
        "  --has-payload VAL    Payload type: NONE, INDEX, OTHER (default: INDEX)\n"
        "  --max-stack-depth N  Stack depth (default: 96)\n"
        "  --simd-parallelism N SIMD parallelism (default: 1)\n"
        "  --binding-vals N     SSBO binding for split values (default: 0)\n"
        "  --binding-dims N     SSBO binding for split dimensions (default: 1)\n"
        "  --binding-bks N      SSBO binding for buckets (default: 2)\n"
        "  --aabb               Emit subtree-box pruning (requires boxes buffer)\n"
        "  --binding-boxes N    SSBO binding for subtree boxes (default: 3)\n"
        "  --max-k N            Max k for kNN queries (default: 8)\n"
        "  -o, --output PATH    Write to file instead of stdout\n"
        "  -h, --help           Print this help\n",
        prog);
}

static bool streq(const char *a, const char *b) {
    return std::strcmp(a, b) == 0;
}

static bool parse_u32(const char *s, unsigned long *out) {
    char *end = nullptr;
    unsigned long v = std::strtoul(s, &end, 10);
    if (!end || *end != '\0' || end == s) return false;
    *out = v;
    return true;
}

int main(int argc, char **argv) {
    kd3::GlslConfigDyn cfg;
    const char *output_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];

        auto next = [&]() -> const char * {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", arg);
                std::exit(1);
            }
            return argv[++i];
        };

        if (streq(arg, "-h") || streq(arg, "--help")) {
            print_usage(argv[0]);
            return 0;
        } else if (streq(arg, "--prefix")) {
            cfg.prefix = next();
        } else if (streq(arg, "-D") || streq(arg, "--dimensions")) {
            unsigned long v;
            if (!parse_u32(next(), &v) || v < 2 || v > 4) {
                std::fprintf(stderr, "Invalid dimensions (2-4)\n");
                return 1;
            }
            cfg.D = v;
        } else if (streq(arg, "--leaf-size")) {
            unsigned long v;
            if (!parse_u32(next(), &v) || v == 0) {
                std::fprintf(stderr, "Invalid leaf-size\n");
                return 1;
            }
            cfg.leaf_size = v;
        } else if (streq(arg, "--has-payload")) {
            const char *v = next();
            if (streq(v, "NONE"))       cfg.has_payload = kd3::cfg_t::has_payload_t::NONE;
            else if (streq(v, "INDEX")) cfg.has_payload = kd3::cfg_t::has_payload_t::INDEX;
            else if (streq(v, "OTHER")) cfg.has_payload = kd3::cfg_t::has_payload_t::OTHER;
            else {
                std::fprintf(stderr, "Invalid has-payload: %s (expect NONE, INDEX, or OTHER)\n", v);
                return 1;
            }
        } else if (streq(arg, "--max-stack-depth")) {
            unsigned long v;
            if (!parse_u32(next(), &v) || v == 0) {
                std::fprintf(stderr, "Invalid max-stack-depth\n");
                return 1;
            }
            cfg.max_stack_depth = v;
        } else if (streq(arg, "--simd-parallelism")) {
            unsigned long v;
            if (!parse_u32(next(), &v) || v == 0) {
                std::fprintf(stderr, "Invalid simd-parallelism\n");
                return 1;
            }
            cfg.simd_parallelism = v;
        } else if (streq(arg, "--binding-vals")) {
            unsigned long v;
            if (!parse_u32(next(), &v)) { std::fprintf(stderr, "Invalid binding-vals\n"); return 1; }
            cfg.binding_vals = static_cast<uint32_t>(v);
        } else if (streq(arg, "--binding-dims")) {
            unsigned long v;
            if (!parse_u32(next(), &v)) { std::fprintf(stderr, "Invalid binding-dims\n"); return 1; }
            cfg.binding_dims = static_cast<uint32_t>(v);
        } else if (streq(arg, "--binding-bks")) {
            unsigned long v;
            if (!parse_u32(next(), &v)) { std::fprintf(stderr, "Invalid binding-bks\n"); return 1; }
            cfg.binding_bks = static_cast<uint32_t>(v);
        } else if (streq(arg, "--binding-boxes")) {
            unsigned long v;
            if (!parse_u32(next(), &v)) { std::fprintf(stderr, "Invalid binding-boxes\n"); return 1; }
            cfg.binding_boxes = static_cast<uint32_t>(v);
        } else if (streq(arg, "--aabb")) {
            cfg.has_aabb = true;
        } else if (streq(arg, "--max-k")) {
            unsigned long v;
            if (!parse_u32(next(), &v) || v == 0) { std::fprintf(stderr, "Invalid max-k\n"); return 1; }
            cfg.max_k = static_cast<uint32_t>(v);
        } else if (streq(arg, "-o") || streq(arg, "--output")) {
            output_path = next();
        } else {
            std::fprintf(stderr, "Unknown option: %s\n", arg);
            print_usage(argv[0]);
            return 1;
        }
    }

    std::string glsl = kd3::generate_glsl_dyn(cfg);

    if (output_path) {
        FILE *f = std::fopen(output_path, "w");
        if (!f) {
            std::fprintf(stderr, "Cannot open %s for writing\n", output_path);
            return 1;
        }
        std::fwrite(glsl.data(), 1, glsl.size(), f);
        std::fclose(f);
    } else {
        std::fwrite(glsl.data(), 1, glsl.size(), stdout);
    }

    return 0;
}
