/**
 * @file datasets.hpp
 * @brief Real-world dataset loaders, structured synthetic generators and
 *        query-set derivation for representative (non-uniform) benchmarking.
 *
 * Uniform random points are the best case for median-split kd-trees; real data
 * lives on surfaces, in clusters and in skewed densities. This header provides:
 *
 * - Loaders: canonical `.bin` float dumps, ASCII `.xyz`, binary/ASCII `.ply`,
 *   and LAS/LAZ via LASlib (only when compiled with `KD3_BENCH_LIBLAS`).
 * - Generators: Gaussian mixtures, anisotropic clouds, sphere/torus shells,
 *   multi-blob surfaces (a proxy for scanned meshes) and power-law densities.
 * - Query sets: uniform-in-volume (raymarcher-like) and sampled-from-dataset
 *   with optional jitter (kNN-graph / sampling workloads).
 *
 * Everything is deterministically seeded from workload names.
 */
#pragma once

#include <kd3/kd3.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <expected>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#ifdef KD3_BENCH_LIBLAS
#include <LASlib/lasreader.hpp>
#endif

namespace kdbench {

using Limits      = kd3::limits<float>;
using FatPoint    = Limits::FatPoint;
using PointCloud  = std::vector<FatPoint>;
using QuerySet    = std::vector<Limits::point_t>;
using LoadResult  = std::expected<PointCloud, std::string>;

/// Upper bound on loaded points, so huge files stay comparable with synthetic sizes.
inline constexpr size_t MAX_POINTS = 5'000'000;

// ---------------------------------------------------------
// Deterministic seeding
// ---------------------------------------------------------

inline size_t seed_from_name(std::string_view name) {
    uint64_t h = 14695981039346656037ull; // FNV-1a
    for (char c : name) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ull;
    }
    return static_cast<size_t>(h ^ (h >> 32));
}

// ---------------------------------------------------------
// Loading
// ---------------------------------------------------------

inline LoadResult subsample(PointCloud&& pc, size_t max_points = MAX_POINTS) {
    if (pc.size() <= max_points) return std::move(pc);
    PointCloud out;
    out.reserve(max_points);
    const double stride = static_cast<double>(pc.size()) / static_cast<double>(max_points);
    for (size_t i = 0; i < max_points; ++i) {
        out.push_back(pc[static_cast<size_t>(static_cast<double>(i) * stride)]);
    }
    return out;
}

/// Canonical fast path: raw little-endian float32 records.
/// Accepts `{float x,y,z; uint32 payload}` (16B, preferred) or plain triplets (12B).
inline LoadResult load_binary(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return std::unexpected("cannot open " + p.string());
    f.seekg(0, std::ios::end);
    const size_t bytes = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    if (bytes >= 16 && bytes % 16 == 0) {
        PointCloud pc(bytes / 16);
        f.read(reinterpret_cast<char*>(pc.data()), static_cast<std::streamsize>(bytes));
        if (!f) return std::unexpected("short read on " + p.string());
        return subsample(std::move(pc));
    }
    if (bytes != 0 && bytes % 12 == 0) {
        const size_t n = bytes / 12;
        std::vector<std::array<float, 3>> raw(n);
        f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
        if (!f) return std::unexpected("short read on " + p.string());
        PointCloud pc(n);
        for (size_t i = 0; i < n; ++i) pc[i] = {raw[i], static_cast<uint32_t>(i)};
        return subsample(std::move(pc));
    }
    return std::unexpected("file size of " + p.string() + " matches no known record layout");
}

/// Whitespace-separated `x y z [ignored...]`, one point per line, '#' comments allowed.
inline LoadResult load_xyz(const std::filesystem::path& p) {
    std::ifstream f(p);
    if (!f) return std::unexpected("cannot open " + p.string());
    PointCloud pc;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        char* end = nullptr;
        const double x = std::strtod(line.c_str(), &end);
        if (end == line.c_str()) continue;
        const double y = std::strtod(end, &end);
        const double z = std::strtod(end, &end);
        pc.push_back({{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)},
                      static_cast<uint32_t>(pc.size())});
    }
    if (pc.empty()) return std::unexpected("no points parsed from " + p.string());
    return subsample(std::move(pc));
}

namespace detail {

struct PlyProp {
    std::string name;
    size_t offset = 0;
    size_t size   = 4;
    enum Type { I8, U8, I16, U16, I32, U32, F32, F64 } type = F32;
};

inline size_t ply_type_size(const std::string& t) {
    if (t == "char" || t == "int8" || t == "uchar" || t == "uint8") return 1;
    if (t == "short" || t == "int16" || t == "ushort" || t == "uint16") return 2;
    return 4; // int/uint/int32/uint32/float/float32; doubles handled separately below
}

inline PlyProp::Type ply_type_enum(const std::string& t) {
    if (t == "char" || t == "int8") return PlyProp::I8;
    if (t == "uchar" || t == "uint8") return PlyProp::U8;
    if (t == "short" || t == "int16") return PlyProp::I16;
    if (t == "ushort" || t == "uint16") return PlyProp::U16;
    if (t == "int" || t == "int32") return PlyProp::I32;
    if (t == "uint" || t == "uint32") return PlyProp::U32;
    if (t == "double" || t == "float64") return PlyProp::F64;
    return PlyProp::F32;
}

template <typename T>
inline float ply_read_as(const unsigned char* src) {
    T v;
    std::memcpy(&v, src, sizeof(T));
    return static_cast<float>(v);
}

inline float ply_prop_value(const PlyProp& prop, const unsigned char* base) {
    switch (prop.type) {
        case PlyProp::I8:  return ply_read_as<int8_t>(base);
        case PlyProp::U8:  return ply_read_as<uint8_t>(base);
        case PlyProp::I16: return ply_read_as<int16_t>(base);
        case PlyProp::U16: return ply_read_as<uint16_t>(base);
        case PlyProp::I32: return ply_read_as<int32_t>(base);
        case PlyProp::U32: return ply_read_as<uint32_t>(base);
        case PlyProp::F64: return ply_read_as<double>(base);
        default:           return ply_read_as<float>(base);
    }
}

} // namespace detail

/// Minimal PLY reader: the first element must be `vertex`, scalar x/y/z properties,
/// ASCII or binary_little_endian. List properties after vertex data are ignored.
inline LoadResult load_ply(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return std::unexpected("cannot open " + p.string());

    std::string format_line;
    std::getline(f, format_line); // "ply"
    std::getline(f, format_line); // "format ..."
    const bool ascii     = format_line.find("ascii") != std::string::npos;
    const bool binary_le = format_line.find("binary_little_endian") != std::string::npos;
    if (!ascii && !binary_le)
        return std::unexpected("unsupported PLY format: " + format_line);

    bool in_vertex = false, found_vertex = false;
    size_t vertex_count = 0, prop_offset = 0, stride = 0;
    int xi = -1, yi = -1, zi = -1;
    std::vector<detail::PlyProp> props;

    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream iss(line);
        std::string token;
        iss >> token;
        if (token == "comment" || token == "obj_info") continue;
        if (token == "element") {
            std::string name;
            size_t count = 0;
            iss >> name >> count;
            in_vertex = (name == "vertex");
            if (in_vertex && !found_vertex) {
                found_vertex = true;
                vertex_count = count;
                prop_offset = stride = 0;
                props.clear();
                xi = yi = zi = -1;
            } else if (found_vertex) {
                break; // stop at the element following the vertex block
            }
        } else if (token == "property" && in_vertex) {
            std::string type, name;
            iss >> type >> name;
            if (type == "list")
                return std::unexpected("list property inside vertex element is not supported");
            detail::PlyProp prop;
            prop.name   = name;
            prop.type   = detail::ply_type_enum(type);
            prop.offset = prop_offset;
            prop.size   = (prop.type == detail::PlyProp::F64) ? 8 : detail::ply_type_size(type);
            if (name == "x") xi = static_cast<int>(props.size());
            if (name == "y") yi = static_cast<int>(props.size());
            if (name == "z") zi = static_cast<int>(props.size());
            prop_offset += prop.size;
            stride += prop.size;
            props.push_back(prop);
        } else if (token == "end_header") {
            break;
        }
    }

    if (!found_vertex || xi < 0 || yi < 0 || zi < 0)
        return std::unexpected("PLY is missing a vertex element with x/y/z: " + p.string());

    PointCloud pc;
    pc.reserve(vertex_count);
    if (binary_le) {
        std::vector<unsigned char> rec(stride);
        for (size_t i = 0; i < vertex_count; ++i) {
            f.read(reinterpret_cast<char*>(rec.data()), static_cast<std::streamsize>(stride));
            if (!f) break;
            const float x = detail::ply_prop_value(props[xi], rec.data());
            const float y = detail::ply_prop_value(props[yi], rec.data());
            const float z = detail::ply_prop_value(props[zi], rec.data());
            pc.push_back({{x, y, z}, static_cast<uint32_t>(i)});
        }
    } else {
        while (pc.size() < vertex_count && std::getline(f, line)) {
            std::istringstream iss(line);
            std::vector<std::string> tokens;
            std::string tok;
            while (iss >> tok) tokens.push_back(tok);
            if (tokens.size() < props.size()) continue;
            const float x = std::strtof(tokens[xi].c_str(), nullptr);
            const float y = std::strtof(tokens[yi].c_str(), nullptr);
            const float z = std::strtof(tokens[zi].c_str(), nullptr);
            pc.push_back({{x, y, z}, static_cast<uint32_t>(pc.size())});
        }
    }

    if (pc.empty()) return std::unexpected("no vertices parsed from " + p.string());
    return subsample(std::move(pc));
}

#ifdef KD3_BENCH_LIBLAS
/// LAS/LAZ point clouds (LiDAR scans), decoded through LASlib/LASzip.
/// NOTE: LASlib aborts the process on malformed input, so files are screened
/// (LASF signature + minimum size) before being handed to the reader.
inline LoadResult load_las(const std::filesystem::path& p) {
    {
        std::ifstream probe(p, std::ios::binary);
        char sig[4] = {};
        probe.read(sig, 4);
        const bool valid = static_cast<bool>(probe) &&
                           std::string_view(sig, 4) == "LASF" &&
                           std::filesystem::file_size(p) >= 512;
        if (!valid)
            return std::unexpected(p.string() + " is not a readable LAS/LAZ file "
                                   "(missing LASF signature or truncated)");
    }

    LASreadOpener opener;
    opener.set_file_name(p.string().c_str());
    LASreader* reader = opener.open();
    if (!reader) return std::unexpected("cannot open " + p.string() + " as LAS/LAZ");

    PointCloud pc;
    pc.reserve(std::min<size_t>(
        static_cast<size_t>(reader->header.get_number_of_point_records_uni()), MAX_POINTS));
    while (reader->read_point()) {
        const LASpoint& pt = reader->point;
        pc.push_back({{static_cast<float>(pt.get_x()),
                       static_cast<float>(pt.get_y()),
                       static_cast<float>(pt.get_z())},
                      static_cast<uint32_t>(pc.size())});
    }
    reader->close();
    delete reader;
    if (pc.empty()) return std::unexpected("no points parsed from " + p.string());
    return subsample(std::move(pc));
}
#endif

inline LoadResult load_dataset(const std::filesystem::path& p) {
    std::string ext = p.extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext == ".bin") return load_binary(p);
    if (ext == ".xyz" || ext == ".pts") return load_xyz(p);
    if (ext == ".ply") return load_ply(p);
#ifdef KD3_BENCH_LIBLAS
    if (ext == ".laz" || ext == ".las") return load_las(p);
#endif
    return std::unexpected("unsupported extension \"" + ext + "\" for " + p.string()
                           + " (build with --with_evaluation to enable LAS/LAZ)");
}

// ---------------------------------------------------------
// Structured synthetic generators
// ---------------------------------------------------------

using RNG = std::mt19937;

inline float uniform(RNG& gen, float lo, float hi) {
    return std::uniform_real_distribution<float>(lo, hi)(gen);
}
inline float gauss(RNG& gen, float sigma = 1.0f) {
    return std::normal_distribution<float>(0.0f, sigma)(gen);
}
inline Limits::point_t gauss3(RNG& gen, float sigma = 1.0f) {
    return {gauss(gen, sigma), gauss(gen, sigma), gauss(gen, sigma)};
}

/// Multi-scale clusters: dense blobs of varying weight and spread. The classic
/// failure mode of spatial indexes built on balanced splits.
inline PointCloud gaussian_mixture(size_t n, size_t seed) {
    RNG gen(static_cast<uint32_t>(seed));
    constexpr size_t K = 64;
    std::array<Limits::point_t, K> centers{};
    std::array<float, K> sigmas{}, cum{};
    float total = 0.0f;
    for (size_t k = 0; k < K; ++k) {
        centers[k] = {uniform(gen, -900.f, 900.f), uniform(gen, -900.f, 900.f),
                      uniform(gen, -900.f, 900.f)};
        sigmas[k] = std::exp(uniform(gen, std::log(2.0f), std::log(60.0f)));
        total += std::exp(uniform(gen, -2.5f, 0.0f));
        cum[k] = total;
    }
    for (auto& c : cum) c /= total;

    PointCloud pc(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t k = static_cast<size_t>(
            std::upper_bound(cum.begin(), cum.end(), uniform(gen, 0.f, 1.f)) - cum.begin());
        const size_t kk = std::min(k, K - 1);
        const Limits::point_t g = gauss3(gen, sigmas[kk]);
        pc[i] = {{centers[kk][0] + g[0], centers[kk][1] + g[1], centers[kk][2] + g[2]},
                 static_cast<uint32_t>(i)};
    }
    return pc;
}

/// One strongly elongated cloud in a random orientation; stresses axis-aligned splits.
inline PointCloud anisotropic_gaussian(size_t n, size_t seed) {
    RNG gen(static_cast<uint32_t>(seed));
    const Limits::point_t scales = {600.0f, 40.0f, 8.0f};
    const float ax = uniform(gen, 0.f, 3.14f), ay = uniform(gen, 0.f, 3.14f),
                az = uniform(gen, 0.f, 3.14f);
    const float cx = std::cos(ax), sx = std::sin(ax);
    const float cy = std::cos(ay), sy = std::sin(ay);
    const float cz = std::cos(az), sz = std::sin(az);

    PointCloud pc(n);
    for (size_t i = 0; i < n; ++i) {
        Limits::point_t v = {gauss(gen, scales[0]), gauss(gen, scales[1]), gauss(gen, scales[2])};
        float y = v[1] * cx - v[2] * sx; // rot X
        float z = v[1] * sx + v[2] * cx;
        float x = v[0] * cy + z * sy;    // rot Y
        z = -v[0] * sy + z * cy;
        const float x2 = x * cz - y * sz; // rot Z
        y = x * sz + y * cz;
        pc[i] = {{x2, y, z}, static_cast<uint32_t>(i)};
    }
    return pc;
}

inline Limits::point_t normalized_gauss3(RNG& gen) {
    Limits::point_t d{};
    float len = 0.0f;
    do {
        d = gauss3(gen);
        len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    } while (len < 1e-6f);
    return {d[0] / len, d[1] / len, d[2] / len};
}

/// A thin spherical shell: the simplest 2-manifold, like a closed scan surface.
inline PointCloud sphere_shell(size_t n, size_t seed) {
    RNG gen(static_cast<uint32_t>(seed));
    constexpr float RADIUS = 800.0f, THICKNESS = 4.0f;
    PointCloud pc(n);
    for (size_t i = 0; i < n; ++i) {
        const Limits::point_t dir = normalized_gauss3(gen);
        const float r = RADIUS + gauss(gen, THICKNESS);
        pc[i] = {{dir[0] * r, dir[1] * r, dir[2] * r}, static_cast<uint32_t>(i)};
    }
    return pc;
}

/// Torus surface: curvature in two directions, empty interior volume.
inline PointCloud torus(size_t n, size_t seed) {
    RNG gen(static_cast<uint32_t>(seed));
    constexpr float R = 700.0f, r = 120.0f;
    PointCloud pc(n);
    for (size_t i = 0; i < n; ++i) {
        const float theta = uniform(gen, 0.f, 6.2831853f);
        const float phi   = uniform(gen, 0.f, 6.2831853f);
        const float w     = R + r * std::cos(phi);
        pc[i] = {{w * std::cos(theta) + gauss(gen, 2.0f),
                  w * std::sin(theta) + gauss(gen, 2.0f),
                  r * std::sin(phi) + gauss(gen, 2.0f)},
                 static_cast<uint32_t>(i)};
    }
    return pc;
}

/// Union of many spheres sampled proportionally to surface area: a stand-in for
/// multi-object mesh/scanned scenes (cf. the SDF surflets of render.raymarch).
inline PointCloud sdf_blobs(size_t n, size_t seed) {
    RNG gen(static_cast<uint32_t>(seed));
    constexpr size_t M = 24;
    std::array<Limits::point_t, M> centers{};
    std::array<float, M> radii{}, cum{};
    float total = 0.0f;
    for (size_t m = 0; m < M; ++m) {
        centers[m] = {uniform(gen, -800.f, 800.f), uniform(gen, -800.f, 800.f),
                      uniform(gen, -800.f, 800.f)};
        radii[m] = std::exp(uniform(gen, std::log(15.0f), std::log(130.0f)));
        total += radii[m] * radii[m];
        cum[m] = total;
    }
    for (auto& c : cum) c /= total;

    PointCloud pc(n);
    for (size_t i = 0; i < n; ++i) {
        const size_t k = std::min(
            static_cast<size_t>(
                std::upper_bound(cum.begin(), cum.end(), uniform(gen, 0.f, 1.f)) - cum.begin()),
            M - 1);
        const Limits::point_t dir = normalized_gauss3(gen);
        const float r = radii[k] * (1.0f + gauss(gen, 0.01f));
        pc[i] = {{centers[k][0] + dir[0] * r,
                  centers[k][1] + dir[1] * r,
                  centers[k][2] + dir[2] * r},
                 static_cast<uint32_t>(i)};
    }
    return pc;
}

/// Solid uniform cube: the classic benchmark distribution, fills its own bbox
/// and keeps every query near data. The regime kd-trees handle best.
inline PointCloud uniform_solid(size_t n, size_t seed) {
    RNG gen(static_cast<uint32_t>(seed));
    PointCloud pc(n);
    for (size_t i = 0; i < n; ++i) {
        pc[i] = {{uniform(gen, -1000.f, 1000.f), uniform(gen, -1000.f, 1000.f),
                  uniform(gen, -1000.f, 1000.f)},
                 static_cast<uint32_t>(i)};
    }
    return pc;
}

/// Power-law radial density (dense core, sparse halo): star-catalog-like skew.
inline PointCloud power_law_radial(size_t n, size_t seed) {
    RNG gen(static_cast<uint32_t>(seed));
    constexpr float ALPHA = 2.5f, R_MIN = 8.0f, R_MAX = 1600.0f;
    const float exp_ = -1.0f / (ALPHA - 1.0f);
    PointCloud pc(n);
    for (size_t i = 0; i < n; ++i) {
        const Limits::point_t dir = normalized_gauss3(gen);
        const float r = std::clamp(R_MIN * std::pow(uniform(gen, 0.f, 1.f), exp_), R_MIN, R_MAX);
        pc[i] = {{dir[0] * r, dir[1] * r, dir[2] * r}, static_cast<uint32_t>(i)};
    }
    return pc;
}

// ---------------------------------------------------------
// Bounding box & query-set derivation
// ---------------------------------------------------------

struct BBox {
    Limits::point_t min{}, max{};
    float diagonal() const {
        const float dx = max[0] - min[0], dy = max[1] - min[1], dz = max[2] - min[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
};

inline BBox compute_bbox(const PointCloud& pc) {
    BBox box{Limits::point_t{1e30f, 1e30f, 1e30f}, Limits::point_t{-1e30f, -1e30f, -1e30f}};
    for (const auto& p : pc) {
        for (size_t d = 0; d < 3; ++d) {
            box.min[d] = std::min(box.min[d], p.coords[d]);
            box.max[d] = std::max(box.max[d], p.coords[d]);
        }
    }
    return box;
}

/// Queries spread over the whole bounding-box volume: raymarcher-like access.
inline QuerySet queries_uniform_in_bbox(const PointCloud& pc, size_t m, size_t seed) {
    const BBox box = compute_bbox(pc);
    RNG gen(static_cast<uint32_t>(seed));
    QuerySet qs(m);
    for (auto& q : qs) {
        for (size_t d = 0; d < 3; ++d)
            q[d] = uniform(gen, box.min[d], box.max[d]);
    }
    return qs;
}

/// Queries drawn from the dataset itself with a small gaussian jitter
/// (fraction of the bbox diagonal): kNN-graph / sampling / denoising access.
inline QuerySet queries_from_dataset(const PointCloud& pc, size_t m, size_t seed,
                                     float jitter_fraction = 0.002f) {
    const BBox box = compute_bbox(pc);
    const float jitter_scale = jitter_fraction * box.diagonal();
    RNG gen(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<size_t> pick(0, pc.size() - 1);
    QuerySet qs(m);
    for (auto& q : qs) {
        const auto& p = pc[pick(gen)].coords;
        for (size_t d = 0; d < 3; ++d)
            q[d] = p[d] + gauss(gen, jitter_scale);
    }
    return qs;
}

} // namespace kdbench
