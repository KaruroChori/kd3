/**
 * @file benchmarks.comparative.cpp
 * @brief Dataset-driven benchmarking of kd3 against structured synthetic
 *        distributions and real-world point clouds (see tools/fetch-datasets.lua).
 *
 * Every cloud is measured twice - classic median-split tree and the optional
 * subtree-AABB variant (cfg_t::has_aabb) - under two query distributions:
 *   - bbox-volume : queries uniform inside the cloud bounding box (raymarch-like access)
 *   - on-cloud    : queries sampled from the cloud itself, small jitter (kNN-graph-like access)
 *
 * All k-NN rows are validated against brute force. nanoflann (when compiled in
 * via --with_demo) has no notion of kd3's AABB flag, so it is measured once per
 * cloud/distribution and reported on every row as a fixed reference column.
 * Build behind `xmake f --with_evaluation=true`.
 */

#include "datasets.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#ifdef KD3_BENCH_NANOFLANN
#include <nanoflann.hpp>
#endif

using TreeType     = kd3::KdTree<kd3::limits<float>, {.leaf_size = 32}>;
using TreeTypeAabb = kd3::KdTree<kd3::limits<float>, {.leaf_size = 32, .has_aabb = true}>;

namespace {

struct Options {
    size_t n_queries = 100'000;
    size_t k         = 10;
    std::string datasets_dir_arg;
};

Options parse_options(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> const char* {
            const size_t eq = arg.find('=');
            return eq == std::string::npos ? "" : arg.c_str() + eq + 1;
        };
        if (arg.rfind("--queries=", 0) == 0) opts.n_queries = std::strtoull(value(), nullptr, 10);
        else if (arg.rfind("--knn=", 0) == 0) opts.k = std::strtoull(value(), nullptr, 10);
        else if (arg.rfind("--dir=", 0) == 0) opts.datasets_dir_arg = value();
        else if (arg == "--aabb")
            std::cout << "[note] --aabb ignored: both configurations are always measured.\n";
    }
    if (opts.n_queries == 0 || opts.k == 0 || opts.k > 128) {
        std::cerr << "Invalid options: queries and knn must be > 0, knn <= 128\n";
        std::exit(1);
    }
    return opts;
}

/// The datasets folder may not be the process working directory when launched
/// through `xmake run` or from a build tree, so resolution order is:
/// explicit --dir -> $KD3_DATASETS_DIR -> ./datasets -> ancestors of the executable.
std::filesystem::path resolve_datasets_dir(const Options& opts, const char* argv0) {
    if (!opts.datasets_dir_arg.empty()) return opts.datasets_dir_arg;
    if (const char* env = std::getenv("KD3_DATASETS_DIR"); env && *env) return env;
    std::error_code ec;
    if (std::filesystem::exists("datasets", ec)) return "datasets";
    if (argv0 && *argv0) {
        if (auto exe = std::filesystem::canonical(argv0, ec); !ec) {
            auto dir = exe.parent_path();
            for (int depth = 0; depth < 6 && !dir.empty(); ++depth) {
                const auto candidate = dir / "datasets";
                if (std::filesystem::exists(candidate, ec)) return candidate;
                dir = dir.parent_path();
            }
        }
    }
    return "datasets";
}

using LimitsF = kd3::limits<float>;

std::vector<LimitsF::KnnResult> linear_scan(const float* target, size_t k,
                                            std::span<const LimitsF::FatPoint> points) {
    std::vector<LimitsF::KnnResult> heap;
    heap.reserve(k + 1);
    for (const auto& p : points) {
        const float dx = target[0] - p.coords[0];
        const float dy = target[1] - p.coords[1];
        const float dz = target[2] - p.coords[2];
        const float dist_sq = dx * dx + dy * dy + dz * dz;
        if (heap.size() < k) {
            heap.push_back({dist_sq, p.payload_id});
            std::push_heap(heap.begin(), heap.end());
        } else if (dist_sq < heap.front().dist_sq) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {dist_sq, p.payload_id};
            std::push_heap(heap.begin(), heap.end());
        }
    }
    std::sort(heap.begin(), heap.end());
    return heap;
}

double seconds_since(std::chrono::high_resolution_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();
}

// ---------------------------------------------------------
// Result collection & formatting
// ---------------------------------------------------------

struct Row {
    std::string dataset, source, distribution;
    bool aabb = false;
    size_t n_points = 0, n_queries = 0, k = 0;
    double build_s = 0.0, kd3_s = 0.0, kd3_qps = 0.0;
    double nf_build_s = -1.0, nf_s = -1.0, nf_qps = -1.0;
    /// -1 = not applicable, 0 = failed, 1 = passed.
    int kd3_ok = -1, nf_ok = -1;
};

std::string thousands(uint64_t v) {
    std::string s = std::to_string(v);
    int insert = static_cast<int>(s.length()) - 3;
    while (insert > 0) {
        s.insert(static_cast<size_t>(insert), ",");
        insert -= 3;
    }
    return s;
}

template <typename T>
std::string fixed(T value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

const char* ok_label(int v) { return v < 0 ? "n/a" : (v ? "YES" : "NO"); }

void write_csv_header(std::ofstream& csv) {
    csv << "dataset,source,distribution,aabb,n_points,n_queries,k,"
           "build_s,kd3_s,kd3_qps,nf_build_s,nf_s,nf_qps,kd3_ok,nf_ok\n";
}

/// Rows are appended as soon as they exist (and flushed per cloud) so that
/// interrupted runs still leave usable partial reports behind.
void append_csv_row(std::ofstream& csv, const Row& r) {
    csv << r.dataset << ',' << r.source << ',' << r.distribution << ',' << (r.aabb ? 1 : 0)
        << ',' << r.n_points << ',' << r.n_queries << ',' << r.k << ',' << r.build_s << ','
        << r.kd3_s << ',' << r.kd3_qps << ',' << r.nf_build_s << ',' << r.nf_s << ','
        << r.nf_qps << ',' << r.kd3_ok << ',' << r.nf_ok << '\n';
}

#ifdef KD3_BENCH_NANOFLANN
constexpr bool kHasNanoflann = true;
#else
constexpr bool kHasNanoflann = false;
#endif

/// Aligned summary table; one row per (cloud x configuration x query distribution).
void print_table(const std::vector<Row>& rows) {
    const int W_CLOUD = 24, W_SRC = 10, W_PTS = 11, W_DIST = 13, W_CFG = 8;
    const int W_MS = 10, W_QPS = 12;

    const auto rule = [&] {
        std::cout << std::string(W_CLOUD + W_SRC + W_PTS + W_DIST + W_CFG +
                                 ((kHasNanoflann ? 6 : 3) * (W_MS + W_QPS + 1)) + 24, '-')
                  << "\n";
    };

    std::cout << "\n";
    rule();
    std::cout << std::left << std::setw(W_CLOUD) << "cloud" << std::setw(W_SRC) << "source"
              << std::right << std::setw(W_PTS) << "points" << "  " << std::left
              << std::setw(W_DIST) << "query-set" << std::setw(W_CFG) << "config" << std::right
              << std::setw(W_MS) << "build_ms" << std::setw(W_MS) << "kd3_ms" << std::setw(W_QPS)
              << "kd3_q/s";
    if constexpr (kHasNanoflann) {
        std::cout << std::setw(W_MS) << "nf_bld_ms" << std::setw(W_MS) << "nf_ms"
                  << std::setw(W_QPS) << "nf_q/s";
    }
    std::cout << "  ok\n";
    rule();

    for (const auto& r : rows) {
        std::cout << std::left << std::setw(W_CLOUD) << r.dataset << std::setw(W_SRC) << r.source
                  << std::right << std::setw(W_PTS) << thousands(r.n_points) << "  " << std::left
                  << std::setw(W_DIST) << r.distribution << std::setw(W_CFG)
                  << (r.aabb ? "aabb" : "base") << std::right << std::setw(W_MS)
                  << fixed(r.build_s * 1000.0, 1) << std::setw(W_MS) << fixed(r.kd3_s * 1000.0, 1)
                  << std::setw(W_QPS) << thousands(static_cast<uint64_t>(r.kd3_qps));
        if constexpr (kHasNanoflann) {
            const bool has_nf = r.nf_qps >= 0;
            std::cout << std::setw(W_MS)
                      << (has_nf ? fixed(r.nf_build_s * 1000.0, 1) : std::string("-"))
                      << std::setw(W_MS) << (has_nf ? fixed(r.nf_s * 1000.0, 1) : std::string("-"))
                      << std::setw(W_QPS)
                      << (has_nf ? thousands(static_cast<uint64_t>(r.nf_qps)) : std::string("-"));
            std::cout << "  " << ok_label(r.kd3_ok) << "/" << ok_label(r.nf_ok);
        } else {
            std::cout << "  " << ok_label(r.kd3_ok);
        }
        std::cout << "\n";
    }
    rule();
}

// ---------------------------------------------------------
// Benchmarking core
// ---------------------------------------------------------

#ifdef KD3_BENCH_NANOFLANN
template <typename TreeType>
struct Kd3PointAdaptor {
    const std::vector<typename TreeType::FatPoint>& pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx].coords[dim];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }
};
template <typename TreeType>
using NanoflannTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, Kd3PointAdaptor<TreeType>>, Kd3PointAdaptor<TreeType>, 3,
    uint32_t>;
#endif

/// Tie-tolerant k-NN comparison: real-world clouds often contain duplicated or
/// equidistant points, so result ORDER between implementations is not defined.
/// We require the sorted squared-distance sequences to agree within a small
/// relative tolerance (absorbing FP reassociation from SIMD/-ffast-math).
bool knn_results_match(std::span<const LimitsF::KnnResult> got,
                       const std::vector<LimitsF::KnnResult>& brute) {
    if (got.size() != brute.size()) return false;
    for (size_t j = 0; j < brute.size(); ++j) {
        const float tol = 1e-4f * std::max(1.0f, brute[j].dist_sq);
        if (std::fabs(got[j].dist_sq - brute[j].dist_sq) > tol) return false;
    }
    return true;
}

template <typename TreeType>
bool validate_against_brute_force(const TreeType& tree, const kdbench::QuerySet& queries,
                                  const std::vector<typename TreeType::FatPoint>& reference,
                                  const Options& opts) {
    const size_t checks = std::min<size_t>(queries.size(), 100);
    for (size_t c = 0; c < checks; ++c) {
        const auto& q = queries[(c * 997) % queries.size()];
        std::vector<typename TreeType::KnnResult> buffer(opts.k);
        const auto got = tree.query_knn(q, buffer);
        if (!got) return false;
        if (!knn_results_match(*got, linear_scan(q.data(), opts.k, reference))) return false;
    }
    return true;
}

/// nanoflann has no kd3-config dependence: measured once per cloud and stamped
/// on every row so base and aabb rows stay directly comparable.
struct NfData {
    bool present = false;
    double build_s = -1.0;
    /// Indexed like the shared workload list (bbox-volume, on-cloud).
    struct Dist {
        double s = -1.0, qps = -1.0;
        int ok = -1;
    };
    Dist dists[2] = {};
};

void measure_nanoflann(const kdbench::PointCloud& reference,
                       const std::array<std::pair<const char*, kdbench::QuerySet>, 2>& workloads,
                       const Options& opts, NfData& out) {
#if defined(KD3_BENCH_NANOFLANN)
    Kd3PointAdaptor<TreeType> adaptor(reference);
    NanoflannTree<TreeType> nf_tree(3, adaptor, {10});
    auto t0 = std::chrono::high_resolution_clock::now();
    nf_tree.buildIndex();
    out.present = true;
    out.build_s = seconds_since(t0);
    std::cout << "[nanoflann] build " << fixed(out.build_s * 1000.0, 1) << " ms\n";

    for (size_t w = 0; w < workloads.size(); ++w) {
        std::vector<uint32_t> nf_indices(opts.k);
        std::vector<float> nf_dists(opts.k);
        volatile uint64_t sink = 0;
        t0 = std::chrono::high_resolution_clock::now();
        for (const auto& q : workloads[w].second) {
            nf_tree.knnSearch(q.data(), opts.k, nf_indices.data(), nf_dists.data());
            sink += nf_indices[0];
        }
        out.dists[w].s   = seconds_since(t0);
        out.dists[w].qps = static_cast<double>(workloads[w].second.size()) / out.dists[w].s;

        bool ok = true;
        const size_t checks = std::min<size_t>(workloads[w].second.size(), 100);
        for (size_t c = 0; c < checks; ++c) {
            const auto& q = workloads[w].second[(c * 997) % workloads[w].second.size()];
            nf_tree.knnSearch(q.data(), opts.k, nf_indices.data(), nf_dists.data());
            const auto brute = linear_scan(q.data(), opts.k, reference);
            for (size_t j = 0; j < opts.k; ++j) {
                if (nf_indices[j] != brute[j].payload_id &&
                    nf_dists[j] != brute[j].dist_sq) { ok = false; break; }
            }
            if (!ok) break;
        }
        out.dists[w].ok = ok ? 1 : 0;
        std::cout << "[nanoflann/" << workloads[w].first << "] "
                  << fixed(out.dists[w].s * 1000.0, 1) << " ms "
                  << thousands(static_cast<uint64_t>(out.dists[w].qps)) << " q/s | ok "
                  << ok_label(out.dists[w].ok) << "\n";
    }
#else
    (void)reference;
    (void)workloads;
    (void)opts;
    (void)out;
#endif
}

/// Runs one kd3 configuration over the shared workload sets.
template <typename TreeType>
void run_config(const std::string& name, const std::string& source,
                kdbench::PointCloud&& raw_points,
                const std::array<std::pair<const char*, kdbench::QuerySet>, 2>& workloads,
                const kdbench::PointCloud& reference_cloud, const NfData& nf,
                const Options& opts, std::vector<Row>& rows) {
    constexpr bool kIsAabb = TreeType::cfg.has_aabb;
    const char* cfg_name = kIsAabb ? "aabb" : "base";

    Row base_row;
    base_row.dataset   = name;
    base_row.source    = source;
    base_row.aabb      = kIsAabb;
    base_row.n_points  = raw_points.size();
    base_row.n_queries = opts.n_queries;
    base_row.k         = opts.k;

    std::vector<typename TreeType::FatPoint> points_copy = raw_points;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tree_result = TreeType::build(raw_points);
    base_row.build_s = seconds_since(t0);
    if (!tree_result) {
        std::cerr << "kd3 build failed!\n";
        std::exit(1);
    }
    const auto& tree = *tree_result;
    std::cout << "[" << cfg_name << "] build kd3 " << fixed(base_row.build_s * 1000.0, 1)
              << " ms\n";

    for (size_t w = 0; w < workloads.size(); ++w) {
        Row row = base_row;
        row.distribution = workloads[w].first;
        const auto& queries = workloads[w].second;

        std::vector<typename TreeType::KnnResult> buffer(opts.k);
        volatile uint64_t sink = 0;

        t0 = std::chrono::high_resolution_clock::now();
        for (const auto& q : queries) {
            const auto res = tree.query_knn_inline(q, buffer);
            sink += res->front().payload_id;
        }
        row.kd3_s   = seconds_since(t0);
        row.kd3_qps = static_cast<double>(queries.size()) / row.kd3_s;
        row.kd3_ok  = validate_against_brute_force(tree, queries, points_copy, opts) ? 1 : 0;

        if (nf.present) {
            row.nf_build_s = nf.build_s;
            row.nf_s       = nf.dists[w].s;
            row.nf_qps     = nf.dists[w].qps;
            row.nf_ok      = nf.dists[w].ok;
        }

        std::cout << "[" << cfg_name << "/" << row.distribution << "] kd3 "
                  << fixed(row.kd3_s * 1000.0, 1) << " ms "
                  << thousands(static_cast<uint64_t>(row.kd3_qps)) << " q/s | ok "
                  << ok_label(row.kd3_ok);
        if (nf.present)
            std::cout << " | nanoflann " << fixed(row.nf_s * 1000.0, 1) << " ms "
                      << thousands(static_cast<uint64_t>(row.nf_qps)) << " q/s";
        std::cout << "\n";
        rows.push_back(row);
    }
}

} // namespace

int main(int argc, char** argv) {
    const Options opts = parse_options(argc, argv);
    const std::filesystem::path datasets_dir = resolve_datasets_dir(opts, argc > 0 ? argv[0] : "");

    std::cout << "kd3 dataset benchmark  [simd: " << TreeType::cfg.simd_parallelism
              << "  leaf_size: " << TreeType::cfg.leaf_size << "  configs: base+aabb"
              << "  k: " << opts.k << "  queries/row: " << opts.n_queries << "]\n";
    std::cout << "datasets dir: " << datasets_dir.string() << "\n\n";
    std::cout << "clouds  : every dataset file found above, plus fixed synthetic generators;\n"
                 "          each cloud indexes identically for all compared implementations.\n"
                 "queries : bbox-volume = uniform inside the cloud bounding box (raymarch-like)\n"
                 "          on-cloud    = sampled from the cloud itself, ~0.2% jitter (kNN-like)\n"
                 "ok      : k-NN results match brute force within tie/fp tolerance\n\n";

    // Reports belong next to the built binaries, independent of the working
    // directory (which is pinned to the project root for datasets discovery).
    const std::string csv_path = [&] {
        std::error_code ec;
        if (argc > 0 && argv[0]) {
            const auto exe = std::filesystem::canonical(argv[0], ec);
            if (!ec) return (exe.parent_path() / "kd3_datasets_report.csv").string();
        }
        return std::string("kd3_datasets_report.csv");
    }();
    std::ofstream csv(csv_path, std::ios::trunc);
    if (!csv) {
        std::cerr << "Cannot open " << csv_path << " for writing\n";
        return 1;
    }
    write_csv_header(csv);
    csv.flush();

    // ---------------------------------------------------------
    // Real datasets discovered in the datasets folder
    // ---------------------------------------------------------
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    if (std::filesystem::exists(datasets_dir, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(datasets_dir, ec)) {
            if (!entry.is_regular_file()) continue;
            const auto path = entry.path();
            const auto ext = path.extension().string();
            if (ext == ".bin" || ext == ".xyz" || ext == ".pts" || ext == ".ply" ||
                ext == ".laz" || ext == ".las") {
                files.push_back(path);
            }
        }
        std::sort(files.begin(), files.end());
    }

    constexpr size_t N_POINTS = 1'000'000;
    const struct {
        const char* name;
        kdbench::PointCloud (*generate)(size_t, size_t);
    } synthetics[] = {
        {"synthetic-solid", kdbench::uniform_solid},
        {"synthetic-mixture", kdbench::gaussian_mixture},
        {"synthetic-anisotropic", kdbench::anisotropic_gaussian},
        {"synthetic-sphere-shell", kdbench::sphere_shell},
        {"synthetic-torus", kdbench::torus},
        {"synthetic-blobs", kdbench::sdf_blobs},
        {"synthetic-powerlaw", kdbench::power_law_radial},
    };

    auto derive_workloads = [&](const std::string& name, const kdbench::PointCloud& reference) {
        return std::array<std::pair<const char*, kdbench::QuerySet>, 2>{
            std::pair{"bbox-volume",
                      kdbench::queries_uniform_in_bbox(reference, opts.n_queries,
                                                       kdbench::seed_from_name(name + "/uniform"))},
            std::pair{"on-cloud",
                      kdbench::queries_from_dataset(reference, opts.n_queries,
                                                    kdbench::seed_from_name(name + "/dataset"))}};
    };

    auto emit_cloud = [&](const std::string& name, const std::string& source,
                          kdbench::PointCloud&& base, std::vector<Row>& rows,
                          std::ofstream& csv) {
        const kdbench::PointCloud reference = base; // unsorted copy for queries & NF
        const auto workloads = derive_workloads(name, reference);

        NfData nf;
        measure_nanoflann(reference, workloads, opts, nf);

        const size_t first = rows.size();
        {
            auto c = base;
            run_config<TreeType>(name, source, std::move(c), workloads, reference, nf, opts, rows);
        }
        {
            auto c = base;
            run_config<TreeTypeAabb>(name, source, std::move(c), workloads, reference, nf, opts,
                                     rows);
        }
        for (size_t i = first; i < rows.size(); ++i) append_csv_row(csv, rows[i]);
        csv.flush();
    };

    std::vector<Row> rows;

    for (const auto& file : files) {
        auto loaded = kdbench::load_dataset(file);
        if (!loaded) {
            std::cerr << "[skip] " << loaded.error() << "\n";
            continue;
        }
        std::cout << "\n[load] " << file.stem().string() << " -> " << thousands(loaded->size())
                  << " points\n";
        emit_cloud(file.stem().string(), "file", std::move(*loaded), rows, csv);
    }
    if (files.empty()) {
        std::cout << "no dataset files found (run `xmake run fetch-datasets`, or pass --dir=PATH)\n";
    }

    for (const auto& s : synthetics) {
        std::cout << "\n[load] " << s.name << " (generator) -> " << thousands(N_POINTS)
                  << " points\n";
        emit_cloud(s.name, "generator", s.generate(N_POINTS, kdbench::seed_from_name(s.name)),
                   rows, csv);
    }

    print_table(rows);
    std::cout << "\nCSV report: "
              << std::filesystem::absolute(csv_path).string() << "\n";
    return 0;
}
