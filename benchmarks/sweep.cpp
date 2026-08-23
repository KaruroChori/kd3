/**
 * @file sweep.cpp
 * @author not me
 * @brief Parametric leaf-size sweep across kd3 tree configurations.
 * @date 2026-05-07
 *
 * Runs a leaf-size x tree-size characterization sweep for any subset of the
 * four benchmarked configurations (base / aabb / fast / fast-aabb), reporting
 * query latency percentiles and build times to kd3_sweep_report.html.
 *
 * Usage: xmake run sweep [--configs=base,aabb,fast,fast-aabb] [--quick]
 */

#include <vector>
#include <chrono>
#include <cstring>
#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>

#include <kd3/kd3.hpp>


struct BenchmarkResult {
    const char* config;
    size_t tree_size;
    size_t leaf_size;
    double build_time_ms;
    double p10_ns, p50_ns, p90_ns, p99_ns;
};

struct SweepConfig {
    const char* name;
    bool aabb;
    bool payload_free;
};

constexpr SweepConfig kConfigs[] = {
    {"base", false, false},
    {"aabb", true, false},
    {"fast", false, true},
    {"fast-aabb", true, true},
};

template <size_t L, bool AABB, bool PAYLOAD_FREE>
void run_benchmark_for_leaf(const char* config_name, size_t N, int trials,
                            std::vector<BenchmarkResult>& results) {
    using namespace kd3;
    using TreeType =
        kd3::KdTree<kd3::limits<float>, {.leaf_size = L,
                                         .has_payload = PAYLOAD_FREE
                                                           ? cfg_t::has_payload_t::NONE
                                                           : cfg_t::has_payload_t::INDEX,
                                         .has_aabb = AABB}>;

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    std::vector<double> build_times;

    // We will run 10,000 queries per trial to get a good statistical distribution
    const size_t NUM_QUERIES = 10000;
    std::vector<double> query_times_ns;
    query_times_ns.reserve(NUM_QUERIES * trials);

    for (int trial = 0; trial < trials; ++trial) {
        // 1. Generate Points
        std::vector<typename TreeType::FatPoint> points(N);
        for (size_t i = 0; i < N; ++i) {
            points[i] = {{dist(gen), dist(gen), dist(gen)}, static_cast<uint32_t>(i)};
        }

        // 2. Measure Build Time
        auto tb0 = std::chrono::high_resolution_clock::now();
        auto tree_expected = TreeType::build(points);
        auto tb1 = std::chrono::high_resolution_clock::now();

        if (!tree_expected) continue;
        const auto& tree = *tree_expected;
        build_times.push_back(std::chrono::duration<double, std::milli>(tb1 - tb0).count());

        // 3. Generate Queries
        std::vector<std::array<float, 3>> queries(NUM_QUERIES);
        for (size_t i = 0; i < NUM_QUERIES; ++i) {
            queries[i] = {dist(gen), dist(gen), dist(gen)};
        }

        std::vector<double> local_query_times(NUM_QUERIES);

        // WARM-UP: Spin the CPU for a few milliseconds to force Turbo Boost
        // to max frequency and warm up the L1/L2 caches.
        volatile uint32_t dummy = 0;
        for (size_t i = 0; i < 1000; ++i) {
            if constexpr (PAYLOAD_FREE) {
                auto res = tree.query_distance2_inline(queries[i % NUM_QUERIES]);
                if (res) dummy += static_cast<uint32_t>(*res);
            } else {
                auto res = tree.query_1nn(queries[i % NUM_QUERIES]);
                if (res) dummy += res->payload_id;
            }
        }

        // 4. Measure Query Latency (Single Threaded to isolate purely architectural SIMD gains)
        for (size_t i = 0; i < NUM_QUERIES; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            if constexpr (PAYLOAD_FREE) {
                auto res = tree.query_distance2_inline(queries[i]);
                auto t1 = std::chrono::high_resolution_clock::now();
                if (res) dummy += static_cast<uint32_t>(*res);
                local_query_times[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
            } else {
                auto res = tree.query_1nn(queries[i]);
                auto t1 = std::chrono::high_resolution_clock::now();

                if (res) dummy += res->payload_id; // prevent optimization
                local_query_times[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
            }
        }

        query_times_ns.insert(query_times_ns.end(), local_query_times.begin(),
                              local_query_times.end());
    }

    // Aggregate Statistics
    std::sort(build_times.begin(), build_times.end());
    double median_build = build_times[build_times.size() / 2];

    std::sort(query_times_ns.begin(), query_times_ns.end());
    size_t total_q = query_times_ns.size();

    BenchmarkResult r;
    r.config = config_name;
    r.tree_size = N;
    r.leaf_size = L;
    r.build_time_ms = median_build;
    r.p10_ns = query_times_ns[total_q * 0.10];
    r.p50_ns = query_times_ns[total_q * 0.50];
    r.p90_ns = query_times_ns[total_q * 0.90];
    r.p99_ns = query_times_ns[total_q * 0.99];
    results.push_back(r);

    std::cout << "  [" << config_name << "] LEAF_SIZE: " << std::setw(3) << L
              << " | Build: " << std::setw(6) << std::fixed << std::setprecision(1)
              << median_build << " ms | Query P50: " << std::setw(5) << r.p50_ns
              << " ns (P99: " << std::setw(5) << r.p99_ns << ")\n";
}

template <bool AABB, bool PAYLOAD_FREE>
void sweep_leaves(const char* config_name, size_t N, int trials,
                  std::vector<BenchmarkResult>& results) {
    run_benchmark_for_leaf<1, AABB, PAYLOAD_FREE>(config_name, N, trials, results);
    run_benchmark_for_leaf<2, AABB, PAYLOAD_FREE>(config_name, N, trials, results);
    run_benchmark_for_leaf<4, AABB, PAYLOAD_FREE>(config_name, N, trials, results);
    run_benchmark_for_leaf<8, AABB, PAYLOAD_FREE>(config_name, N, trials, results);
    run_benchmark_for_leaf<16, AABB, PAYLOAD_FREE>(config_name, N, trials, results);
    run_benchmark_for_leaf<32, AABB, PAYLOAD_FREE>(config_name, N, trials, results);
    run_benchmark_for_leaf<64, AABB, PAYLOAD_FREE>(config_name, N, trials, results);
    run_benchmark_for_leaf<128, AABB, PAYLOAD_FREE>(config_name, N, trials, results);
    run_benchmark_for_leaf<256, AABB, PAYLOAD_FREE>(config_name, N, trials, results);
}

void sweep_config(const SweepConfig& sc, const std::vector<size_t>& tree_sizes, int trials,
                  std::vector<BenchmarkResult>& results) {
    std::cout << "\n=== Configuration: " << sc.name << " ===\n";
    for (size_t N : tree_sizes) {
        std::cout << "\n--- Tree Size: " << N << " points ---\n";
        if (sc.aabb && sc.payload_free)
            sweep_leaves<true, true>(sc.name, N, trials, results);
        else if (sc.aabb)
            sweep_leaves<true, false>(sc.name, N, trials, results);
        else if (sc.payload_free)
            sweep_leaves<false, true>(sc.name, N, trials, results);
        else
            sweep_leaves<false, false>(sc.name, N, trials, results);
    }
}

int main(int argc, char** argv) {
    int TRIALS = 5;
    bool quick = false;
    std::vector<const SweepConfig*> selected;
    for (const auto& c : kConfigs) selected.push_back(&c);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--quick") {
            quick = true;
        } else if (arg.rfind("--configs=", 0) == 0) {
            selected.clear();
            std::string names = arg.substr(10);
            size_t pos = 0;
            while (pos <= names.size()) {
                size_t comma = names.find(',', pos);
                std::string one = names.substr(pos, comma == std::string::npos
                                                        ? std::string::npos
                                                        : comma - pos);
                bool matched = false;
                for (const auto& c : kConfigs)
                    if (one == c.name) { selected.push_back(&c); matched = true; break; }
                if (!matched && !one.empty())
                    std::cerr << "[warn] unknown configuration \"" << one << "\"\n";
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
            if (selected.empty()) {
                std::cerr << "No valid configuration selected.\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            return 1;
        }
    }

    std::vector<size_t> tree_sizes = {100'000,      500'000,  1'000'000, 5'000'000,
                                      10'000'000, 50'000'000, 100'000'000};
    if (quick) {
        TRIALS = 1;
        tree_sizes = {10'000, 100'000};
        std::cout << "[quick mode: reduced sizes and trials]\n";
    }

    std::vector<BenchmarkResult> results;

    std::cout << "Starting kd3 LEAF_SIZE Characterization Sweep (" << TRIALS
              << " trials per config, " << selected.size() << " configurations)...\n";

    for (const auto* sc : selected) sweep_config(*sc, tree_sizes, TRIALS, results);

    // ---------------------------------------------------------
    // HTML report: one query/build chart pair per configuration
    // ---------------------------------------------------------
    std::ofstream html("kd3_sweep_report.html");
    html << R"(<!DOCTYPE html><html><head><title>kd3 Sweep Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>body{font-family: sans-serif; background:#111; color:#eee; margin:0; padding:20px;} 
    .container{display:flex; flex-direction:column; gap:20px; align-items:center;}</style></head><body>
    <h2>kd3 Leaf Size Characterization</h2><div class="container">
    )";

    for (const auto* sc : selected) {
        html << "<h3 style='margin-bottom:0'>" << sc->name << "</h3>\n"
             << "<div id=\"q_" << sc->name << "\" style=\"width:90vw;height:55vh;\"></div>\n"
             << "<div id=\"b_" << sc->name << "\" style=\"width:90vw;height:35vh;\"></div>\n";
    }

    html << R"(</div><script>
    const rawData = [)";

    for (const auto& r : results) {
        html << "{c:'" << r.config << "',N:" << r.tree_size << ",L:" << r.leaf_size
             << ",build:" << r.build_time_ms << ",p10:" << r.p10_ns << ",p50:" << r.p50_ns
             << ",p90:" << r.p90_ns << "},";
    }

    html << R"(];
    const colors = ['#00d2ff', '#00e676', '#ff0055', '#ff9900', '#ff00aa', '#7d22b3', '#a6ce17'];

    [...new Set(rawData.map(d => d.c))].forEach(cfg => {
        const cfgData = rawData.filter(d => d.c === cfg);
        const treeSizes = [...new Set(cfgData.map(d => d.N))];
        let queryTraces = [];
        let buildTraces = [];

        treeSizes.forEach((size, idx) => {
            const subset = cfgData.filter(d => d.N === size);
            const x = subset.map(d => d.L);
            const label = size >= 1000000 ? (size/1000000) + 'M Pts' : (size/1000) + 'K Pts';
            const color = colors[idx % colors.length];

            queryTraces.push({ x: x, y: subset.map(d=>d.p50), name: label + ' (Median)', 
                               type: 'scatter', mode: 'lines+markers', line: {color: color, width: 3} });
            queryTraces.push({ x: x.concat(x.slice().reverse()), 
                               y: subset.map(d=>d.p90).concat(subset.map(d=>d.p10).reverse()), 
                               fill: 'toself', fillcolor: color, opacity: 0.15, line: {color: 'transparent'},
                               name: label + ' (P10-P90)', showlegend: false, type: 'scatter' });

            buildTraces.push({ x: x, y: subset.map(d=>d.build), name: label,
                               type: 'scatter', mode: 'lines+markers', line: {color: color} });
        });

        const dark = {paper_bgcolor:'#111', plot_bgcolor:'#111', font: {color:'#eee'}};
        Plotly.newPlot('q_' + cfg, queryTraces, Object.assign({
            title: 'Query Latency vs Leaf Size', xaxis: {title: 'Leaf Size (Items per Bucket)', type: 'category'},
            yaxis: {title: 'Latency (ns)'}}, dark));
        Plotly.newPlot('b_' + cfg, buildTraces, Object.assign({
            title: 'Build Time (ms) vs Leaf Size', xaxis: {title: 'Leaf Size (Items per Bucket)', type: 'category'},
            yaxis: {title: 'Build Time (ms)'}}, dark));
    });
    </script></body></html>)";

    std::cout << "\nResults rendered directly to kd3_sweep_report.html\nOpen it in any browser.\n";
    return 0;
}
