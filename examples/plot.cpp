/**
 * @file plot.cpp
 * @author not me
 * @brief A totally vibecoded utility, just to get out some fancy plot.
 * @date 2026-05-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <vector>
#include <chrono>
#include <random>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>

#include <kd3/kd3.hpp>

struct BenchmarkResult {
    size_t tree_size;
    size_t leaf_size;
    double build_time_ms;
    double p10_ns, p50_ns, p90_ns, p99_ns;
};

template <size_t L>
void run_benchmark_for_leaf(size_t N, int trials, std::vector<BenchmarkResult>& results) {
    using namespace kd3;
    
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    std::vector<double> build_times;
    
    // We will run 10,000 queries per trial to get a good statistical distribution
    const size_t NUM_QUERIES = 10000;
    std::vector<double> query_times_ns; 
    query_times_ns.reserve(NUM_QUERIES * trials);

    for (int trial = 0; trial < trials; ++trial) {
        // 1. Generate Points
        std::vector<Point> points(N);
        for (size_t i = 0; i < N; ++i) {
            points[i] = {{dist(gen), dist(gen), dist(gen)}, static_cast<uint32_t>(i)};
        }

        // 2. Measure Build Time
        auto tb0 = std::chrono::high_resolution_clock::now();
        auto tree_expected = KdTree<L>::build(points);
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
            auto res = tree.query_1nn(queries[i % NUM_QUERIES].data());
            if (res) dummy += res->payload_id;
        }

        // 4. Measure Query Latency (Single Threaded to isolate purely architectural SIMD gains)
        for (size_t i = 0; i < NUM_QUERIES; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            auto res = tree.query_1nn(queries[i].data());
            auto t1 = std::chrono::high_resolution_clock::now();
            
            if (res) dummy += res->payload_id; // prevent optimization
            local_query_times[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }

        query_times_ns.insert(query_times_ns.end(), local_query_times.begin(), local_query_times.end());
    }

    // Aggregate Statistics
    std::sort(build_times.begin(), build_times.end());
    double median_build = build_times[build_times.size() / 2];

    std::sort(query_times_ns.begin(), query_times_ns.end());
    size_t total_q = query_times_ns.size();

    BenchmarkResult r;
    r.tree_size = N;
    r.leaf_size = L;
    r.build_time_ms = median_build;
    r.p10_ns = query_times_ns[total_q * 0.10];
    r.p50_ns = query_times_ns[total_q * 0.50];
    r.p90_ns = query_times_ns[total_q * 0.90];
    r.p99_ns = query_times_ns[total_q * 0.99];
    results.push_back(r);

    std::cout << "  LeafSize: " << std::setw(2) << L 
              << " | Build: " << std::setw(6) << std::fixed << std::setprecision(1) << median_build << " ms"
              << " | Query P50: " << std::setw(5) << r.p50_ns << " ns (P99: " << std::setw(5) << r.p99_ns << ")\n";
}

int main() {
    const int TRIALS = 5;
    //You might want to reduce this a notch or two depending on your hardware. It will take minutes on high end workstations so...
    std::vector<size_t> tree_sizes = {100'000, 500'000, 1'000'000, 5'000'000, 10'000'000, 50'000'000, 100'000'000};
    std::vector<BenchmarkResult> results;

    std::cout << "Starting kd3 LeafSize Characterization Sweep (" << TRIALS << " trials per config)...\n";

    for (size_t N : tree_sizes) {
        std::cout << "\n--- Tree Size: " << N << " points ---\n";
        // Instantiate the template for different SIMD bucket sizes
        run_benchmark_for_leaf<1>(N, TRIALS, results);
        run_benchmark_for_leaf<2>(N, TRIALS, results);
        run_benchmark_for_leaf<4>(N, TRIALS, results);
        run_benchmark_for_leaf<8>(N, TRIALS, results);
        run_benchmark_for_leaf<16>(N, TRIALS, results);
        run_benchmark_for_leaf<32>(N, TRIALS, results);
    }

    std::ofstream html("kd3_leaf_report.html");
    html << R"(<!DOCTYPE html><html><head><title>kd3 LeafSize Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>body{font-family: sans-serif; background:#111; color:#eee; margin:0; padding:20px;} 
    .container{display:flex; flex-direction:column; gap:20px; align-items:center;}</style></head><body>
    <h2>kd3 SIMD Leaf Size Characterization</h2><div class="container">
    <div id="queryPlot" style="width:90vw;height:60vh;"></div>
    <div id="buildPlot" style="width:90vw;height:40vh;"></div></div>
    <script>
    const rawData = [)";
    
    for (const auto& r : results) {
        html << "{N:" << r.tree_size << ",L:" << r.leaf_size 
             << ",build:" << r.build_time_ms << ",p10:" << r.p10_ns 
             << ",p50:" << r.p50_ns << ",p90:" << r.p90_ns << "},";
    }
    
    html << R"(];
    const treeSizes = [...new Set(rawData.map(d => d.N))];
    const colors = ['#00d2ff', '#00e676', '#ff0055', '#ff9900', '#ff00aa', '#7d22b3', '#a6ce17'];
    
    let queryTraces = [];
    let buildTraces = [];

    treeSizes.forEach((size, idx) => {
        const subset = rawData.filter(d => d.N === size);
        const x = subset.map(d => d.L); // X-axis is now Leaf Size
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

    Plotly.newPlot('queryPlot', queryTraces, {
        title: 'Query Latency vs SIMD Leaf Size', paper_bgcolor:'#111', plot_bgcolor:'#111', font: {color:'#eee'},
        xaxis: {title: 'Leaf Size (Items per Bucket)', type: 'category'},
        yaxis: {title: 'Latency (ns)'}
    });

    Plotly.newPlot('buildPlot', buildTraces, {
        title: 'Tree Build Time (ms) vs SIMD Leaf Size', paper_bgcolor:'#111', plot_bgcolor:'#111', font: {color:'#eee'},
        xaxis: {title: 'Leaf Size (Items per Bucket)', type: 'category'},
        yaxis: {title: 'Build Time (ms)'}
    });
    </script></body></html>)";

    std::cout << "\nResults rendered directly to kd3_leaf_report.html\nOpen it in any browser.\n";
    return 0;
}
