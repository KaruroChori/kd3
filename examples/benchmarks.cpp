#include <kd3/kd3.hpp>
#include <vector>
#include <expected>
#include <algorithm>
#include <chrono>
#include <random>
#include <iostream>

// ---------------------------------------------------------
// Baseline Brute Force validation
// ---------------------------------------------------------
std::vector<kd3::KnnResult> linear_scan(const float target[3], size_t k, std::span<const kd3::Point> points) {
    std::vector<kd3::KnnResult> heap;
    heap.reserve(k + 1);
    for (const auto& p : points) {
        float dx = target[0] - p.coords[0];
        float dy = target[1] - p.coords[1];
        float dz = target[2] - p.coords[2];
        float dist_sq = dx*dx + dy*dy + dz*dz;
        
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

// ---------------------------------------------------------
// Benchmarking Suite
// ---------------------------------------------------------
int main() {
    using namespace kd3;
    constexpr size_t PARALLELISM = 32;
    std::cout << "--- KD-Tree HPC Benchmark --- ["<<SIMD_PARALLELISM<<"]\n";
    
    const size_t N_POINTS = 5'000'000;
    const size_t N_QUERIES = 100'000;
    const size_t K = 10;

    std::random_device rd;
    std::mt19937 gen(1337); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    std::cout << "Generating " << N_POINTS << " random points...\n";
    std::vector<Point> raw_points(N_POINTS);
    for (size_t i = 0; i < N_POINTS; ++i) {
        raw_points[i] = {{dist(gen), dist(gen), dist(gen)}, static_cast<uint32_t>(i)};
    }
    
    // Copy for baseline verification
    std::vector<Point> points_copy = raw_points; 

    // Benchmark Build
    std::cout << "Building tree with OpenMP...\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    auto tree_result = KdTree<PARALLELISM>::build(raw_points);
    auto t2 = std::chrono::high_resolution_clock::now();
    
    if (!tree_result) { std::cerr << "Build failed!\n"; return 1; }
    const auto& tree = *tree_result;
    
    double build_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "Build Time: " << build_ms << " ms\n";

    // Generate Queries
    std::vector<std::array<float, 3>> queries(N_QUERIES);
    for (size_t i = 0; i < N_QUERIES; ++i) {
        queries[i] = {dist(gen), dist(gen), dist(gen)};
    }

    // Benchmark KD-Tree
    std::cout << "Running " << N_QUERIES << " queries via KD-Tree...\n";
    auto t3 = std::chrono::high_resolution_clock::now();
    size_t dummy = 0; // Prevent compiler from optimizing away the loop
    for (const auto& q : queries) {
        std::array<kd3::KnnResult,K> storage{};
        auto res = tree.query_knn(q.data(), storage);
        dummy += res.front().payload_id;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    
    double kd_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    std::cout << "KD-Tree Query Time: " << kd_ms << " ms (" 
              << (N_QUERIES / (kd_ms / 1000.0)) << " QPS)\n";

    bool correct = true;
    double linear_ms = 0;

    std::cout << "Validating correctness against linear scan...\n";

    for(int i=0;i<N_QUERIES/1000;i++){

    // Validation Check (Run 1 query against linear scan to verify correctness)
    std::array<kd3::KnnResult,K> storage{};
    auto kd_res = tree.query_knn(queries[i%1000].data(), storage);
    
    auto t5 = std::chrono::high_resolution_clock::now();
    auto brute_res = linear_scan(queries[i%1000].data(), K, points_copy);
    auto t6 = std::chrono::high_resolution_clock::now();
    
    linear_ms += std::chrono::duration<double, std::milli>(t6 - t5).count();
    
    for (size_t i = 0; i < K; ++i) {
        // Due to floating point math eps errors, check ID instead of exact distance match
        if (kd_res[i].payload_id != brute_res[i].payload_id) {
            correct = false;
        }
    }
    }

    if (correct) {
        std::cout << "[PASS] KD-Tree results perfectly match brute force.\n";
    } else {
        std::cout << "[FAIL] KD-Tree results differ!\n";
    }

    std::cout << "Single Brute Force Query: " << linear_ms/(N_QUERIES/1000.0) << " ms\n";
    std::cout << "KD-Tree Speedup vs Brute: " << (linear_ms/(N_QUERIES/1000.0) / (kd_ms / N_QUERIES)) << "x faster per query\n";

    return dummy == 0 ? 1 : 0;
}
