#include <kd3/kd3.hpp>
#include <nanoflann.hpp>
#include <vector>
#include <expected>
#include <algorithm>
#include <chrono>
#include <random>
#include <iostream>
#include <span>
#include <array>

using TreeType = kd3::KdTree<kd3::limits<float>, {.leaf_size=32}>;


struct Kd3PointAdaptor {
    const std::vector<TreeType::FatPoint>& pts;
    
    Kd3PointAdaptor(const std::vector<TreeType::FatPoint>& pts) : pts(pts) {}
    
    inline size_t kdtree_get_point_count() const { 
        return pts.size(); 
    }
    
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const { 
        return pts[idx].coords[dim]; 
    }
    
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const { 
        return false; 
    }
};

// ---------------------------------------------------------
// Baseline Brute Force validation
// ---------------------------------------------------------
std::vector<TreeType::KnnResult> linear_scan(const TreeType::scalar_t target[3], size_t k, std::span<const TreeType::FatPoint> points) {
    std::vector<TreeType::KnnResult> heap;
    heap.reserve(k + 1);
    for (const auto& p : points) {
        TreeType::distance_t dx = target[0] - p.coords[0];
        TreeType::distance_t dy = target[1] - p.coords[1];
        TreeType::distance_t dz = target[2] - p.coords[2];
        TreeType::distance_t dist_sq = dx*dx + dy*dy + dz*dz;
        
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
    std::cout << "--- KD-Tree Benchmark --- [simd: "<<TreeType::cfg.simd_parallelism<<", parallelism: "<<PARALLELISM<<"]\n";

    constexpr size_t N_POINTS = 5'000'000;
    constexpr size_t N_QUERIES = 100'000;
    constexpr size_t K = 100;

    std::random_device rd;
    std::mt19937 gen(1337); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    std::cout << "Generating " << N_POINTS << " random points...\n";
    std::vector<TreeType::FatPoint> raw_points(N_POINTS);
    for (size_t i = 0; i < N_POINTS; ++i) {
        raw_points[i] = {{dist(gen), dist(gen), dist(gen)}, static_cast<uint32_t>(i)};
    }

    // Copy for baseline verification & nanoflann (in case kd3::build mutates raw_points)
    std::vector<TreeType::FatPoint> points_copy = raw_points; 

    // ---------------------------------------------------------
    // 1. Build Phase
    // ---------------------------------------------------------
    std::cout << "\n[ BUILD PHASE ]\n";
    
    // kd3 Build
    auto t1 = std::chrono::high_resolution_clock::now();
    auto tree_result = TreeType::build(raw_points);
    auto t2 = std::chrono::high_resolution_clock::now();
    if (!tree_result) { std::cerr << "kd3 Build failed!\n"; return 1; }
    const auto& tree = *tree_result;
    double kd3_build_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << "kd3 Build Time:       " << kd3_build_ms << " ms\n";

    // Nanoflann Build
    using nanoflann_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, Kd3PointAdaptor>, 
        Kd3PointAdaptor, 
        3,         /* Dimensions */
        uint32_t   /* IndexType -> matches payload_id */
    >;
    
    Kd3PointAdaptor nf_adaptor(points_copy);
    nanoflann_tree_t nf_tree(3, nf_adaptor, {10 /* max leaf size */});
    
    auto tnf1 = std::chrono::high_resolution_clock::now();
    nf_tree.buildIndex();
    auto tnf2 = std::chrono::high_resolution_clock::now();
    double nf_build_ms = std::chrono::duration<double, std::milli>(tnf2 - tnf1).count();
    std::cout << "nanoflann Build Time: " << nf_build_ms << " ms\n";

    // Generate Queries
    std::vector<TreeType::point_t> queries(N_QUERIES);
    for (size_t i = 0; i < N_QUERIES; ++i) {
        queries[i] = {dist(gen), dist(gen), dist(gen)};
    }

    // ---------------------------------------------------------
    // 2. Query Phase
    // ---------------------------------------------------------
    std::cout << "\n[ QUERY PHASE ] - " << N_QUERIES << " queries\n";
    
    // kd3 Query
    auto t3 = std::chrono::high_resolution_clock::now();
    size_t dummy_kd3 = 0;
    for (const auto& q : queries) {
        std::array<TreeType::KnnResult, K> storage{};
        auto res = *tree.query_knn_inline(q, storage);
        dummy_kd3 += res.front().payload_id;
    }
    auto t4 = std::chrono::high_resolution_clock::now();
    double kd_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
    
    std::cout << "kd3 Query Time:       " << kd_ms << " ms (" 
              << (N_QUERIES / (kd_ms / 1000.0)) << " QPS)\n";

    // Nanoflann Query
    // Pre-allocate buffers for nanoflann
    std::vector<uint32_t> nf_indices(K);
    std::vector<float> nf_dists(K);
    
    auto tnf3 = std::chrono::high_resolution_clock::now();
    size_t dummy_nf = 0;
    for (const auto& q : queries) {
        nf_tree.knnSearch(q.data(), K, nf_indices.data(), nf_dists.data());
        dummy_nf += nf_indices[0]; // matches front().payload_id
    }
    auto tnf4 = std::chrono::high_resolution_clock::now();
    double nf_ms = std::chrono::duration<double, std::milli>(tnf4 - tnf3).count();
    
    std::cout << "nanoflann Query Time: " << nf_ms << " ms (" 
              << (N_QUERIES / (nf_ms / 1000.0)) << " QPS)\n";

    // Prevent compiler optimizations
    if (dummy_kd3 == 0 && dummy_nf == 0) std::cout << " ";

    // ---------------------------------------------------------
    // 3. Validation Phase
    // ---------------------------------------------------------
    std::cout << "\n[ VALIDATION PHASE ]\n";
    std::cout << "Validating correctness against linear scan...\n";

    bool kd3_correct = true;
    bool nf_correct = true;
    double linear_ms = 0;

    int num_validation_checks = N_QUERIES / 1000;
    for (int i = 0; i < num_validation_checks; i++) {
        const auto& q = queries[i % 1000];

        // 1. Run kd3
        std::array<TreeType::KnnResult, K> storage{};
        auto kd_res = *tree.query_knn(q, storage);

        // 2. Run Nanoflann
        nf_tree.knnSearch(q.data(), K, nf_indices.data(), nf_dists.data());

        // 3. Run Brute Force
        auto t5 = std::chrono::high_resolution_clock::now();
        auto brute_res = linear_scan(q.data(), K, points_copy);
        auto t6 = std::chrono::high_resolution_clock::now();
        linear_ms += std::chrono::duration<double, std::milli>(t6 - t5).count();

        // Check Correctness
        for (size_t j = 0; j < K; ++j) {
            if (kd_res[j].payload_id != brute_res[j].payload_id) {
                kd3_correct = false;
            }
            if (nf_indices[j] != brute_res[j].payload_id) {
                nf_correct = false;
            }
        }
    }
    
    std::cout << "Linear Scan Time (Subset):       " << linear_ms << " ms\n";
    std::cout << "kd3 matches brute force:         " << (kd3_correct ? "YES" : "NO") << "\n";
    std::cout << "nanoflann matches brute force:   " << (nf_correct ? "YES" : "NO") << "\n";

    return 0;
}
