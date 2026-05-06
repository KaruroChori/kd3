#pragma once 
/**
 * @file kd3.hpp
 * @author karurochari
 * @brief Single header library for a vec3 kd-tree
 * @date 2026-05-04
 * 
 * @copyright Copyright (c) 2026
 */

#include <span>
#include <vector>
#include <expected>
#include <algorithm>
#include <bit>
#include <limits>
#include <cstdint>
#include <omp.h>

namespace kd3 {

using std::sort;
using std::nth_element;

#ifndef KD3_SIMD_PARALLELISM
    #if (__cpp_lib_simd || __cpp_lib_experimental_simd) && __has_include(<simd>)
        #include <simd>
        constexpr size_t SIMD_PARALLELISM = std::native_simd<float>::size();
    #elif (__cpp_lib_simd || __cpp_lib_experimental_simd) && __has_include(<experimental/simd>)
        #include <experimental/simd>
        constexpr size_t SIMD_PARALLELISM = std::experimental::native_simd<float>::size();
    #elif defined(__AVX512F__)
        constexpr size_t SIMD_PARALLELISM = 16;
    #elif defined(__AVX2__) || defined(__AVX__)
        constexpr size_t SIMD_PARALLELISM = 8;
    #elif defined(__SSE4_2__) || defined(__SSE2__)
        constexpr size_t SIMD_PARALLELISM = 4;
    #else
        constexpr size_t SIMD_PARALLELISM = 1;
    #endif
#else
    constexpr size_t SIMD_PARALLELISM = KD3_SIMD_PARALLELISM;
    #undef KD3_SIMD_PARALLELISM
#endif

constexpr size_t THRES_PARALLELISM = 10'000;

/**
 * @brief Error codes for KdTree operations.
 */
enum class KdTreeError { EmptyInput };

/**
 * @brief 3D point structure used for building the tree.
 */
struct Point {
    float coords[3];
    uint32_t payload_id;
};

/**
 * @brief A bucket representing a leaf node in the kd-tree containing multiple points.
 * 
 * @tparam LeafSize The maximum number of points this leaf can hold.
 */
template <size_t LeafSize = SIMD_PARALLELISM>
struct LeafBucket {
    alignas(std::max<size_t>(64, LeafSize*8)) float x[LeafSize];
    alignas(std::max<size_t>(64, LeafSize*8)) float y[LeafSize];
    alignas(std::max<size_t>(64, LeafSize*8)) float z[LeafSize];
    uint32_t ids[LeafSize];
    size_t count = 0; 
};

/**
 * @brief Represents a nearest neighbor search result.
 */
struct KnnResult {
    float dist_sq;
    uint32_t payload_id;
    bool operator<(const KnnResult& o) const { return dist_sq < o.dist_sq; }
};

// ---------------------------------------------------------
// NON-OWNING VIEW (Trivially Copyable, Device-Offloadable)
// ---------------------------------------------------------

/**
 * @brief Non-owning view of a kd-tree, capable of querying and device offload.
 * 
 * This class consists only of spans, making it trivially copyable and safe to 
 * map to device memory (e.g., via OpenMP target offload) for parallel queries.
 * 
 * @tparam LeafSize The number of elements packed into each SIMD-friendly leaf.
 */
template <size_t LeafSize = SIMD_PARALLELISM>
class KdTreeView {
public:
    std::span<const float> split_vals;
    std::span<const uint64_t> split_dims;
    std::span<const LeafBucket<LeafSize>> buckets;

    /**
     * @brief Default constructor creating an empty view.
     */
    KdTreeView() = default;

    /**
     * @brief Construct a view from existing data arrays.
     * 
     * @param vals Span of split values.
     * @param dims Span of split dimensions (packed).
     * @param bks Span of leaf buckets.
     */
    KdTreeView(std::span<const float> vals, 
               std::span<const uint64_t> dims, 
               std::span<const LeafBucket<LeafSize>> bks)
        : split_vals(vals), split_dims(dims), buckets(bks) {}

    /**
     * @brief Retrieves the split dimension for a given node index.
     * 
     * @param i Node index.
     * @return uint8_t The dimension (0 for X, 1 for Y, 2 for Z).
     */
    inline uint8_t get_dim(size_t i) const noexcept {
        size_t block = i / 32;
        size_t offset = (i % 32) * 2;
        return (split_dims[block] >> offset) & 0b11;
    }

    /**
     * @brief Find the single nearest neighbor (1-NN) for a given target.
     * 
     * @param target 3-float array representing the query coordinates.
     * @return std::optional<KnnResult> The closest point found, or std::nullopt if the tree is empty.
     */
    std::optional<KnnResult> query_1nn(const float target[3]) const noexcept {
        if (buckets.empty()) return std::nullopt;

        float min_dist_sq = std::numeric_limits<float>::max();
        uint32_t best_id = 0;

        struct SearchNode { size_t idx; float node_min_dist_sq; };
        SearchNode stack[128]; 
        size_t stack_sz = 0;
        
        stack[stack_sz++] = {0, 0.0f};

        const size_t LEAF_THRESHOLD = buckets.size() - 1;

        while (stack_sz > 0) {
            auto [curr, node_min_dist_sq] = stack[--stack_sz];

            if (node_min_dist_sq >= min_dist_sq) continue;

            if (curr >= LEAF_THRESHOLD) {
                size_t bucket_idx = curr - LEAF_THRESHOLD;
                const auto& b = buckets[bucket_idx];

                float dists[LeafSize];
                
                #pragma omp simd
                for (size_t i = 0; i < b.count; ++i) {
                    float dx = target[0] - b.x[i];
                    float dy = target[1] - b.y[i];
                    float dz = target[2] - b.z[i];
                    dists[i] = dx*dx + dy*dy + dz*dz;
                }

                for (size_t i = 0; i < b.count; ++i) {
                    if (dists[i] < min_dist_sq) {
                        min_dist_sq = dists[i];
                        best_id = b.ids[i];
                    }
                }
                continue;
            }

            uint8_t dim = get_dim(curr);
            float split = split_vals[curr];

            float axis_dist = target[dim] - split;
            float axis_dist_sq = axis_dist * axis_dist;
            
            size_t left = 2 * curr + 1;
            size_t right = 2 * curr + 2;

            size_t first = (axis_dist < 0.0f) ? left : right;
            size_t second = (axis_dist < 0.0f) ? right : left;

            if (axis_dist_sq < min_dist_sq) {
                stack[stack_sz++] = {second, axis_dist_sq};
            }
            stack[stack_sz++] = {first, 0.0f};
        }

        return KnnResult{min_dist_sq, best_id};
    }

    /**
     * @brief Find the k-nearest neighbors (k-NN) for a given target.
     * 
     * @param target 3-float array representing the query coordinates.
     * @param buffer A pre-allocated span used to store and manage the max-heap of results. 
     *               The size of this span determines 'k'.
     * @return std::span<KnnResult> A subspan of the buffer containing the found results, sorted from nearest to furthest.
     */
    std::span<KnnResult> query_knn(const float target[3], std::span<KnnResult> buffer) const noexcept {
        const size_t k = buffer.size();
        if (buckets.empty() || k == 0) return {};
        
        size_t heap_size = 0; // Tracks the active number of elements in the buffer

        // Lambda to manage the max-heap directly inside the span
        auto push_heap = [&](float dist, uint32_t id) {
            if (heap_size < k) {
                // Buffer not full: append and sift up
                buffer[heap_size] = {dist, id};
                heap_size++;
                std::push_heap(buffer.begin(), buffer.begin() + heap_size);
            } else if (dist < buffer.front().dist_sq) {
                // Buffer full, new element is closer than the furthest in heap: replace root
                std::pop_heap(buffer.begin(), buffer.begin() + heap_size);
                buffer[heap_size - 1] = {dist, id};
                std::push_heap(buffer.begin(), buffer.begin() + heap_size);
            }
        };

        struct SearchNode { size_t idx; float min_dist_sq; };
        SearchNode stack[128]; 
        size_t stack_sz = 0;
        stack[stack_sz++] = {0, 0.0f};

        const size_t LEAF_THRESHOLD = buckets.size() - 1;

        while (stack_sz > 0) {
            auto [curr, node_min_dist] = stack[--stack_sz];

            // Pruning condition: if the heap is full AND the closest possible 
            // point in this node is further than our worst accepted point, skip it.
            if (heap_size == k && node_min_dist >= buffer.front().dist_sq) continue;

            if (curr >= LEAF_THRESHOLD) {
                size_t bucket_idx = curr - LEAF_THRESHOLD;
                const auto& b = buckets[bucket_idx];

                float dists[LeafSize];
                
                #pragma omp simd
                for (size_t i = 0; i < b.count; ++i) {
                    float dx = target[0] - b.x[i];
                    float dy = target[1] - b.y[i];
                    float dz = target[2] - b.z[i];
                    dists[i] = dx*dx + dy*dy + dz*dz;
                }

                for (size_t i = 0; i < b.count; ++i) {
                    push_heap(dists[i], b.ids[i]);
                }
                continue;
            }

            uint8_t dim = get_dim(curr);
            float split = split_vals[curr];

            float axis_dist = target[dim] - split;
            float axis_dist_sq = axis_dist * axis_dist;
            
            size_t left = 2 * curr + 1;
            size_t right = 2 * curr + 2;

            size_t first = (axis_dist < 0.0f) ? left : right;
            size_t second = (axis_dist < 0.0f) ? right : left;

            if (heap_size < k || axis_dist_sq < buffer.front().dist_sq) {
                stack[stack_sz++] = {second, axis_dist_sq};
            }
            stack[stack_sz++] = {first, 0.0f};
        }

        // Sort the resulting heap so the closest points are at the front
        std::sort(buffer.begin(), buffer.begin() + heap_size);
        
        // Return a subspan covering only the valid results
        return buffer.subspan(0, heap_size);
    }
};


// ---------------------------------------------------------
// OWNING CONTAINER (Handles Allocations / Builder)
// ---------------------------------------------------------

/**
 * @brief Owning kd-tree container that handles memory allocation and building.
 * 
 * Can be implicitly converted to a KdTreeView to perform queries.
 * 
 * @tparam LeafSize The number of elements packed into each SIMD-friendly leaf.
 */
template <size_t LeafSize = SIMD_PARALLELISM>
class KdTree {
private:
    std::vector<float> split_vals;
    std::vector<uint64_t> split_dims;
    std::vector<LeafBucket<LeafSize>> buckets;

    KdTree(std::vector<float> vals, std::vector<uint64_t> dims, std::vector<LeafBucket<LeafSize>> bks)
        : split_vals(std::move(vals)), split_dims(std::move(dims)), buckets(std::move(bks)) {}

    static constexpr size_t calc_left_nodes(size_t n) noexcept {
        if (n <= 1) return 0;
        size_t h = std::bit_width(n) - 1; 
        size_t perfect_left = (size_t(1) << (h - 1)) - 1; 
        size_t bottom_level = n - ((size_t(1) << h) - 1); 
        size_t bottom_left = std::min(bottom_level, size_t(1) << (h - 1)); 
        return perfect_left + bottom_left;
    }

    static constexpr size_t calc_left_buckets(size_t total_buckets) noexcept {
        if (total_buckets <= 1) return 0;
        size_t total_nodes = 2 * total_buckets - 1; 
        size_t left_nodes = calc_left_nodes(total_nodes);
        return (left_nodes + 1) / 2;
    }

    struct Builder {
        std::span<Point> temp_pts;
        std::vector<float>& vals;
        std::vector<uint64_t>& dims;
        std::vector<LeafBucket<LeafSize>>& buckets;
        size_t B;

        inline void set_dim(size_t i, uint8_t dim) {
            size_t block = i / 32;
            size_t offset = (i % 32) * 2;
            dims[block] |= (static_cast<uint64_t>(dim) << offset);
        }

        void build(size_t start, size_t end, size_t node_idx, size_t current_buckets) {
            size_t size = end - start;
            if (size == 0) return;

            if (node_idx >= B - 1) {
                size_t bucket_idx = node_idx - (B - 1);
                buckets[bucket_idx].count = size;
                for (size_t i = 0; i < size; ++i) {
                    buckets[bucket_idx].x[i] = temp_pts[start + i].coords[0];
                    buckets[bucket_idx].y[i] = temp_pts[start + i].coords[1];
                    buckets[bucket_idx].z[i] = temp_pts[start + i].coords[2];
                    buckets[bucket_idx].ids[i] = temp_pts[start + i].payload_id;
                }
                return;
            }

            float min_b[3] = { temp_pts[start].coords[0], temp_pts[start].coords[1], temp_pts[start].coords[2] };
            float max_b[3] = { min_b[0], min_b[1], min_b[2] };
            
            for (size_t i = start + 1; i < end; ++i) {
                for (int d = 0; d < 3; ++d) {
                    min_b[d] = std::min(min_b[d], temp_pts[i].coords[d]);
                    max_b[d] = std::max(max_b[d], temp_pts[i].coords[d]);
                }
            }

            uint8_t best_dim = 0;
            float max_ex = max_b[0] - min_b[0];
            for (uint8_t d = 1; d < 3; ++d) {
                if (max_b[d] - min_b[d] > max_ex) {
                    max_ex = max_b[d] - min_b[d];
                    best_dim = d;
                }
            }

            size_t left_buckets = calc_left_buckets(current_buckets);
            size_t left_points = left_buckets * LeafSize;
            size_t mid = start + left_points;

            nth_element(
                temp_pts.begin() + start, 
                temp_pts.begin() + mid, 
                temp_pts.begin() + end,
                [best_dim](const Point& a, const Point& b) {
                    return a.coords[best_dim] < b.coords[best_dim];
                }
            );

            vals[node_idx] = temp_pts[mid].coords[best_dim];
            set_dim(node_idx, best_dim);

            if (size > THRES_PARALLELISM) {
                #pragma omp task shared(temp_pts, vals, dims, buckets)
                build(start, mid, 2 * node_idx + 1, left_buckets);
                
                #pragma omp task shared(temp_pts, vals, dims, buckets)
                build(mid, end, 2 * node_idx + 2, current_buckets - left_buckets);
                
                #pragma omp taskwait
            } else {
                build(start, mid, 2 * node_idx + 1, left_buckets);
                build(mid, end, 2 * node_idx + 2, current_buckets - left_buckets);
            }
        }
    };

public:
    /**
     * @brief Builds a kd-tree from a mutable span of points.
     * 
     * Note: The input span will be mutated (partially sorted) during construction.
     * 
     * @param temp_pts Mutable span of points to build the tree from.
     * @return std::expected<KdTree, KdTreeError> The built tree, or KdTreeError::EmptyInput if the input was empty.
     */
    static std::expected<KdTree, KdTreeError> build(std::span<Point> temp_pts) {
        if (temp_pts.empty()) return std::unexpected(KdTreeError::EmptyInput);

        size_t n = temp_pts.size();
        size_t B = (n + LeafSize - 1) / LeafSize; 

        std::vector<float> vals(B > 0 ? B - 1 : 0);
        std::vector<uint64_t> dims((vals.size() + 31) / 32, 0);
        std::vector<LeafBucket<LeafSize>> buckets(B);

        Builder builder{temp_pts, vals, dims, buckets, B};

        #pragma omp parallel
        {
            #pragma omp single
            {
                builder.build(0, n, 0, B);
            }
        }

        return KdTree(std::move(vals), std::move(dims), std::move(buckets));
    }

    /**
     * @brief Extracts the non-owning view capable of device offload.
     * 
     * @return KdTreeView<LeafSize> The view of this tree.
     */
    KdTreeView<LeafSize> view() const noexcept {
        return KdTreeView<LeafSize>(split_vals, split_dims, buckets);
    }

    /**
     * @brief Implicit conversion to view.
     */
    operator KdTreeView<LeafSize>() const noexcept {
        return view();
    }

    /**
     * @brief Forwarding method to find the single nearest neighbor (1-NN) for a given target.
     * 
     * @param target 3-float array representing the query coordinates.
     * @return std::optional<KnnResult> The closest point found, or std::nullopt if the tree is empty.
     */
    std::optional<KnnResult> query_1nn(const float target[3]) const noexcept {
        return view().query_1nn(target);
    }

    /**
     * @brief Forwarding method to find the k-nearest neighbors (k-NN) for a given target.
     * 
     * @param target 3-float array representing the query coordinates.
     * @param buffer A pre-allocated span used to store and manage the max-heap of results. 
     * @return std::span<KnnResult> A subspan of the buffer containing the found results, sorted from nearest to furthest.
     */
    std::span<KnnResult> query_knn(const float target[3], std::span<KnnResult> buffer) const noexcept {
        return view().query_knn(target, buffer);
    }
};

}