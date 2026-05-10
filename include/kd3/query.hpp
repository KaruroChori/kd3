#pragma once 
/**
 * @file query.hpp
 * @author karurochari
 * @brief Lightweight part of the kd3 library, without memory allocations and embedded-friendly.
 * @date 2026-05-04
 * 
 * @copyright Copyright (c) 2026
 */

#include <span>
#include <expected>
#include <algorithm>
#include <bit>
#include <limits>
#include <cstdint>

namespace kd3 {

constexpr float INF =  1e15f;
constexpr float INF2 = 1e29f;
constexpr size_t MAX_STACK_DEPTH = 48;

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
    alignas(std::min<size_t>(64, LeafSize*8)) float x[LeafSize];
    alignas(std::min<size_t>(64, LeafSize*8)) float y[LeafSize];
    alignas(std::min<size_t>(64, LeafSize*8)) float z[LeafSize];
    uint32_t ids[LeafSize];
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
        SearchNode stack[MAX_STACK_DEPTH]; 
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
                
                for (size_t i = 0; i < LeafSize; ++i) {
                    float dx = target[0] - b.x[i];
                    float dy = target[1] - b.y[i];
                    float dz = target[2] - b.z[i];
                    dists[i] = dx*dx + dy*dy + dz*dz;
                }

                for (size_t i = 0; i < LeafSize; ++i) {
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

        // Filter out pseudo-infinity points in case tree only had dummy data (impossible from Builder) 
        // or user queried 1e15f explicitly.
        if (min_dist_sq >= INF2) return std::nullopt;

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
        
        size_t heap_size = 0;

        auto push_heap = [&](float dist, uint32_t id) {
            if (heap_size < k) {
                buffer[heap_size] = {dist, id};
                heap_size++;
                std::push_heap(buffer.begin(), buffer.begin() + heap_size);
            } else if (dist < buffer.front().dist_sq) {
                std::pop_heap(buffer.begin(), buffer.begin() + heap_size);
                buffer[heap_size - 1] = {dist, id};
                std::push_heap(buffer.begin(), buffer.begin() + heap_size);
            }
        };

        struct SearchNode { size_t idx; float min_dist_sq; };
        SearchNode stack[MAX_STACK_DEPTH]; 
        size_t stack_sz = 0;
        stack[stack_sz++] = {0, 0.0f};

        const size_t LEAF_THRESHOLD = buckets.size() - 1;

        while (stack_sz > 0) {
            auto [curr, node_min_dist] = stack[--stack_sz];

            if (heap_size == k && node_min_dist >= buffer.front().dist_sq) continue;

            if (curr >= LEAF_THRESHOLD) {
                size_t bucket_idx = curr - LEAF_THRESHOLD;
                const auto& b = buckets[bucket_idx];

                float dists[LeafSize];
                
                for (size_t i = 0; i < LeafSize; ++i) {
                    float dx = target[0] - b.x[i];
                    float dy = target[1] - b.y[i];
                    float dz = target[2] - b.z[i];
                    dists[i] = dx*dx + dy*dy + dz*dz;
                }

                for (size_t i = 0; i < LeafSize; ++i) {
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

        std::sort(buffer.begin(), buffer.begin() + heap_size);
        
        // Filter out pseudo-infinity padded points if the user asked for 
        // more nearest-neighbors than points exist in the tree
        size_t valid_results = 0;
        for (size_t i = 0; i < heap_size; ++i) {
            if (buffer[i].dist_sq < INF2) {
                valid_results++;
            } else {
                break;
            }
        }
        
        return buffer.subspan(0, valid_results);
    }
};

}