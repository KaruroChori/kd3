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
#include <optional>

namespace kd3_fast {

constexpr float INF =  1e15f;
constexpr float INF2 = 1e29f;
constexpr size_t MAX_STACK_DEPTH = 48*2;

using std::sort;
using std::nth_element;
using array3 = std::array<float,3>;

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
    array3 coords;
    uint32_t payload_id;
};

/**
 * @brief Information for ray hits.
 * 
 */
struct RayHit {
    float t;
    uint32_t payload_id;
};

/**
 * @brief A bucket representing a leaf node in the kd-tree containing multiple points.
 * 
 * @tparam LeafSize The maximum number of points this leaf can hold.
 */
template <size_t LeafSize = SIMD_PARALLELISM*4>
struct LeafBucket {
    alignas(std::min<size_t>(64, SIMD_PARALLELISM*8)) float x[LeafSize];
    alignas(std::min<size_t>(64, SIMD_PARALLELISM*8)) float y[LeafSize];
    alignas(std::min<size_t>(64, SIMD_PARALLELISM*8)) float z[LeafSize];
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
template <size_t LeafSize = SIMD_PARALLELISM*4>
class KdTreeView {
public:
    std::span<const float> split_vals;
    std::span<const uint64_t> split_dims;
    std::span<const LeafBucket<LeafSize>> buckets;
    array3 min_root;
    array3 max_root;

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
               std::span<const LeafBucket<LeafSize>> bks,
               array3 min_root,
               array3 max_root
                )
        : split_vals(vals), split_dims(dims), buckets(bks), min_root(min_root), max_root(max_root) {}

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
     * @brief Find the closest intersection between a ray and points in the Kd-Tree.
     * 
     * @param ro 3-float array representing the ray origin.
     * @param rd 3-float array representing the ray direction.
     * @param max_t The maximum traversal distance along the ray.
     * @param radius Optional. Represents the mathematical thickness of the ray (Cylinder query).
     * @return std::optional<RayHit> The closest point hit, or std::nullopt if none.
     */
    std::optional<float> query_ray(const array3 ro, const array3 rd, float max_t = INF, float radius = 0.0f) const noexcept {
        if (buckets.empty()) return std::nullopt;


        struct RayNode { size_t idx; float t_min; float t_max; };
        RayNode stack[MAX_STACK_DEPTH];
        size_t stack_sz = 0;
        

        
        float best_t = max_t;
        uint32_t best_id = static_cast<uint32_t>(-1);
        bool hit = false;
        
        array3 inv_rd;
        for (int i = 0; i < 3; ++i) {
            inv_rd[i] = (rd[i] == 0.0f) ? 0.0f : (1.0f / rd[i]); 
        }

        array3 t1_aabb = {(min_root[0] - ro[0]) * inv_rd[0],(min_root[1] - ro[1]) * inv_rd[1],(min_root[2] - ro[2]) * inv_rd[2]};
        array3 t2_aabb = {(max_root[0] - ro[0]) * inv_rd[0],(max_root[1] - ro[1]) * inv_rd[1],(max_root[2] - ro[2]) * inv_rd[2]};

        float tN = std::max(std::max(std::min(t1_aabb[0], t2_aabb[0]), std::min(t1_aabb[1], t2_aabb[1])), std::max(std::min(t1_aabb[2], t2_aabb[2]), float{}));
        float tF = std::min(std::min(std::max(t1_aabb[0], t2_aabb[0]), std::max(t1_aabb[1], t2_aabb[1])), std::min(std::max(t1_aabb[2], t2_aabb[2]), max_t));

        // 2. Instantly cull any ray that misses the scene entirely (e.g., Skybox rays)
        if (tN > tF) return std::nullopt;
        
        stack[stack_sz++] = {0, tN, tF};

        float rd_len_sq = rd[0]*rd[0] + rd[1]*rd[1] + rd[2]*rd[2];
        float inv_rd_len_sq = rd_len_sq > 0.0f ? 1.0f / rd_len_sq : 0.0f;
        float radius_sq = radius * radius;
        float eps = radius_sq > 0.0f ? 0.0f : 1e-5f;

        const size_t LEAF_THRESHOLD = buckets.size() - 1;

        while (stack_sz > 0) {
            auto [curr, t_min, node_t_max] = stack[--stack_sz];
            
            // Front-to-back culling
            if (t_min >= best_t) continue;
            
            if (curr >= LEAF_THRESHOLD) {
                size_t bucket_idx = curr - LEAF_THRESHOLD;
                const auto& b = buckets[bucket_idx];
                
                for (size_t i = 0; i < LeafSize; ++i) {
                    float dx = b.x[i] - ro[0];
                    float dy = b.y[i] - ro[1];
                    float dz = b.z[i] - ro[2];
                    
                    // Project point onto the mathematical line
                    float t = (dx * rd[0] + dy * rd[1] + dz * rd[2]) * inv_rd_len_sq;
                    
                    if (t >= 0.0f && t < best_t) {
                        float px = ro[0] + t * rd[0];
                        float py = ro[1] + t * rd[1];
                        float pz = ro[2] + t * rd[2];
                        
                        float dist_sq = (b.x[i] - px) * (b.x[i] - px) + 
                                        (b.y[i] - py) * (b.y[i] - py) + 
                                        (b.z[i] - pz) * (b.z[i] - pz);
                                        
                        if (dist_sq <= radius_sq + eps) {
                            best_t = t;
                            best_id = b.ids[i];
                            hit = true;
                        }
                    }
                }
                continue;
            }
            
            uint8_t dim = get_dim(curr);
            float split = split_vals[curr];
            
            size_t left_child = 2 * curr + 1;
            size_t right_child = 2 * curr + 2;
            
            // Handle ray traveling exactly parallel to the partition plane
            if (rd[dim] == 0.0f) {
                float dist_to_split = split - ro[dim];
                if (dist_to_split > radius) {
                    stack[stack_sz++] = {left_child, t_min, node_t_max};
                } else if (dist_to_split < -radius) {
                    stack[stack_sz++] = {right_child, t_min, node_t_max};
                } else {
                    stack[stack_sz++] = {right_child, t_min, node_t_max};
                    stack[stack_sz++] = {left_child, t_min, node_t_max};
                }
                continue;
            }
            
            size_t first  = rd[dim] >= 0.0f ? left_child : right_child;
            size_t second = rd[dim] >= 0.0f ? right_child : left_child;
            
            float t_split = (split - ro[dim]) * inv_rd[dim];
            float abs_inv_rd = inv_rd[dim] < 0.0f ? -inv_rd[dim] : inv_rd[dim];
            float t_margin = radius * abs_inv_rd;
            
            float t_enter_second = t_split - t_margin;
            float t_leave_first  = t_split + t_margin;
            
            bool visit_first  = t_min <= t_leave_first;
            bool visit_second = node_t_max >= t_enter_second;
            
            if (visit_first && visit_second) {
                // Front-to-back: Push furthest child 'second' first, so we evaluate 'first' child first.
                float max_t_enter = t_min > t_enter_second ? t_min : t_enter_second;
                float min_t_leave = node_t_max < t_leave_first ? node_t_max : t_leave_first;
                
                stack[stack_sz++] = {second, max_t_enter, node_t_max};
                stack[stack_sz++] = {first, t_min, min_t_leave};
            } else if (visit_first) {
                stack[stack_sz++] = {first, t_min, node_t_max};
            } else if (visit_second) {
                stack[stack_sz++] = {second, t_min, node_t_max};
            }
        }
        
        // Filter out dummy padding points safely
        if (hit && best_id != static_cast<uint32_t>(-1)) {
            return best_t;
        }
        
        return std::nullopt;
    }

    /**
     * @brief Find the single nearest neighbor (1-NN) for a given target.
     * 
     * @param target 3-float array representing the query coordinates.
     * @return std::optional<KnnResult> The closest point found, or std::nullopt if the tree is empty.
     */
    std::optional<float> query_distance(const array3 target) const noexcept {
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

        return min_dist_sq;
    }

};

}