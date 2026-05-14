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

namespace kd3 {

using std::sort;
using std::nth_element;
using default_scalar_t = float;

template<typename Scalar=default_scalar_t, typename Distance =default_scalar_t>
struct limits_base{
    using scalar_t = Scalar;
    using distance_t = Distance;
    using point_t = std::array<scalar_t,3>;

    /**
    * @brief 3D point structure used for building the tree.
    */
    struct FatPoint {
        point_t coords;
        uint32_t payload_id;
    };

    /**
    * @brief Information for ray hits.
    * 
    */
    struct RayHit {
        distance_t t;
        uint32_t payload_id;
    };

    /**
    * @brief Represents a nearest neighbor search result.
    */
    struct KnnResult {
        distance_t dist_sq;
        uint32_t payload_id;
        bool operator<(const KnnResult& o) const { return dist_sq < o.dist_sq; }
    };

};

template<typename T>
struct limits{};

template<>
struct limits<float> : limits_base<float,float>{
    constexpr static float INF =  1e15f;
    constexpr static float INF2 = 1e29f;
};

template<>
struct limits<int32_t> : limits_base<int32_t,float>{
    constexpr static int32_t INF =  2147483647;
    constexpr static double INF2 = 1e29f;   //TODO: change
};

template<>
struct limits<int16_t> : limits_base<int16_t,int32_t>{
    constexpr static int16_t INF =  32767;
    constexpr static int32_t INF2 = 2147483647;
};

template<>
struct limits<int8_t> : limits_base<int8_t,float>{
    constexpr static int8_t INF =  127;
    constexpr static int16_t INF2 = 32767;
};



struct cfg_t{
#ifndef KD3_SIMD_PARALLELISM
    #if (__cpp_lib_simd || __cpp_lib_experimental_simd) && __has_include(<simd>)
        #include <simd>
        constexpr static  size_t SIMD_PARALLELISM = std::native_simd<default_scalar_t>::size();
    #elif (__cpp_lib_simd || __cpp_lib_experimental_simd) && __has_include(<experimental/simd>)
        #include <experimental/simd>
        constexpr static  size_t SIMD_PARALLELISM = std::experimental::native_simd<default_scalar_t>::size();
    #elif defined(__AVX512F__)
        constexpr static  size_t SIMD_PARALLELISM = 16;
    #elif defined(__AVX2__) || defined(__AVX__)
        constexpr static size_t SIMD_PARALLELISM = 8;
    #elif defined(__SSE4_2__) || defined(__SSE2__)
        constexpr static  size_t SIMD_PARALLELISM = 4;
    #else
        constexpr static  size_t SIMD_PARALLELISM = 1;
    #endif
#else
    constexpr static  size_t SIMD_PARALLELISM = KD3_SIMD_PARALLELISM;
    #undef KD3_SIMD_PARALLELISM
#endif
    size_t MAX_STACK_DEPTH = 48*2;
    size_t THRES_PARALLELISM = 10'000;
    size_t LeafSize = SIMD_PARALLELISM*4;
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
template <typename Limits=limits<default_scalar_t>, cfg_t _cfg = {}>
class KdTreeView {
public:
    using scalar_t = Limits::scalar_t;
    using distance_t = Limits::distance_t;
    using point_t = Limits::point_t;
    using RayHit = Limits::RayHit;
    using FatPoint = Limits::FatPoint;
    using KnnResult = Limits::KnnResult;

    constexpr static cfg_t cfg = _cfg;

    /**
    * @brief A bucket representing a leaf node in the kd-tree containing multiple points.
    * 
    * @tparam LeafSize The maximum number of points this leaf can hold.
    */
    struct LeafBucket {
        alignas(std::min<size_t>(64, cfg.SIMD_PARALLELISM*8)) scalar_t x[cfg.LeafSize];
        alignas(std::min<size_t>(64, cfg.SIMD_PARALLELISM*8)) scalar_t y[cfg.LeafSize];
        alignas(std::min<size_t>(64, cfg.SIMD_PARALLELISM*8)) scalar_t z[cfg.LeafSize];
        uint32_t ids[cfg.LeafSize];
    };

    /**
    * @brief Error codes for KdTree operations.
    */
    enum struct error_t { EmptyInput };

    std::span<const scalar_t> split_vals;
    std::span<const uint64_t> split_dims;
    std::span<const LeafBucket> buckets;
    point_t min_root;
    point_t max_root;

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
    KdTreeView(std::span<const scalar_t> vals, 
               std::span<const uint64_t> dims, 
               std::span<const LeafBucket> bks,
               point_t min_root,
               point_t max_root
    ) : split_vals(vals), split_dims(dims), buckets(bks), min_root(min_root), max_root(max_root) {}

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
     * @param ro 3-scalar_t array representing the ray origin.
     * @param rd 3-scalar_t array representing the ray direction.
     * @param max_t The maximum traversal distance along the ray.
     * @param radius Optional. Represents the mathematical thickness of the ray (Cylinder query).
     * @return std::optional<RayHit> The closest point hit, or std::nullopt if none.
     */
    std::optional<RayHit> query_ray(const point_t ro, const point_t rd, scalar_t max_t = Limits::INF, scalar_t radius = 0.0f) const noexcept {
        //Not working for type which are not float-like (cannot represent the inverse and hard to use INF without a mantissa)
        if constexpr(!std::is_floating_point_v<distance_t>)return std::nullopt;

        if (buckets.empty()) return std::nullopt;


        struct RayNode { size_t idx; scalar_t t_min; scalar_t t_max; };
        RayNode stack[cfg.MAX_STACK_DEPTH];
        size_t stack_sz = 0;
        

        
        scalar_t best_t = max_t;
        uint32_t best_id = static_cast<uint32_t>(-1);
        bool hit = false;
        
        point_t inv_rd;
        for (int i = 0; i < 3; ++i) {
            inv_rd[i] = (rd[i] == 0.0f) ? 0.0f : (1.0f / rd[i]); 
        }

        point_t t1_aabb = {(min_root[0] - ro[0]) * inv_rd[0],(min_root[1] - ro[1]) * inv_rd[1],(min_root[2] - ro[2]) * inv_rd[2]};
        point_t t2_aabb = {(max_root[0] - ro[0]) * inv_rd[0],(max_root[1] - ro[1]) * inv_rd[1],(max_root[2] - ro[2]) * inv_rd[2]};

        scalar_t tN = std::max(std::max(std::min(t1_aabb[0], t2_aabb[0]), std::min(t1_aabb[1], t2_aabb[1])), std::max(std::min(t1_aabb[2], t2_aabb[2]), scalar_t{}));
        scalar_t tF = std::min(std::min(std::max(t1_aabb[0], t2_aabb[0]), std::max(t1_aabb[1], t2_aabb[1])), std::min(std::max(t1_aabb[2], t2_aabb[2]), max_t));

        // 2. Instantly cull any ray that misses the scene entirely (e.g., Skybox rays)
        if (tN > tF) return std::nullopt;
        
        stack[stack_sz++] = {0, tN, tF};

        scalar_t rd_len_sq = rd[0]*rd[0] + rd[1]*rd[1] + rd[2]*rd[2];
        scalar_t inv_rd_len_sq = rd_len_sq > 0.0f ? 1.0f / rd_len_sq : 0.0f;
        scalar_t radius_sq = radius * radius;
        scalar_t eps = radius_sq > 0.0f ? 0.0f : 1e-5f;

        const size_t LEAF_THRESHOLD = buckets.size() - 1;

        while (stack_sz > 0) {
            auto [curr, t_min, node_t_max] = stack[--stack_sz];
            
            // Front-to-back culling
            if (t_min >= best_t) continue;
            
            if (curr >= LEAF_THRESHOLD) {
                size_t bucket_idx = curr - LEAF_THRESHOLD;
                const auto& b = buckets[bucket_idx];
                
                for (size_t i = 0; i < cfg.LeafSize; ++i) {
                    scalar_t dx = b.x[i] - ro[0];
                    scalar_t dy = b.y[i] - ro[1];
                    scalar_t dz = b.z[i] - ro[2];
                    
                    // Project point onto the mathematical line
                    scalar_t t = (dx * rd[0] + dy * rd[1] + dz * rd[2]) * inv_rd_len_sq;
                    
                    if (t >= 0.0f && t < best_t) {
                        scalar_t px = ro[0] + t * rd[0];
                        scalar_t py = ro[1] + t * rd[1];
                        scalar_t pz = ro[2] + t * rd[2];
                        
                        scalar_t dist_sq = (b.x[i] - px) * (b.x[i] - px) + 
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
            scalar_t split = split_vals[curr];
            
            size_t left_child = 2 * curr + 1;
            size_t right_child = 2 * curr + 2;
            
            // Handle ray traveling exactly parallel to the partition plane
            if (rd[dim] == 0.0f) {
                scalar_t dist_to_split = split - ro[dim];
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
            
            scalar_t t_split = (split - ro[dim]) * inv_rd[dim];
            scalar_t abs_inv_rd = inv_rd[dim] < 0.0f ? -inv_rd[dim] : inv_rd[dim];
            scalar_t t_margin = radius * abs_inv_rd;
            
            scalar_t t_enter_second = t_split - t_margin;
            scalar_t t_leave_first  = t_split + t_margin;
            
            bool visit_first  = t_min <= t_leave_first;
            bool visit_second = node_t_max >= t_enter_second;
            
            if (visit_first && visit_second) {
                // Front-to-back: Push furthest child 'second' first, so we evaluate 'first' child first.
                scalar_t max_t_enter = t_min > t_enter_second ? t_min : t_enter_second;
                scalar_t min_t_leave = node_t_max < t_leave_first ? node_t_max : t_leave_first;
                
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
            return RayHit{best_t, best_id};
        }
        
        return std::nullopt;
    }

    /**
     * @brief Find the single nearest neighbor (1-NN) for a given target.
     * 
     * @param target 3-scalar_t array representing the query coordinates.
     * @return std::optional<KnnResult> The closest point found, or std::nullopt if the tree is empty.
     */
    std::optional<KnnResult> query_1nn(const point_t target) const noexcept {
        if (buckets.empty()) return std::nullopt;

        distance_t min_dist_sq = std::numeric_limits<distance_t>::max();
        uint32_t best_id = 0;

        struct SearchNode { size_t idx; distance_t node_min_dist_sq; };
        SearchNode stack[cfg.MAX_STACK_DEPTH]; 
        size_t stack_sz = 0;
        
        stack[stack_sz++] = {0, scalar_t{}};

        const size_t LEAF_THRESHOLD = buckets.size() - 1;

        while (stack_sz > 0) {
            auto [curr, node_min_dist_sq] = stack[--stack_sz];

            if (node_min_dist_sq >= min_dist_sq) continue;

            if (curr >= LEAF_THRESHOLD) {
                size_t bucket_idx = curr - LEAF_THRESHOLD;
                const auto& b = buckets[bucket_idx];

                distance_t dists[cfg.LeafSize];
                
                for (size_t i = 0; i < cfg.LeafSize; ++i) {
                    distance_t dx = target[0] - b.x[i];
                    distance_t dy = target[1] - b.y[i];
                    distance_t dz = target[2] - b.z[i];
                    dists[i] = dx*dx + dy*dy + dz*dz;
                }

                for (size_t i = 0; i < cfg.LeafSize; ++i) {
                    if (dists[i] < min_dist_sq) {
                        min_dist_sq = dists[i];
                        best_id = b.ids[i];
                    }
                }
                continue;
            }

            uint8_t dim = get_dim(curr);
            scalar_t split = split_vals[curr];

            distance_t axis_dist = target[dim] - split;
            distance_t axis_dist_sq = axis_dist * axis_dist;
            
            size_t left = 2 * curr + 1;
            size_t right = 2 * curr + 2;

            size_t first = (axis_dist < scalar_t{}) ? left : right;
            size_t second = (axis_dist < scalar_t{}) ? right : left;

            if (axis_dist_sq < min_dist_sq) {
                stack[stack_sz++] = {second, axis_dist_sq};
            }
            stack[stack_sz++] = {first, distance_t{}};
        }

        // Filter out pseudo-infinity points in case tree only had dummy data (impossible from Builder) 
        // or user queried 1e15f explicitly.
        if (min_dist_sq >= Limits::INF2) return std::nullopt;

        return KnnResult{min_dist_sq, best_id};
    }

    /**
     * @brief Find the k-nearest neighbors (k-NN) for a given target.
     * 
     * @param target 3-scalar_t array representing the query coordinates.
     * @param buffer A pre-allocated span used to store and manage the max-heap of results. 
     *               The size of this span determines 'k'.
     * @return std::span<KnnResult> A subspan of the buffer containing the found results, sorted from nearest to furthest.
     */
    std::span<KnnResult> query_knn(const point_t target, std::span<KnnResult> buffer) const noexcept {
        const size_t k = buffer.size();
        if (buckets.empty() || k == 0) return {};
        
        size_t heap_size = 0;

        auto push_heap = [&](distance_t dist, uint32_t id) {
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

        struct SearchNode { size_t idx; distance_t min_dist_sq; };
        SearchNode stack[cfg.MAX_STACK_DEPTH]; 
        size_t stack_sz = 0;
        stack[stack_sz++] = {0, distance_t{}};

        const size_t LEAF_THRESHOLD = buckets.size() - 1;

        while (stack_sz > 0) {
            auto [curr, node_min_dist] = stack[--stack_sz];

            if (heap_size == k && node_min_dist >= buffer.front().dist_sq) continue;

            if (curr >= LEAF_THRESHOLD) {
                size_t bucket_idx = curr - LEAF_THRESHOLD;
                const auto& b = buckets[bucket_idx];

                distance_t dists[cfg.LeafSize];
                
                for (size_t i = 0; i < cfg.LeafSize; ++i) {
                    distance_t dx = target[0] - b.x[i];
                    distance_t dy = target[1] - b.y[i];
                    distance_t dz = target[2] - b.z[i];
                    dists[i] = dx*dx + dy*dy + dz*dz;
                }

                for (size_t i = 0; i < cfg.LeafSize; ++i) {
                    push_heap(dists[i], b.ids[i]);
                }
                continue;
            }

            uint8_t dim = get_dim(curr);
            scalar_t split = split_vals[curr];

            distance_t axis_dist = target[dim] - split;
            distance_t axis_dist_sq = axis_dist * axis_dist;
            
            size_t left = 2 * curr + 1;
            size_t right = 2 * curr + 2;

            size_t first = (axis_dist < distance_t{}) ? left : right;
            size_t second = (axis_dist < distance_t{}) ? right : left;

            if (heap_size < k || axis_dist_sq < buffer.front().dist_sq) {
                stack[stack_sz++] = {second, axis_dist_sq};
            }
            stack[stack_sz++] = {first, distance_t{}};
        }

        std::sort(buffer.begin(), buffer.begin() + heap_size);
        
        // Filter out pseudo-infinity padded points if the user asked for 
        // more nearest-neighbors than points exist in the tree
        size_t valid_results = 0;
        for (size_t i = 0; i < heap_size; ++i) {
            if (buffer[i].dist_sq < Limits::INF2) {
                valid_results++;
            } else {
                break;
            }
        }
        
        return buffer.subspan(0, valid_results);
    }
};

}