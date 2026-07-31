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
#include <limits>
#include <cstdint>
#include <array>
#include <bit>
#include "version.h"

//TODO: change a bit to make it more compatible with other compilers as well.
#define KD3_INLINE inline __attribute__((always_inline))

namespace kd3 {

using std::sort;
using std::nth_element;
using default_scalar_t = float;

template<typename Scalar, size_t Dims = 3, typename Distance=float, typename Payload=uint32_t>
struct limits{
    using scalar_t = Scalar;
    using distance_t = Distance;
    using payload_t = Payload;
    static constexpr size_t D = Dims;
    using point_t = std::array<scalar_t, D>;

    template<typename T>
    constexpr static T INF_v(){
        if constexpr (std::is_same_v<T,int32_t>){return 2147483647;}    //Not used
        else if constexpr (std::is_same_v<T,int16_t>){return 32767;}    //Not used
        else if constexpr (std::is_same_v<T,int8_t>){return 127;}       //Not used
        else if constexpr (std::is_same_v<T,float>){return 1e15f;}
        else return std::numeric_limits<T>::max() / 2; // Generic fallback
    }

    template<typename T>
    constexpr static T INF2_v(){
        if constexpr (std::is_same_v<T,int32_t>){return 2147483647;}
        else if constexpr (std::is_same_v<T,int16_t>){return 32767;}
        else if constexpr (std::is_same_v<T,int8_t>){return 127;}
        else if constexpr (std::is_same_v<T,float>){return 1e29f;}
        else return std::numeric_limits<T>::max() / 2; // Generic fallback
    }

    template<typename T>
    constexpr static T INF_IDX_v(){
        return std::numeric_limits<T>::max();
    }
    
    /**
    * @brief D-dimensional point structure used for building the tree.
    */
    struct FatPoint {
        point_t coords;
        payload_t payload_id;
    };

    /**
    * @brief Information for ray hits.
    * 
    */
    struct RayHit {
        distance_t t;
        payload_t payload_id;
    };

    /**
    * @brief Represents a nearest neighbor search result.
    */
    struct KnnResult {
        distance_t dist_sq;
        payload_t payload_id;
        bool operator<(const KnnResult& o) const { return dist_sq < o.dist_sq; }
    };

    constexpr static scalar_t INF = INF_v<scalar_t>();
    constexpr static distance_t INF2 = INF2_v<distance_t>();
    constexpr static payload_t INF_IDX = INF_IDX_v<payload_t>();
};

struct cfg_t{
#ifndef KD3_SIMD_PARALLELISM
    #if (__cpp_lib_simd || __cpp_lib_experimental_simd) && __has_include(<simd>)
        #include <simd>
        constexpr static  size_t simd_parallelism = std::native_simd<default_scalar_t>::size();
    #elif (__cpp_lib_simd || __cpp_lib_experimental_simd) && __has_include(<experimental/simd>)
        #include <experimental/simd>
        constexpr static  size_t simd_parallelism = std::experimental::native_simd<default_scalar_t>::size();
    #elif defined(__AVX512F__)
        constexpr static  size_t simd_parallelism = 16;
    #elif defined(__AVX2__) || defined(__AVX__)
        constexpr static size_t simd_parallelism = 8;
    #elif defined(__SSE4_2__) || defined(__SSE2__)
        constexpr static  size_t simd_parallelism = 4;
    #else
        constexpr static  size_t simd_parallelism = 1;
    #endif
#else
    constexpr static  size_t simd_parallelism = KD3_SIMD_PARALLELISM;
    #undef KD3_SIMD_PARALLELISM
#endif
    size_t max_stack_depth = 48*2;
    size_t thres_thread = 10'000;
    size_t leaf_size = simd_parallelism*4;
    enum has_payload_t{NONE, INDEX, OTHER} has_payload= has_payload_t::INDEX;
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
 * @tparam Limits picking types for its content.
 * @tparam _cfg configuration flags and constants
 */
template <typename Limits=limits<default_scalar_t>, cfg_t _cfg = {}>
class KdTreeView {
public:
    using scalar_t = Limits::scalar_t;
    using distance_t = Limits::distance_t;
    using payload_t = Limits::payload_t;
    using point_t = Limits::point_t;
    using RayHit = Limits::RayHit;
    using FatPoint = Limits::FatPoint;
    using KnnResult = Limits::KnnResult;

    constexpr static cfg_t cfg = _cfg;
    
    // Dynamic bit-packing based on dimensions
    static constexpr size_t dim_bits = []() {
        if constexpr (Limits::D <= 2) return 1;
        else if constexpr (Limits::D <= 4) return 2;
        else if constexpr (Limits::D <= 16) return 4;
        else return 8; // Max 256 dimensions
    }();
    static constexpr size_t dim_mask = (1ULL << dim_bits) - 1;
    static constexpr size_t dims_per_word = 64 / dim_bits;

    /**
    * @brief A bucket representing a leaf node in the kd-tree containing multiple points.
    */
    struct LeafBucket {
        alignas(std::min<size_t>(64, cfg.simd_parallelism*8)) scalar_t coords[Limits::D][cfg.leaf_size];
        payload_t ids[cfg.has_payload!=cfg_t::has_payload_t::NONE?cfg.leaf_size:0];
    };

    /**
    * @brief Error codes for KdTree operations.
    */
    enum struct error_t { Ok, EmptyInput, EmptyContainer, NotSupported, NotImplemented, NotFound };

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
     * @param min_root Root BBOX min point
     * @param max_root Root BBOX max point
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
    KD3_INLINE uint8_t get_dim(size_t i) const noexcept {
        size_t block = i / dims_per_word;
        size_t offset = (i % dims_per_word) * dim_bits;
        return (split_dims[block] >> offset) & dim_mask;
    }

    /**
     * @brief Find the closest intersection between a ray and points in the Kd-Tree.
     * 
     * @param ro 3-scalar_t array representing the ray origin.
     * @param rd 3-scalar_t array representing the ray direction.
     * @param max_t The maximum traversal distance along the ray.
     * @param radius Optional. Represents the mathematical thickness of the ray (Cylinder query).
     * @return  The closest point hit, or error.
     */
    std::expected<RayHit, error_t> query_ray(const point_t ro, const point_t rd, scalar_t max_t = Limits::INF, scalar_t radius = scalar_t{}) const noexcept {
        return query_ray_inline(ro, rd, max_t, radius);
    }

    KD3_INLINE std::expected<RayHit, error_t> query_ray_inline(const point_t ro, const point_t rd, scalar_t max_t = Limits::INF, scalar_t radius = scalar_t{}) const noexcept {
        if constexpr (cfg.has_payload!=cfg_t::has_payload_t::INDEX) return std::unexpected{error_t::NotSupported};
        if constexpr (!std::is_floating_point_v<distance_t>) return std::unexpected{error_t::NotSupported};
        if (buckets.empty()) [[unlikely]] return std::unexpected{error_t::EmptyContainer};

        struct RayNode { size_t idx; scalar_t t_min; scalar_t t_max; };
        RayNode stack[cfg.max_stack_depth];
        size_t stack_sz = 0;
        
        scalar_t best_t = max_t;
        payload_t best_id = Limits::INF_IDX;
        bool hit = false;
        
        point_t inv_rd;
        for (size_t i = 0; i < Limits::D; ++i) {
            inv_rd[i] = (rd[i] == 0.0f) ? 0.0f : (1.0f / rd[i]); 
        }

        scalar_t tN = scalar_t{};
        scalar_t tF = max_t;
        for (size_t i = 0; i < Limits::D; ++i) {
            scalar_t t1 = (min_root[i] - ro[i]) * inv_rd[i];
            scalar_t t2 = (max_root[i] - ro[i]) * inv_rd[i];
            tN = std::max(tN, std::min(t1, t2));
            tF = std::min(tF, std::max(t1, t2));
        }

        if (tN > tF) return std::unexpected{error_t::NotFound};
        
        stack[stack_sz++] = {0, tN, tF};

        scalar_t rd_len_sq = 0.0f;
        for (size_t d = 0; d < Limits::D; ++d) rd_len_sq += rd[d]*rd[d];

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
                
                for (size_t i = 0; i < cfg.leaf_size; ++i) {
                    scalar_t t_num = 0.0f;
                    for(size_t d = 0; d < Limits::D; ++d) {
                        t_num += (b.coords[d][i] - ro[d]) * rd[d];
                    }

                    // Project point onto the mathematical line
                    scalar_t t = t_num * inv_rd_len_sq;
                    
                    if (t >= 0.0f && t < best_t) {
                        scalar_t dist_sq = 0.0f;
                        for(size_t d = 0; d < Limits::D; ++d) {
                            scalar_t px = ro[d] + t * rd[d];
                            scalar_t diff = b.coords[d][i] - px;
                            dist_sq += diff * diff;
                        }
                                        
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
        if (hit && best_id != Limits::INF_IDX) {
            return RayHit{best_t, best_id};
        }
        
        return std::unexpected{error_t::NotFound};
    }

    /**
     * @brief Find the closest intersection between a ray and points in the Kd-Tree.
     * 
     * @param ro 3-scalar_t array representing the ray origin.
     * @param rd 3-scalar_t array representing the ray direction.
     * @param max_t The maximum traversal distance along the ray.
     * @param radius Optional. Represents the mathematical thickness of the ray (Cylinder query).
     * @return  The distance to the closest point hit, or error.
     */
    std::expected<distance_t, error_t> query_ray_distance(const point_t ro, const point_t rd, scalar_t max_t = Limits::INF, scalar_t radius = scalar_t{}) const noexcept {
        return query_ray_distance_inline(ro,rd,max_t,radius);
    }

    KD3_INLINE std::expected<distance_t, error_t> query_ray_distance_inline(const point_t ro, const point_t rd, scalar_t max_t = Limits::INF, scalar_t radius = scalar_t{}) const noexcept {
        if constexpr (!std::is_floating_point_v<distance_t>) return std::unexpected{error_t::NotSupported};
        if (buckets.empty()) [[unlikely]] return std::unexpected{error_t::EmptyContainer};

        struct RayNode { size_t idx; scalar_t t_min; scalar_t t_max; };
        RayNode stack[cfg.max_stack_depth];
        size_t stack_sz = 0;
        
        scalar_t best_t = max_t;
        bool hit = false;
        
        point_t inv_rd;
        for (size_t i = 0; i < Limits::D; ++i) {
            inv_rd[i] = (rd[i] == 0.0f) ? 0.0f : (1.0f / rd[i]); 
        }

        scalar_t tN = scalar_t{};
        scalar_t tF = max_t;
        for (size_t i = 0; i < Limits::D; ++i) {
            scalar_t t1 = (min_root[i] - ro[i]) * inv_rd[i];
            scalar_t t2 = (max_root[i] - ro[i]) * inv_rd[i];
            tN = std::max(tN, std::min(t1, t2));
            tF = std::min(tF, std::max(t1, t2));
        }

        if (tN > tF) return std::unexpected{error_t::NotFound};
        
        stack[stack_sz++] = {0, tN, tF};

        scalar_t rd_len_sq = 0.0f;
        for (size_t d = 0; d < Limits::D; ++d) rd_len_sq += rd[d]*rd[d];

        scalar_t inv_rd_len_sq = rd_len_sq > 0.0f ? 1.0f / rd_len_sq : 0.0f;
        scalar_t radius_sq = radius * radius;
        scalar_t eps = radius_sq > 0.0f ? 0.0f : 1e-5f;

        const size_t LEAF_THRESHOLD = buckets.size() - 1;

        while (stack_sz > 0) {
            auto [curr, t_min, node_t_max] = stack[--stack_sz];
            
            if (t_min >= best_t) continue;
            
            if (curr >= LEAF_THRESHOLD) {
                size_t bucket_idx = curr - LEAF_THRESHOLD;
                const auto& b = buckets[bucket_idx];
                
                for (size_t i = 0; i < cfg.leaf_size; ++i) {
                    scalar_t t_num = 0.0f;
                    for(size_t d = 0; d < Limits::D; ++d) {
                        t_num += (b.coords[d][i] - ro[d]) * rd[d];
                    }
                    scalar_t t = t_num * inv_rd_len_sq;
                    
                    if (t >= 0.0f && t < best_t) {
                        scalar_t dist_sq = 0.0f;
                        for(size_t d = 0; d < Limits::D; ++d) {
                            scalar_t px = ro[d] + t * rd[d];
                            scalar_t diff = b.coords[d][i] - px;
                            dist_sq += diff * diff;
                        }
                                        
                        if (dist_sq <= radius_sq + eps) {
                            best_t = t;
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
        
        if (hit) return static_cast<distance_t>(best_t);
        
        return std::unexpected{error_t::NotFound};
    }

    /**
     * @brief Find the single nearest neighbor (1-NN) for a given target.
     * 
     * @param target 3-scalar_t array representing the query coordinates.
     * @return The closest point found, or error.
     */
    KD3_INLINE std::expected<KnnResult, error_t> query_1nn(const point_t target) const noexcept {
        return query_1nn_inline(target);
    }

    KD3_INLINE std::expected<KnnResult, error_t> query_1nn_inline(const point_t target) const noexcept {
        if constexpr (cfg.has_payload!=cfg_t::has_payload_t::INDEX) return std::unexpected{error_t::NotSupported};
        if (buckets.empty()) [[unlikely]] return std::unexpected{error_t::EmptyContainer};

        distance_t min_dist_sq = std::numeric_limits<distance_t>::max();
        payload_t best_id = {};

        struct SearchNode { size_t idx; distance_t node_min_dist_sq; };
        SearchNode stack[cfg.max_stack_depth]; 
        size_t stack_sz = 0;
        
        stack[stack_sz++] = {0, scalar_t{}};

        const size_t LEAF_THRESHOLD = buckets.size() - 1;

        while (stack_sz > 0) {
            auto [curr, node_min_dist_sq] = stack[--stack_sz];

            if (node_min_dist_sq >= min_dist_sq) continue;

            if (curr >= LEAF_THRESHOLD) {
                size_t bucket_idx = curr - LEAF_THRESHOLD;
                const auto& b = buckets[bucket_idx];

                distance_t dists[cfg.leaf_size] = {};
                for (size_t d = 0; d < Limits::D; ++d) {
                    for (size_t i = 0; i < cfg.leaf_size; ++i) {
                        distance_t diff = target[d] - b.coords[d][i];
                        dists[i] += diff * diff;
                    }
                }

                for (size_t i = 0; i < cfg.leaf_size; ++i) {
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

        if (min_dist_sq >= Limits::INF2) [[unlikely]] return std::unexpected{error_t::NotFound};

        return KnnResult{min_dist_sq, best_id};
    }

    /**
     * @brief Find the k-nearest neighbors (k-NN) for a given target.
     * 
     * @param target 3-scalar_t array representing the query coordinates.
     * @param buffer A pre-allocated span used to store and manage the max-heap of results. 
     *               The size of this span determines 'k'.
     * @return A subspan of the buffer containing the found results, sorted from nearest to furthest. Or error.
     */
    std::expected<std::span<KnnResult>, error_t> query_knn(const point_t target, std::span<KnnResult> buffer) const noexcept {
        return query_knn_inline(target, buffer);
    }

    KD3_INLINE std::expected<std::span<KnnResult>, error_t> query_knn_inline(const point_t target, std::span<KnnResult> buffer) const noexcept {
        if constexpr (cfg.has_payload!=cfg_t::has_payload_t::INDEX) return std::unexpected{error_t::NotSupported};
        if (buckets.empty()) [[unlikely]] return std::unexpected{error_t::EmptyContainer};

        const size_t k = buffer.size();
        if (buckets.empty() || k == 0) [[unlikely]] return {};
        
        size_t heap_size = 0;

        auto push_heap = [&](distance_t dist, payload_t id) {
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
        SearchNode stack[cfg.max_stack_depth]; 
        size_t stack_sz = 0;
        stack[stack_sz++] = {0, distance_t{}};

        const size_t LEAF_THRESHOLD = buckets.size() - 1;

        while (stack_sz > 0) {
            auto [curr, node_min_dist] = stack[--stack_sz];

            if (heap_size == k && node_min_dist >= buffer.front().dist_sq) continue;

            if (curr >= LEAF_THRESHOLD) {
                size_t bucket_idx = curr - LEAF_THRESHOLD;
                const auto& b = buckets[bucket_idx];

                distance_t dists[cfg.leaf_size] = {};
                for (size_t d = 0; d < Limits::D; ++d) {
                    for (size_t i = 0; i < cfg.leaf_size; ++i) {
                        distance_t diff = target[d] - b.coords[d][i];
                        dists[i] += diff * diff;
                    }
                }

                for (size_t i = 0; i < cfg.leaf_size; ++i) {
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

        sort(buffer.begin(), buffer.begin() + heap_size);
        
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

    /**
     * @brief Find the single nearest neighbor (1-NN) for a given target.
     * 
     * @param target 3-float array representing the query coordinates.
     * @return The distance to the closest point found, or error.
     */
    std::expected<distance_t,error_t> query_distance2(const point_t target) const noexcept {
        return query_distance2_inline(target);
    }

    KD3_INLINE std::expected<distance_t,error_t> query_distance2_inline(const point_t target) const noexcept {
        if (buckets.empty()) [[unlikely]] return std::unexpected{error_t::EmptyContainer};

        distance_t min_dist_sq = std::numeric_limits<distance_t>::max();

        struct SearchNode { size_t idx; distance_t node_min_dist_sq; };
        SearchNode stack[cfg.max_stack_depth]; 
        size_t stack_sz = 0;
        
        stack[stack_sz++] = {0, scalar_t{}};

        const size_t LEAF_THRESHOLD = buckets.size() - 1;

        while (stack_sz > 0) {
            auto [curr, node_min_dist_sq] = stack[--stack_sz];

            if (node_min_dist_sq >= min_dist_sq) continue;

            if (curr >= LEAF_THRESHOLD) {
                size_t bucket_idx = curr - LEAF_THRESHOLD;
                const auto& b = buckets[bucket_idx];

                distance_t dists[cfg.leaf_size] = {};
                for (size_t d = 0; d < Limits::D; ++d) {
                    for (size_t i = 0; i < cfg.leaf_size; ++i) {
                        distance_t diff = target[d] - b.coords[d][i];
                        dists[i] += diff * diff;
                    }
                }

                for (size_t i = 0; i < cfg.leaf_size; ++i) {
                    if (dists[i] < min_dist_sq) {
                        min_dist_sq = dists[i];
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
        if (min_dist_sq >= Limits::INF2) [[unlikely]] return std::unexpected{error_t::NotFound};

        return min_dist_sq;
    }
};

}