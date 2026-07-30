#pragma once 
/**
 * @file kd3.hpp
 * @author karurochari
 * @brief Main header file for a vec3 kd-tree, usually you want to use this.
 * @date 2026-05-04
 * 
 * @copyright Copyright (c) 2026
 */

#include <span>
#include <vector>
#include <expected>
#include <algorithm>
#include <bit>
#include <cstdint>

#include "query.hpp"
#include "version.h"

namespace kd3 {

// ---------------------------------------------------------
// OWNING CONTAINER (Handles Allocations / Builder)
// ---------------------------------------------------------

/**
 * @brief Owning kd-tree container that handles memory allocation and building.
 * 
 * Can be implicitly converted to a KdTreeView to perform queries.
 * 
 */
template <typename Limits=limits<default_scalar_t>, cfg_t _cfg = {}>
class KdTree {
public:
    constexpr static cfg_t cfg = _cfg;

    using scalar_t = Limits::scalar_t;
    using distance_t = Limits::distance_t;
    using point_t = Limits::point_t;
    using RayHit = Limits::RayHit;
    using FatPoint = Limits::FatPoint;
    using KnnResult = Limits::KnnResult;
    using LeafBucket = KdTreeView<Limits,cfg>::LeafBucket;
    using error_t = KdTreeView<Limits,cfg>::error_t;

private:
    std::vector<scalar_t> split_vals;
    std::vector<uint64_t> split_dims;
    std::vector<LeafBucket> buckets;
    point_t min_root;
    point_t max_root;

    KdTree(std::vector<scalar_t> vals, std::vector<uint64_t> dims, std::vector<LeafBucket> bks)
        : split_vals(std::move(vals)), split_dims(std::move(dims)), buckets(std::move(bks)) {
        for (size_t d = 0; d < Limits::D; ++d) {
            min_root[d] = -Limits::INF;
            max_root[d] = Limits::INF;
        }
    }

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

    template<bool preordered=false>
    struct Builder {
        std::span<FatPoint> temp_pts;
        std::vector<scalar_t>& vals;
        std::vector<uint64_t>& dims;
        std::vector<LeafBucket>& buckets;
        size_t B;

        inline void set_dim(size_t i, uint8_t dim) {
            size_t block = i / KdTreeView<Limits, cfg>::dims_per_word;
            size_t offset = (i % KdTreeView<Limits, cfg>::dims_per_word) * KdTreeView<Limits, cfg>::dim_bits;
            #pragma omp atomic
            dims[block] |= (static_cast<uint64_t>(dim) << offset);
        }

        void build(size_t start, size_t end, size_t node_idx, size_t current_buckets) {
            size_t size = end - start;
            if (size == 0) return;

            // Base case: store in a Leaf Bucket
            if (node_idx >= B - 1) {
                size_t bucket_idx = node_idx - (B - 1);
                
                // Copy genuine elements
                for (size_t i = 0; i < size; ++i) {
                    for (size_t d = 0; d < Limits::D; ++d) {
                        buckets[bucket_idx].coords[d][i] = temp_pts[start + i].coords[d];
                    }
                    if constexpr (cfg.has_payload==cfg_t::has_payload_t::INDEX) buckets[bucket_idx].ids[i] = temp_pts[start + i].payload_id;
                }
                
                // Fill the remainder with pseudo-infinity points
                for (size_t i = size; i < cfg.leaf_size; ++i) {
                    for (size_t d = 0; d < Limits::D; ++d) {
                        buckets[bucket_idx].coords[d][i] = Limits::INF; // Extremely far away
                    }
                    if constexpr (cfg.has_payload==cfg_t::has_payload_t::INDEX) buckets[bucket_idx].ids[i] = Limits::INF_IDX;
                }
                
                return;
            }

            scalar_t min_b[Limits::D];
            scalar_t max_b[Limits::D];
            
            for (size_t d = 0; d < Limits::D; ++d) {
                min_b[d] = temp_pts[start].coords[d];
                max_b[d] = temp_pts[start].coords[d];
            }
            
            for (size_t i = start + 1; i < end; ++i) {
                for (size_t d = 0; d < Limits::D; ++d) {
                    min_b[d] = std::min(min_b[d], temp_pts[i].coords[d]);
                    max_b[d] = std::max(max_b[d], temp_pts[i].coords[d]);
                }
            }

            uint8_t best_dim = 0;
            scalar_t max_ex = max_b[0] - min_b[0];
            for (uint8_t d = 1; d < Limits::D; ++d) {
                if (max_b[d] - min_b[d] > max_ex) {
                    max_ex = max_b[d] - min_b[d];
                    best_dim = d;
                }
            }

            size_t left_buckets = calc_left_buckets(current_buckets);
            size_t left_points = left_buckets * cfg.leaf_size;
            size_t mid = start + left_points;

            if constexpr(!preordered) nth_element(
                temp_pts.begin() + start, 
                temp_pts.begin() + mid, 
                temp_pts.begin() + end,
                [best_dim](const FatPoint& a, const FatPoint& b) {
                    return a.coords[best_dim] < b.coords[best_dim];
                }
            );

            vals[node_idx] = temp_pts[mid].coords[best_dim];
            set_dim(node_idx, best_dim);

            if (size > cfg.thres_thread) {
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
     * @return std::expected<KdTree, error_t> The built tree, or error_t::EmptyInput if the input was empty.
     */
    static std::expected<KdTree, error_t> build(std::span<FatPoint> temp_pts) {
        if (temp_pts.empty()) return std::unexpected(error_t::EmptyInput);

        size_t n = temp_pts.size();
        size_t B = (n + cfg.leaf_size - 1) / cfg.leaf_size; 

        size_t dims_per_word = KdTreeView<Limits, cfg>::dims_per_word;
        std::vector<scalar_t> vals(B > 0 ? B - 1 : 0);
        std::vector<uint64_t> dims((vals.size() + dims_per_word - 1) / dims_per_word, 0);
        std::vector<LeafBucket> buckets(B);

        Builder<false> builder{temp_pts, vals, dims, buckets, B};

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
     * @brief Builds a kd-tree from a span of points which are assumed to be already ordered.
     * 
     * Note: The input span will be mutated (partially sorted) during construction.
     * 
     * @param temp_pts Mutable span of points to build the tree from.
     * @return std::expected<KdTree, error_t> The built tree, or error_t::EmptyInput if the input was empty.
     */
    static std::expected<KdTree, error_t> build_from_ordered(std::span<const FatPoint> temp_pts) {
        if (temp_pts.empty()) return std::unexpected(error_t::EmptyInput);

        size_t n = temp_pts.size();
        size_t B = (n + cfg.leaf_size - 1) / cfg.leaf_size; 

        size_t dims_per_word = KdTreeView<Limits, cfg>::dims_per_word;
        std::vector<scalar_t> vals(B > 0 ? B - 1 : 0);
        std::vector<uint64_t> dims((vals.size() + dims_per_word - 1) / dims_per_word, 0);
        std::vector<LeafBucket> buckets(B);

        Builder<true> builder{{const_cast<FatPoint*>(temp_pts.data()), temp_pts.size()}, vals, dims, buckets, B};

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
     * @return KdTreeView The view of this tree after building.
     */
    KdTreeView<Limits,cfg> view() const noexcept {
        return KdTreeView<Limits,cfg>(split_vals, split_dims, buckets, min_root, max_root);
    }

    /**
     * @brief Implicit conversion to view.
     */
    operator KdTreeView<Limits,cfg>() const noexcept {
        return view();
    }

    /**
     * @brief Forwarding method to find the closest intersection between a ray and points in the Kd-Tree.
     * 
     * @param ro scalar_t array representing the ray origin.
     * @param rd scalar_t array representing the ray direction.
     * @param max_t The maximum traversal distance along the ray.
     * @param radius Optional. Represents the mathematical thickness of the ray (Cylinder query).
     * @return The closest point hit, or error codes.
     */
    KD3_INLINE std::expected<RayHit, error_t> query_ray(const point_t ro, const point_t rd, scalar_t max_t=Limits::INF, scalar_t radius={}) const noexcept {
        return view().query_ray(ro, rd, max_t, radius);
    }

    /**
     * @brief Forwarding method to find the closest intersection distance between a ray and points in the Kd-Tree.
     * 
     * @param ro scalar_t array representing the ray origin.
     * @param rd scalar_t array representing the ray direction.
     * @param max_t The maximum traversal distance along the ray.
     * @param radius Optional. Represents the mathematical thickness of the ray (Cylinder query).
     * @return The distance to the closest point hit, or error codes.
     */
    KD3_INLINE std::expected<distance_t, error_t> query_ray_distance(const point_t ro, const point_t rd, scalar_t max_t=Limits::INF, scalar_t radius={}) const noexcept {
        return view().query_ray_distance(ro, rd, max_t, radius);
    }

    /**
     * @brief Forwarding method to find the single nearest neighbor (1-NN) for a given target.
     * 
     * @param target scalar_t array representing the query coordinates.
     * @return The closest point found, or error codes.
     */
    KD3_INLINE std::expected<KnnResult, error_t> query_1nn(const point_t target) const noexcept {
        return view().query_1nn(target);
    }

    /**
     * @brief Forwarding method to find the distance (squared) from the nearest neighbor (1-NN) of a given target.
     * 
     * @param target scalar_t array representing the query coordinates.
     * @return The closest point found, or error codes.
     */
    KD3_INLINE std::expected<distance_t, error_t> query_distance2(const point_t target) const noexcept {
        return view().query_distance2(target);
    }

    /**
     * @brief Forwarding method to find the k-nearest neighbors (k-NN) for a given target.
     * 
     * @param target scalar_t array representing the query coordinates.
     * @param buffer A pre-allocated span used to store and manage the max-heap of results. 
     * @return A subspan of the buffer containing the found results, sorted from nearest to furthest. Or error codes.
     */
    KD3_INLINE std::expected<std::span<KnnResult>, error_t> query_knn(const point_t target, std::span<KnnResult> buffer) const noexcept {
        return view().query_knn(target, buffer);
    }

    KD3_INLINE std::expected<RayHit, error_t> query_ray_inline(const point_t ro, const point_t rd, scalar_t max_t=Limits::INF, scalar_t radius={}) const noexcept {
        return view().query_ray_inline(ro, rd, max_t, radius);
    }

    KD3_INLINE std::expected<distance_t, error_t> query_ray_distance_inline(const point_t ro, const point_t rd, scalar_t max_t=Limits::INF, scalar_t radius={}) const noexcept {
        return view().query_ray_distance_inline(ro, rd, max_t, radius);
    }

    KD3_INLINE std::expected<KnnResult, error_t> query_1nn_inline(const point_t target) const noexcept {
        return view().query_1nn_inline(target);
    }

    KD3_INLINE std::expected<distance_t, error_t> query_distance2_inline(const point_t target) const noexcept {
        return view().query_distance2_inline(target);
    }

    KD3_INLINE std::expected<std::span<KnnResult>, error_t> query_knn_inline(const point_t target, std::span<KnnResult> buffer) const noexcept {
        return view().query_knn_inline(target, buffer);
    }

    void set_bbox(point_t min, point_t max){
        min_root = min;
        max_root = max;
    }
};

}