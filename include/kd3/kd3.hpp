#pragma once 
/**
 * @file kd3.hpp
 * @author karurochari
 * @brief Main header file for a vec3 kd-tree, usually you want to use this.
 * @date 2026-05-04
 * 
 * @copyright Copyright (c) 2026
 */

#include <cassert>
#include <span>
#include <vector>
#include <expected>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <omp.h>

#include "query.hpp"

namespace kd3 {

// ---------------------------------------------------------
// OWNING CONTAINER (Handles Allocations / Builder)
// ---------------------------------------------------------

/**
 * @brief Owning kd-tree container that handles memory allocation and building.
 * 
 * Can be implicitly converted to a KdTreeView to perform queries.
 * 
 * @tparam LEAF_SIZE The number of elements packed into each SIMD-friendly leaf.
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
    point_t min_root = {-Limits::INF,-Limits::INF,-Limits::INF};
    point_t max_root = {Limits::INF,Limits::INF,Limits::INF};

    KdTree(std::vector<scalar_t> vals, std::vector<uint64_t> dims, std::vector<LeafBucket> bks)
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

    template<bool preordered=false>
    struct Builder {
        std::span<FatPoint> temp_pts;
        std::vector<scalar_t>& vals;
        std::vector<uint64_t>& dims;
        std::vector<LeafBucket>& buckets;
        size_t B;

        inline void set_dim(size_t i, uint8_t dim) {
            size_t block = i / 32;
            size_t offset = (i % 32) * 2;
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
                    buckets[bucket_idx].x[i] = temp_pts[start + i].coords[0];
                    buckets[bucket_idx].y[i] = temp_pts[start + i].coords[1];
                    buckets[bucket_idx].z[i] = temp_pts[start + i].coords[2];
                    if constexpr (cfg.had_index) buckets[bucket_idx].ids[i] = temp_pts[start + i].payload_id;
                }
                
                // Fill the remainder with pseudo-infinity points
                for (size_t i = size; i < cfg.leaf_size; ++i) {
                    buckets[bucket_idx].x[i] = Limits::INF; // Extremely far away, 1e15^2 = 1e30 < 3.4e38 (scalar_t max)
                    buckets[bucket_idx].y[i] = Limits::INF;
                    buckets[bucket_idx].z[i] = Limits::INF;
                    if constexpr (cfg.had_index) buckets[bucket_idx].ids[i] = static_cast<uint32_t>(-1);
                }
                
                return;
            }

            scalar_t min_b[3] = { temp_pts[start].coords[0], temp_pts[start].coords[1], temp_pts[start].coords[2] };
            scalar_t max_b[3] = { min_b[0], min_b[1], min_b[2] };
            
            for (size_t i = start + 1; i < end; ++i) {
                for (int d = 0; d < 3; ++d) {
                    min_b[d] = std::min(min_b[d], temp_pts[i].coords[d]);
                    max_b[d] = std::max(max_b[d], temp_pts[i].coords[d]);
                }
            }

            uint8_t best_dim = 0;
            scalar_t max_ex = max_b[0] - min_b[0];
            for (uint8_t d = 1; d < 3; ++d) {
                if (max_b[d] - min_b[d] > max_ex) {
                    max_ex = max_b[d] - min_b[d];
                    best_dim = d;
                }
            }

            size_t left_buckets = calc_left_buckets(current_buckets);
            size_t left_points = left_buckets * cfg.leaf_size;
            size_t mid = start + left_points;

            if constexpr(!preordered)nth_element(
                temp_pts.begin() + start, 
                temp_pts.begin() + mid, 
                temp_pts.begin() + end,
                [best_dim](const FatPoint& a, const FatPoint& b) {
                    return a.coords[best_dim] < b.coords[best_dim];
                    //if(a.coords[best_dim] < b.coords[best_dim]) return true;
                    //else if(a.coords[best_dim] > b.coords[best_dim])return false;
                    //else [[unlikely]] return (a.payload_id < b.payload_id);
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

        std::vector<scalar_t> vals(B > 0 ? B - 1 : 0);
        std::vector<uint64_t> dims((vals.size() + 31) / 32, 0);
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

        std::vector<scalar_t> vals(B > 0 ? B - 1 : 0);
        std::vector<uint64_t> dims((vals.size() + 31) / 32, 0);
        std::vector<LeafBucket> buckets(B);

        Builder<true> builder{temp_pts, vals, dims, buckets, B};

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
     * @return KdTreeView<LEAF_SIZE> The view of this tree after building.
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
     * @param ro 3-scalar_t array representing the ray origin.
     * @param rd 3-scalar_t array representing the ray direction.
     * @param max_t The maximum traversal distance along the ray.
     * @param radius Optional. Represents the mathematical thickness of the ray (Cylinder query).
     * @return The closest point hit, or error codes.
     */
    std::expected<RayHit, error_t> query_ray(const point_t ro, const point_t rl, scalar_t max_t=Limits::INF, scalar_t radius={}) const noexcept {
        return view().query_ray(ro,rl,max_t,radius);
    }

    /**
     * @brief Forwarding method to find the single nearest neighbor (1-NN) for a given target.
     * 
     * @param target 3-scalar_t array representing the query coordinates.
     * @return The closest point found, or error codes.
     */
    std::expected<KnnResult, error_t>  query_1nn(const point_t target) const noexcept {
        return view().query_1nn(target);
    }

    /**
     * @brief Forwarding method to find the distance (squared) from the nearest neighbor (1-NN) of a given target.
     * 
     * @param target 3-scalar_t array representing the query coordinates.
     * @return The closest point found, or error codes.
     */
    std::expected<distance_t, error_t>  query_distance2(const point_t target) const noexcept {
        return view().query_distance2(target);
    }

    /**
     * @brief Forwarding method to find the k-nearest neighbors (k-NN) for a given target.
     * 
     * @param target 3-scalar_t array representing the query coordinates.
     * @param buffer A pre-allocated span used to store and manage the max-heap of results. 
     * @return A subspan of the buffer containing the found results, sorted from nearest to furthest. Or error codes.
     */
    std::expected<std::span<KnnResult>, error_t> query_knn(const point_t target, std::span<KnnResult> buffer) const noexcept {
        return view().query_knn(target, buffer);
    }

    void set_bbox(point_t min, point_t max){
        min_root=min;
        max_root=max;
    }
};

}