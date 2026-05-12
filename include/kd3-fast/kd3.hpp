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

namespace kd3_fast {

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
    array3 min_root = {-INF,-INF,-INF};
    array3 max_root = {INF,INF,INF};

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

    template<bool preordered=false>
    struct Builder {
        std::span<Point> temp_pts;
        std::vector<float>& vals;
        std::vector<uint64_t>& dims;
        std::vector<LeafBucket<LeafSize>>& buckets;
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
                }
                
                // Fill the remainder with pseudo-infinity points
                for (size_t i = size; i < LeafSize; ++i) {
                    buckets[bucket_idx].x[i] = INF; // Extremely far away, 1e15^2 = 1e30 < 3.4e38 (float max)
                    buckets[bucket_idx].y[i] = INF;
                    buckets[bucket_idx].z[i] = INF;
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

            if constexpr(!preordered)nth_element(
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
     * @return std::expected<KdTree, KdTreeError> The built tree, or KdTreeError::EmptyInput if the input was empty.
     */
    static std::expected<KdTree, KdTreeError> build_from_ordered(std::span<const Point> temp_pts) {
        if (temp_pts.empty()) return std::unexpected(KdTreeError::EmptyInput);

        size_t n = temp_pts.size();
        size_t B = (n + LeafSize - 1) / LeafSize; 

        std::vector<float> vals(B > 0 ? B - 1 : 0);
        std::vector<uint64_t> dims((vals.size() + 31) / 32, 0);
        std::vector<LeafBucket<LeafSize>> buckets(B);

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
     * @return KdTreeView<LeafSize> The view of this tree after building.
     */
    KdTreeView<LeafSize> view() const noexcept {
        return KdTreeView<LeafSize>(split_vals, split_dims, buckets, min_root, max_root);
    }

    /**
     * @brief Implicit conversion to view.
     */
    operator KdTreeView<LeafSize>() const noexcept {
        return view();
    }

    /**
     * @brief Forwarding method to find the closest intersection between a ray and points in the Kd-Tree.
     * 
     * @param ro 3-float array representing the ray origin.
     * @param rd 3-float array representing the ray direction.
     * @param max_t The maximum traversal distance along the ray.
     * @param radius Optional. Represents the mathematical thickness of the ray (Cylinder query).
     * @return std::optional<RayHit> Distance of the closest point hit, or std::nullopt if none.
     */
    std::optional<float> query_ray(const array3 ro, const array3 rl, float max_t=INF, float radius=0.0f) const noexcept {
        return view().query_ray(ro,rl,max_t,radius);
    }

    /**
     * @brief Forwarding method to find the single nearest neighbor (1-NN) for a given target.
     * 
     * @param target 3-float array representing the query coordinates.
     * @return std::optional<float> Distance of the closest point found, or std::nullopt if the tree is empty.
     */
    std::optional<float> query_distance(const array3 target) const noexcept {
        return view().query_distance(target);
    }

    void set_bbox(array3 min, array3 max){
        min_root=min;
        max_root=max;
    }
};

}