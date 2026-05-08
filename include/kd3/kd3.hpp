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
#include <omp.h>

#include "query.hpp"

namespace kd3 {

using std::sort;
using std::nth_element;

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
                    buckets[bucket_idx].ids[i] = temp_pts[start + i].payload_id;
                }
                
                // Fill the remainder with pseudo-infinity points
                for (size_t i = size; i < LeafSize; ++i) {
                    buckets[bucket_idx].x[i] = INF; // Extremely far away, 1e15^2 = 1e30 < 3.4e38 (float max)
                    buckets[bucket_idx].y[i] = INF;
                    buckets[bucket_idx].z[i] = INF;
                    buckets[bucket_idx].ids[i] = static_cast<uint32_t>(-1);
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