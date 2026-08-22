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
#include "build.hpp"

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
    using NodeBox = KdTreeView<Limits,cfg>::NodeBox;
    using error_t = KdTreeView<Limits,cfg>::error_t;

private:
    std::vector<scalar_t> split_vals;
    std::vector<uint64_t> split_dims;
    std::vector<LeafBucket> buckets;
    std::vector<typename KdTreeView<Limits,cfg>::NodeBox> node_boxes;
    point_t min_root;
    point_t max_root;

    KdTree(std::vector<scalar_t> vals, std::vector<uint64_t> dims, std::vector<LeafBucket> bks,
           std::vector<typename KdTreeView<Limits,cfg>::NodeBox> boxes = {})
        : split_vals(std::move(vals)), split_dims(std::move(dims)), buckets(std::move(bks)),
          node_boxes(std::move(boxes)) {
        for (size_t d = 0; d < Limits::D; ++d) {
            min_root[d] = -Limits::INF;
            max_root[d] = Limits::INF;
        }
    }

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
        const size_t B = (temp_pts.size() + cfg.leaf_size - 1) / cfg.leaf_size;
        const size_t dims_per_word = KdTreeView<Limits, cfg>::dims_per_word;
        constexpr bool want_boxes = cfg.has_aabb;
        std::vector<scalar_t> vals(B > 0 ? B - 1 : 0);
        std::vector<uint64_t> dims((vals.size() + dims_per_word - 1) / dims_per_word, 0);
        std::vector<LeafBucket> buckets(B);
        std::vector<typename KdTreeView<Limits,cfg>::NodeBox> boxes(want_boxes ? 2 * B - 1 : 0);
        auto view = build_into<Limits, cfg>(temp_pts, BuildTarget<Limits, cfg>{vals, dims, buckets, boxes});
        if (!view) return std::unexpected(view.error());
        return KdTree(std::move(vals), std::move(dims), std::move(buckets), std::move(boxes));
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
        const size_t B = (temp_pts.size() + cfg.leaf_size - 1) / cfg.leaf_size;
        const size_t dims_per_word = KdTreeView<Limits, cfg>::dims_per_word;
        constexpr bool want_boxes = cfg.has_aabb;
        std::vector<scalar_t> vals(B > 0 ? B - 1 : 0);
        std::vector<uint64_t> dims((vals.size() + dims_per_word - 1) / dims_per_word, 0);
        std::vector<LeafBucket> buckets(B);
        std::vector<typename KdTreeView<Limits,cfg>::NodeBox> boxes(want_boxes ? 2 * B - 1 : 0);
        auto view = build_from_ordered_into<Limits, cfg>(
            std::span<FatPoint>{const_cast<FatPoint*>(temp_pts.data()), temp_pts.size()},
            BuildTarget<Limits, cfg>{vals, dims, buckets, boxes});
        if (!view) return std::unexpected(view.error());
        return KdTree(std::move(vals), std::move(dims), std::move(buckets), std::move(boxes));
    }

    /**
     * @brief Extracts the non-owning view capable of device offload.
     * 
     * @return KdTreeView The view of this tree after building.
     */
    KdTreeView<Limits,cfg> view() const noexcept {
        return KdTreeView<Limits,cfg>(split_vals, split_dims, buckets, min_root, max_root, node_boxes);
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