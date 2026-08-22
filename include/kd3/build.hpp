#pragma once
/// @file build.hpp
/// @author karurochari
/// @brief Builder for the kd3 over a span, as the full size is determinable.
/// @date 2026-05-04
/// @copyright Copyright (c) 2026

//   B       = ceil(n / leaf_size)
//   buckets = B
//   vals    = B - 1                     (internal split values)
//   dims    = ceil((B-1) / dims_per_word)
//   boxes   = 2*B - 1                   (per-subtree AABBs, only when cfg.has_aabb)

#include <kd3/query.hpp>   // KdTreeView, limits, cfg_t (vector-free)

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <omp.h>

namespace kd3 {

/// @brief Caller-provided storage the span builder fills.
/// @details `vals` must hold >= B-1 entries, `dims` >= ceil((B-1)/dims_per_word),
/// `buckets` >= B, where B = ceil(n/leaf_size) for the n points being built.
/// When `Cfg.has_aabb` is set, `boxes` must hold >= 2*B-1 entries.
template <typename Limits, cfg_t Cfg>
struct BuildTarget {
    using scalar_t = Limits::scalar_t;
    using LeafBucket = typename KdTreeView<Limits, Cfg>::LeafBucket;
    using NodeBox = typename KdTreeView<Limits, Cfg>::NodeBox;
    std::span<scalar_t>  vals;
    std::span<uint64_t>  dims;
    std::span<LeafBucket> buckets;
    std::span<NodeBox>   boxes;
};

/// Number of internal-tree nodes on the left half of a binary tree with `n`
/// nodes (kd3's balanced split helper).
inline constexpr std::size_t build_left_nodes(std::size_t n) noexcept {
    if (n <= 1) return 0;
    const std::size_t h = std::bit_width(n) - 1;
    const std::size_t perfect_left = (std::size_t(1) << (h - 1)) - 1;
    const std::size_t bottom_level = n - ((std::size_t(1) << h) - 1);
    const std::size_t bottom_left = std::min(bottom_level, std::size_t(1) << (h - 1));
    return perfect_left + bottom_left;
}

/// Number of leaf buckets on the left half of a balanced tree with
/// `total_buckets` leaves.
inline constexpr std::size_t build_left_buckets(std::size_t total_buckets) noexcept {
    if (total_buckets <= 1) return 0;
    return (build_left_nodes(2 * total_buckets - 1) + 1) / 2;
}

/// @brief The KD-tree construction pass: partitions points along the widest
/// axis and fills the split-value / split-dimension / leaf-bucket arrays.
/// @details Operates on preallocated spans (no std::vector). `preordered`
/// skips the nth_element partition (input assumed already sorted by centroid).
template <typename Limits, cfg_t Cfg, bool preordered = false>
struct Builder {
    static constexpr cfg_t cfg = Cfg;

    using scalar_t = Limits::scalar_t;
    using FatPoint = Limits::FatPoint;
    using LeafBucket = typename KdTreeView<Limits, Cfg>::LeafBucket;
    using NodeBox = typename KdTreeView<Limits, Cfg>::NodeBox;

    std::span<FatPoint>  temp_pts;
    std::span<scalar_t>  vals;
    std::span<uint64_t>  dims;
    std::span<LeafBucket> buckets;
    std::span<NodeBox>   boxes;
    std::size_t B;

    inline void set_dim(std::size_t i, uint8_t dim) {
        const std::size_t dims_per_word = KdTreeView<Limits, Cfg>::dims_per_word;
        const std::size_t dim_bits = KdTreeView<Limits, Cfg>::dim_bits;
        const std::size_t block = i / dims_per_word;
        const std::size_t offset = (i % dims_per_word) * dim_bits;
        #pragma omp atomic
        dims[block] |= (static_cast<uint64_t>(dim) << offset);
    }

    void build(std::size_t start, std::size_t end, std::size_t node_idx,
               std::size_t current_buckets) {
        const std::size_t size = end - start;
        if (size == 0) return;

        if (node_idx >= B - 1) {
            const std::size_t bucket_idx = node_idx - (B - 1);
            if constexpr (cfg.has_aabb) {
                scalar_t min_b[Limits::D];
                scalar_t max_b[Limits::D];
                for (std::size_t d = 0; d < Limits::D; ++d)
                    min_b[d] = max_b[d] = temp_pts[start].coords[d];
                for (std::size_t i = start + 1; i < end; ++i)
                    for (std::size_t d = 0; d < Limits::D; ++d) {
                        min_b[d] = std::min(min_b[d], temp_pts[i].coords[d]);
                        max_b[d] = std::max(max_b[d], temp_pts[i].coords[d]);
                    }
                for (std::size_t d = 0; d < Limits::D; ++d) {
                    boxes[node_idx].minc[d] = min_b[d];
                    boxes[node_idx].maxc[d] = max_b[d];
                }
            }
            for (std::size_t i = 0; i < size; ++i) {
                for (std::size_t d = 0; d < Limits::D; ++d)
                    buckets[bucket_idx].coords[d][i] = temp_pts[start + i].coords[d];
                if constexpr (cfg.has_payload == cfg_t::has_payload_t::INDEX)
                    buckets[bucket_idx].ids[i] = temp_pts[start + i].payload_id;
            }
            for (std::size_t i = size; i < cfg.leaf_size; ++i) {
                for (std::size_t d = 0; d < Limits::D; ++d)
                    buckets[bucket_idx].coords[d][i] = Limits::INF;
                if constexpr (cfg.has_payload == cfg_t::has_payload_t::INDEX)
                    buckets[bucket_idx].ids[i] = Limits::INF_IDX;
            }
            return;
        }

        scalar_t min_b[Limits::D];
        scalar_t max_b[Limits::D];
        for (std::size_t d = 0; d < Limits::D; ++d)
            min_b[d] = max_b[d] = temp_pts[start].coords[d];
        for (std::size_t i = start + 1; i < end; ++i)
            for (std::size_t d = 0; d < Limits::D; ++d) {
                min_b[d] = std::min(min_b[d], temp_pts[i].coords[d]);
                max_b[d] = std::max(max_b[d], temp_pts[i].coords[d]);
            }

        if constexpr (cfg.has_aabb) {
            for (std::size_t d = 0; d < Limits::D; ++d) {
                boxes[node_idx].minc[d] = min_b[d];
                boxes[node_idx].maxc[d] = max_b[d];
            }
        }

        std::uint8_t best_dim = 0;
        scalar_t max_ex = max_b[0] - min_b[0];
        for (std::uint8_t d = 1; d < Limits::D; ++d)
            if (max_b[d] - min_b[d] > max_ex) { max_ex = max_b[d] - min_b[d]; best_dim = d; }

        const std::size_t left_buckets = build_left_buckets(current_buckets);
        const std::size_t left_points = left_buckets * cfg.leaf_size;
        const std::size_t mid = start + left_points;

        if constexpr (!preordered)
            std::nth_element(
                temp_pts.begin() + start, temp_pts.begin() + mid, temp_pts.begin() + end,
                [best_dim](const FatPoint& a, const FatPoint& b) {
                    return a.coords[best_dim] < b.coords[best_dim];
                });

        vals[node_idx] = temp_pts[mid].coords[best_dim];
        set_dim(node_idx, best_dim);

        if (size > cfg.thres_thread) {
            #pragma omp task shared(temp_pts, vals, dims, buckets, boxes)
            build(start, mid, 2 * node_idx + 1, left_buckets);
            #pragma omp task shared(temp_pts, vals, dims, buckets, boxes)
            build(mid, end, 2 * node_idx + 2, current_buckets - left_buckets);
            #pragma omp taskwait
        } else {
            build(start, mid, 2 * node_idx + 1, left_buckets);
            build(mid, end, 2 * node_idx + 2, current_buckets - left_buckets);
        }
    }
};

/// @brief Build a KD-tree into caller-provided preallocated spans.
/// @param temp_pts Mutable span of points (mutated / sorted by centroid).
/// @param target   Storage to fill (`vals` >= B-1, `dims` >= ceil((B-1)/dims_per_word),
///                 `buckets` >= B).
/// @return A non-owning KdTreeView over the filled prefix, or an error.
template <typename Limits, cfg_t Cfg>
std::expected<KdTreeView<Limits, Cfg>, typename KdTreeView<Limits, Cfg>::error_t>
build_into(std::span<typename Limits::FatPoint> temp_pts, BuildTarget<Limits, Cfg> target) {
    using error_t = typename KdTreeView<Limits, Cfg>::error_t;
    if (temp_pts.empty()) return std::unexpected(error_t::EmptyInput);
    const std::size_t n = temp_pts.size();
    const std::size_t B = (n + Cfg.leaf_size - 1) / Cfg.leaf_size;
    const std::size_t dims_per_word = KdTreeView<Limits, Cfg>::dims_per_word;
    const std::size_t n_vals = B > 0 ? B - 1 : 0;
    const std::size_t n_dims = (n_vals + dims_per_word - 1) / dims_per_word;
    const std::size_t n_nodes = 2 * B - 1;
    constexpr bool want_boxes = Cfg.has_aabb;
    if (target.vals.size() < n_vals || target.dims.size() < n_dims
        || target.buckets.size() < B)
        return std::unexpected(error_t::InsufficientStorage);
    if constexpr (want_boxes) {
        if (target.boxes.size() < n_nodes)
            return std::unexpected(error_t::InsufficientStorage);
    }

    Builder<Limits, Cfg> builder{temp_pts, target.vals, target.dims, target.buckets,
                                 target.boxes, B};
    #pragma omp parallel
    {
        #pragma omp single
        { builder.build(0, n, 0, B); }
    }
    typename Limits::point_t rmin, rmax;
    for (std::size_t d = 0; d < Limits::D; ++d) { rmin[d] = -Limits::INF; rmax[d] = Limits::INF; }
    return KdTreeView<Limits, Cfg>{target.vals.first(n_vals), target.dims.first(n_dims),
                                   target.buckets.first(B), rmin, rmax,
                                   want_boxes ? target.boxes.first(n_nodes)
                                              : std::span<const typename KdTreeView<Limits, Cfg>::NodeBox>{}};
}

/// @brief Build into preallocated spans from points assumed already ordered.
template <typename Limits, cfg_t Cfg>
std::expected<KdTreeView<Limits, Cfg>, typename KdTreeView<Limits, Cfg>::error_t>
build_from_ordered_into(std::span<typename Limits::FatPoint> temp_pts, BuildTarget<Limits, Cfg> target) {
    using error_t = typename KdTreeView<Limits, Cfg>::error_t;
    if (temp_pts.empty()) return std::unexpected(error_t::EmptyInput);
    const std::size_t n = temp_pts.size();
    const std::size_t B = (n + Cfg.leaf_size - 1) / Cfg.leaf_size;
    const std::size_t dims_per_word = KdTreeView<Limits, Cfg>::dims_per_word;
    const std::size_t n_vals = B > 0 ? B - 1 : 0;
    const std::size_t n_dims = (n_vals + dims_per_word - 1) / dims_per_word;
    const std::size_t n_nodes = 2 * B - 1;
    constexpr bool want_boxes = Cfg.has_aabb;
    if (target.vals.size() < n_vals || target.dims.size() < n_dims
        || target.buckets.size() < B)
        return std::unexpected(error_t::InsufficientStorage);
    if constexpr (want_boxes) {
        if (target.boxes.size() < n_nodes)
            return std::unexpected(error_t::InsufficientStorage);
    }

    Builder<Limits, Cfg, true> builder{temp_pts, target.vals, target.dims, target.buckets,
                                       target.boxes, B};
    #pragma omp parallel
    {
        #pragma omp single
        { builder.build(0, n, 0, B); }
    }
    typename Limits::point_t rmin, rmax;
    for (std::size_t d = 0; d < Limits::D; ++d) { rmin[d] = -Limits::INF; rmax[d] = Limits::INF; }
    return KdTreeView<Limits, Cfg>{target.vals.first(n_vals), target.dims.first(n_dims),
                                   target.buckets.first(B), rmin, rmax,
                                   want_boxes ? target.boxes.first(n_nodes)
                                              : std::span<const typename KdTreeView<Limits, Cfg>::NodeBox>{}};
}

}  // namespace kd3
