/**
 * @file kd3-c.h
 * @author karurochari
 * @brief C header for the vec3 kd-tree
 * @date 2026-05-06
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef KD3_NS
#   define KD3_NS kd3
#   pragma once
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define KD3$_1(ns, x) ns ## _ ## x
#define KD3$_2(ns, x) KD3$_1(ns,x)

#define KD3$(x) KD3$_2(KD3_NS,x)

#ifdef __cplusplus
extern "C" {
#endif

#ifndef KD3_DIMENSIONS
#   define KD3_DIMENSIONS 3
#endif

#ifndef KD3_SIMD_PARALLELISM
    #if defined(__AVX512F__)
        #define  KD3_SIMD_PARALLELISM  16;
    #elif defined(__AVX2__) || defined(__AVX__)
        #define  KD3_SIMD_PARALLELISM 8;
    #elif defined(__SSE4_2__) || defined(__SSE2__)
        #define  KD3_SIMD_PARALLELISM 4;
    #else
        #define  KD3_SIMD_PARALLELISM 1;
    #endif
#endif

#ifndef KD3_BASE_TYPE
#   define KD3_BASE_TYPE float
#endif

#ifndef KD3_DISTANCE_TYPE
#   define KD3_DISTANCE_TYPE float
#endif

#ifndef KD3_LEAF_SIZE
#   define KD3_LEAF_SIZE 32 //TODO: actually it shoul depend on KD3_SIMD_PARALLELISM
#endif

#ifndef KD3_HAS_INDEX
#   define KD3_HAS_INDEX true
#endif

#ifndef KD3_THRES_THREAD
#   define KD3_THRES_THREAD 10000
#endif

#ifndef KD3_MAX_STACK_DEPTH
#   define KD3_MAX_STACK_DEPTH (48*2)
#endif

typedef KD3_BASE_TYPE     KD3$(scalar_t);
typedef KD3_DISTANCE_TYPE KD3$(distance_t);

/**
 * @brief Error states which can be returned by the library.
 * 
 */
typedef enum  {
    KD3$(Ok),
    KD3$(EmptyInput),
    KD3$(EmptyContainer),
    KD3$(NotSupported),
    KD3$(NotImplemented),
    KD3$(NotFound)
} KD3$(error_t);

/**
 * @brief Matching kd3::Point
 * 
 */
typedef struct {
    KD3$(scalar_t) coords[KD3_DIMENSIONS];
    uint32_t payload_id;
} KD3$(point_t);

/**
 * @brief Matching kd3::KnnResult
 * 
 */
typedef struct {
    KD3$(distance_t) dist_sq;
    uint32_t payload_id;
} KD3$(knn_result_t);

/**
 * @brief Matching kd3::RayHit
 * 
 */
typedef struct{ 
    KD3$(scalar_t) t;
    uint32_t payload_id;
} KD3$(ray_hit_t);

/**
 * @brief Opaque handle to a built kd-tree
 */
typedef struct KD3$(tree_t) KD3$(tree_t);

/**
 * @brief Build a kd-tree from a given array of points.
 *
 * @param points  Pointer to an array of `kd3_point_t`.
 * @param npoints Number of points in the array.
 * @param error   Optional output pointer, receives 0 on success or >0 on error.
 * 
 * @return kd3_tree_t* A non-NULL handle on success, NULL on failure.
 */
KD3$(tree_t) *KD3$(tree_create)(const KD3$(point_t) *points,
                                size_t npoints,
                                KD3$(error_t) *error);

/**
 * @brief Destroy a tree created with `kd3_tree_create`.
 * 
 * @param tree Pointer to the tree handle to destroy.
 */
void KD3$(tree_destroy)(KD3$(tree_t) *tree);

/**
 * @brief Find the single nearest neighbor (1-NN) for a given target.
 *
 * @param tree   The tree handle.
 * @param target Array with query coordinates.
 * @param out    Pointer to a result struct that will be filled with the best match.
 * 
 * @return kd3_error_t
 */
KD3$(error_t) KD3$(tree_query_1nn)(const KD3$(tree_t) *tree,
                                   const KD3$(scalar_t) target[KD3_DIMENSIONS],
                                   KD3$(knn_result_t) *out);

/**
 * @brief Find the distance squared to the nearest neighbor (1-NN) of a given target.
 *
 * @param tree   The tree handle.
 * @param target Array with query coordinates.
 * @param out    Pointer to a variable where to store the result.
 * 
 * @return kd3_error_t
 */
KD3$(error_t) KD3$(tree_query_distance2)(const KD3$(tree_t) *tree,
                                         const KD3$(scalar_t) target[KD3_DIMENSIONS],
                                         KD3$(distance_t) *out);

/**
 * @brief Find the k-nearest neighbors (k-NN) for a given target.
 *
 * @param tree    The tree handle.
 * @param target  Array with query coordinates.
 * @param results Caller-allocated linear array of at least `k` elements to store results.
 * @param k       Number of neighbours to return (k > 0), later overridden with their actual number.
 * 
 * @return kd3_error_t
 */
KD3$(error_t) KD3$(tree_query_knn)(const KD3$(tree_t) *tree,
                                   const KD3$(scalar_t) target[KD3_DIMENSIONS],
                                   KD3$(knn_result_t) *results,
                                   size_t *k);

/**
 * @brief Find the closest intersection between a ray and points in the Kd-Tree.
 *
 * @param tree   The tree handle.
 * @param ro     Ray origin coordinates.
 * @param rd     Ray direction coordinates.
 * @param max_t  The maximum traversal distance along the ray.
 * @param radius Optional. Represents the mathematical thickness of the ray.
 * @param out    Pointer to a result struct that will be filled with the best match.
 * 
 * @return kd3_error_t
 */
KD3$(error_t) KD3$(tree_query_ray)(const KD3$(tree_t) *tree,
                                   const KD3$(scalar_t) ro[KD3_DIMENSIONS],
                                   const KD3$(scalar_t) rd[KD3_DIMENSIONS],
                                   KD3$(scalar_t) max_t,
                                   KD3$(scalar_t) radius,
                                   KD3$(ray_hit_t) *out);

/**
 * @brief Find the distance to the closest intersection between a ray and points in the Kd-Tree (ignores index).
 *
 * @param tree   The tree handle.
 * @param ro     Ray origin coordinates.
 * @param rd     Ray direction coordinates.
 * @param max_t  The maximum traversal distance along the ray.
 * @param radius Optional. Represents the mathematical thickness of the ray.
 * @param out    Pointer to a variable where to store the resulting distance.
 * 
 * @return kd3_error_t
 */
KD3$(error_t) KD3$(tree_query_ray_distance)(const KD3$(tree_t) *tree,
                                            const KD3$(scalar_t) ro[KD3_DIMENSIONS],
                                            const KD3$(scalar_t) rd[KD3_DIMENSIONS],
                                            KD3$(scalar_t) max_t,
                                            KD3$(scalar_t) radius,
                                            KD3$(distance_t) *out);


#ifdef __cplusplus
}
#endif

#ifdef KD3_CXX_IMPL
#include <kd3/kd3.hpp>
#include <memory>

constexpr kd3::cfg_t cfg{
    .max_stack_depth = KD3_MAX_STACK_DEPTH,
    .thres_thread = KD3_THRES_THREAD,
    .leaf_size = KD3_LEAF_SIZE,
    .has_index = KD3_HAS_INDEX
};

using Tree = kd3::KdTree<kd3::limits<KD3_BASE_TYPE, KD3_DIMENSIONS, KD3_DISTANCE_TYPE>, cfg>;

namespace {
    /**
     * @brief Helper to hold the C++ object while exposing a C‑compatible opaque pointer
     * 
     */
    struct TreeHandle {
        std::unique_ptr<Tree> tree;
    };
}

KD3$(tree_t) *KD3$(tree_create)(const KD3$(point_t) *points,
                                size_t npoints,
                                KD3$(error_t) *error)
{
    if (!points || npoints == 0) {
        if (error) *error = KD3$(EmptyInput);
        return nullptr;
    }
    
    // Convert C points to the C++ type (no allocation, just a view)
    std::span<const Tree::FatPoint> span((const Tree::FatPoint*)points, npoints);

    // Casting away const since KdTree::build mutates the array to sort it. 
    // This maintains the original behavior where the C array was mutated.
    auto maybe = Tree::build(std::span<Tree::FatPoint>(const_cast<Tree::FatPoint*>(span.data()), span.size()));
    if (!maybe) {
        if (error) *error = (KD3$(error_t))maybe.error();
        return nullptr;
    }

    auto *handle = new TreeHandle;
    handle->tree = std::make_unique<Tree>(std::move(*maybe));
    if (error) *error = KD3$(Ok);
    return reinterpret_cast<KD3$(tree_t)*>(handle);
}

void KD3$(tree_destroy)(KD3$(tree_t) *tree)
{
    delete reinterpret_cast<TreeHandle*>(tree);
}

KD3$(error_t) KD3$(tree_query_ray)(const KD3$(tree_t) *tree,
                                   const KD3$(scalar_t) ro[KD3_DIMENSIONS],
                                   const KD3$(scalar_t) rd[KD3_DIMENSIONS],
                                   KD3$(scalar_t) max_t,
                                   KD3$(scalar_t) radius,
                                   KD3$(ray_hit_t) *out)
{
    if (!tree || !out) return KD3$(EmptyInput);

    const auto *handle = reinterpret_cast<const TreeHandle*>(tree);
    Tree::point_t p_ro, p_rd;
    for(size_t i=0; i<KD3_DIMENSIONS; ++i) { p_ro[i] = ro[i]; p_rd[i] = rd[i]; }

    auto opt = handle->tree->query_ray(p_ro, p_rd, max_t, radius);
    if (!opt) return (KD3$(error_t))opt.error();

    out->t = opt->t;
    out->payload_id = opt->payload_id;
    return KD3$(Ok);
}

KD3$(error_t) KD3$(tree_query_ray_distance)(const KD3$(tree_t) *tree,
                                            const KD3$(scalar_t) ro[KD3_DIMENSIONS],
                                            const KD3$(scalar_t) rd[KD3_DIMENSIONS],
                                            KD3$(scalar_t) max_t,
                                            KD3$(scalar_t) radius,
                                            KD3$(distance_t) *out)
{
    if (!tree || !out) return KD3$(EmptyInput);

    const auto *handle = reinterpret_cast<const TreeHandle*>(tree);
    Tree::point_t p_ro, p_rd;
    for(size_t i=0; i<KD3_DIMENSIONS; ++i) { p_ro[i] = ro[i]; p_rd[i] = rd[i]; }

    auto opt = handle->tree->query_ray_distance(p_ro, p_rd, max_t, radius);
    if (!opt) return (KD3$(error_t))opt.error();

    *out = *opt;
    return KD3$(Ok);
}

KD3$(error_t) KD3$(tree_query_1nn)(const KD3$(tree_t) *tree,
                                   const KD3$(scalar_t) target[KD3_DIMENSIONS],
                                   KD3$(knn_result_t) *out)
{
    if (!tree || !out) return KD3$(EmptyInput);

    const auto *handle = reinterpret_cast<const TreeHandle*>(tree);
    Tree::point_t pt;
    for(size_t i=0; i<KD3_DIMENSIONS; ++i) pt[i] = target[i];

    auto opt = handle->tree->query_1nn(pt);
    if (!opt) return (KD3$(error_t))opt.error();

    out->dist_sq = opt->dist_sq;
    out->payload_id = opt->payload_id;
    return KD3$(Ok);
}

KD3$(error_t) KD3$(tree_query_distance2)(const KD3$(tree_t) *tree,
                                         const KD3$(scalar_t) target[KD3_DIMENSIONS],
                                         KD3$(distance_t) *out)
{
    if (!tree || !out) return KD3$(EmptyInput);

    const auto *handle = reinterpret_cast<const TreeHandle*>(tree);
    Tree::point_t pt;
    for(size_t i=0; i<KD3_DIMENSIONS; ++i) pt[i] = target[i];

    auto opt = handle->tree->query_distance2(pt);
    if (!opt) return (KD3$(error_t))opt.error();

    *out = *opt;
    return KD3$(Ok);
}

KD3$(error_t) KD3$(tree_query_knn)(const KD3$(tree_t) *tree,
                                   const KD3$(scalar_t) target[KD3_DIMENSIONS],
                                   KD3$(knn_result_t) *results,
                                   size_t* k)
{
    if (!tree || k == nullptr || !results) { 
        if (k) *k = 0; 
        return KD3$(EmptyInput);
    }

    const auto *handle = reinterpret_cast<const TreeHandle*>(tree);
    Tree::point_t pt;
    for(size_t i=0; i<KD3_DIMENSIONS; ++i) pt[i] = target[i];

    // KnnResult has identical memory layout to kd3_knn_result_t.
    auto ret = handle->tree->query_knn(pt, {reinterpret_cast<Tree::KnnResult*>(results), *k});
    if(!ret.has_value()){
        *k = 0;
        return (KD3$(error_t))ret.error();
    }
    *k = ret.value().size();
    return KD3$(Ok);
}

#undef KD3_CXX_IMPL

#endif

#undef KD3_NS
#undef KD3$_1
#undef KD3$_2
#undef KD3$
#undef KD3_SIMD_PARALLELISM
#undef KD3_BASE_TYPE
#undef KD3_DISTANCE_TYPE
#undef KD3_LEAF_SIZE
#undef KD3_HAS_INDEX
#undef KD3_THRES_THREAD
#undef KD3_MAX_STACK_DEPTH
#undef KD3_DIMENSIONS
