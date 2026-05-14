#pragma once
/**
 * @file kd3-c.h
 * @author karurochari
 * @brief C header for the vec3 kd-tree
 * @date 2026-05-06
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <stddef.h>
#include <stdint.h>

#ifndef KD3_NS
#   define KD3_NS kd3
#endif

#define KD3$_1(ns, x) ns ## _ ## x
#define KD3$_2(ns, x) KD3$_1(ns,x)

#define KD3$(x) KD3$_2(KD3_NS,x)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Matching kd3::Point
 * 
 */
typedef struct {
    float coords[3];
    uint32_t payload_id;
} KD3$(point_t);

/**
 * @brief Matching kd3::KnnResult
 * 
 */
typedef struct {
    float dist_sq;
    uint32_t payload_id;
} KD3$(knn_result_t);

/**
 * @brief Matching kd3::RayHit
 * 
 */
typedef struct{ 
    float t;
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
 * @param error   Optional output pointer, receives 0 on success or 1 (e.g., empty input).
 * 
 * @return kd3_tree_t* A non-NULL handle on success, NULL on failure.
 */
KD3$(tree_t) *KD3$(tree_create)(const KD3$(point_t) *points,
                                size_t npoints,
                                int *error);

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
 * @param target 3-float array with query coordinates.
 * @param out    Pointer to a result struct that will be filled with the best match.
 * 
 * @return int 0 on success, 1 if the tree is empty or invalid.
 */
int KD3$(tree_query_1nn)(const KD3$(tree_t) *tree,
                         const float target[3],
                         KD3$(knn_result_t) *out);

/**
 * @brief Find the k-nearest neighbors (k-NN) for a given target.
 *
 * @param tree    The tree handle.
 * @param target  3-float array with query coordinates.
 * @param results Caller-allocated linear array of at least `k` elements to store results.
 * @param k       Number of neighbours to return (k > 0).
 * 
 * @return size_t The number of valid results found (may be < k if the tree holds fewer points than k).
 */
size_t KD3$(tree_query_knn)(const KD3$(tree_t) *tree,
                            const float target[3],
                            KD3$(knn_result_t) *results,
                            size_t k);

#ifdef __cplusplus
}
#endif
