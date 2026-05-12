#include <kd3/kd3.hpp>
#include <kd3/kd3-c.h>
#include <memory>

namespace {
    /**
     * @brief Helper to hold the C++ object while exposing a C‑compatible opaque pointer
     * 
     */
    struct TreeHandle {
        std::unique_ptr<kd3::KdTree<>> tree;
    };
}
 
kd3_tree_t *kd3_tree_create(const kd3_point_t *points,
                            size_t npoints,
                            int *error)
{
    if (!points || npoints == 0) {
        if (error) *error = 1;          // EmptyInput
        return nullptr;
    }

    // Convert C points to the C++ type (no allocation, just a view)
    std::span<kd3::Point> span((kd3::Point*)points, npoints);

    auto maybe = kd3::KdTree<>::build(span);
    if (!maybe) {
        if (error) *error = 1;          // Currently only EmptyInput
        return nullptr;
    }

    auto *handle = new TreeHandle;
    handle->tree = std::make_unique<kd3::KdTree<>>(std::move(*maybe));
    if (error) *error = 0;
    return reinterpret_cast<kd3_tree_t*>(handle);
}

void kd3_tree_destroy(kd3_tree_t *tree)
{
    delete reinterpret_cast<TreeHandle*>(tree);
}


int kd3_tree_query_ray(const kd3_tree_t *tree,
                       const float ro[3],
                       const float rd[3],
                       float max_t,
                       float radius,
                       kd3_ray_hit_t *out)
{
    if (!tree || !out) return 1;

    const auto *handle = reinterpret_cast<const TreeHandle*>(tree);
    auto opt = handle->tree->query_ray(kd3::array3{ro[0], ro[1], ro[2]},kd3::array3{rd[0], rd[1], rd[2]},max_t,radius);
    if (!opt) return 1;   // empty tree

    out->t     = opt->t;
    out->payload_id  = opt->payload_id;
    return 0;
}

int kd3_tree_query_1nn(const kd3_tree_t *tree,
                       const float target[3],
                       kd3_knn_result_t *out)
{
    if (!tree || !out) return 1;

    const auto *handle = reinterpret_cast<const TreeHandle*>(tree);
    auto opt = handle->tree->query_1nn(kd3::array3{target[0], target[1], target[2]});
    if (!opt) return 1;   // empty tree

    out->dist_sq     = opt->dist_sq;
    out->payload_id  = opt->payload_id;
    return 0;
}

size_t kd3_tree_query_knn(const kd3_tree_t *tree,
                          const float target[3],
                          kd3_knn_result_t *results,
                          size_t k)
{
    if (!tree || k == 0 || !results) return 0;

    const auto *handle = reinterpret_cast<const TreeHandle*>(tree);
    std::span<kd3::KnnResult> cpp_res = handle->tree->query_knn(kd3::array3{target[0], target[1], target[2]}, {(kd3::KnnResult*)results,k});

    return cpp_res.size();
}
