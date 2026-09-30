// doctest suite for the kd3 C interface (stb-style, KD3_CXX_IMPL).
// Validates tree_create/destroy, 1-NN / distance2 / k-NN / ray queries against
// brute force, error handling, and the GLSL helper entry points.
#define KD3_CXX_IMPL
#include <kd3/kd3-c.h>

#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

using CPt = kd3_point_t;  // { float coords[3]; uint32_t payload_id; }

static std::vector<CPt> make_points(std::mt19937& g, std::size_t n) {
    std::uniform_real_distribution<float> d(-1000.0f, 1000.0f);
    std::vector<CPt> p(n);
    for (std::size_t i = 0; i < n; ++i)
        p[i] = {{d(g), d(g), d(g)}, static_cast<uint32_t>(i)};
    return p;
}

static float brute_1nn(const std::vector<CPt>& pts, const float t[3]) {
    float best = std::numeric_limits<float>::max();
    for (const auto& p : pts) {
        float s = 0.0f;
        for (int d = 0; d < 3; ++d) { const float x = t[d] - p.coords[d]; s += x * x; }
        best = std::min(best, s);
    }
    return best;
}

static std::vector<kd3_knn_result_t> brute_knn(const std::vector<CPt>& pts, const float t[3],
                                               std::size_t k) {
    std::vector<kd3_knn_result_t> v;
    v.reserve(pts.size());
    for (const auto& p : pts) {
        float s = 0.0f;
        for (int d = 0; d < 3; ++d) { const float x = t[d] - p.coords[d]; s += x * x; }
        v.push_back({s, p.payload_id});
    }
    std::sort(v.begin(), v.end(),
              [](const kd3_knn_result_t& a, const kd3_knn_result_t& b) { return a.dist_sq < b.dist_sq; });
    v.resize(std::min(k, v.size()));
    return v;
}

static bool brute_ray(const std::vector<CPt>& pts, const float ro[3], const float rd[3],
                      float max_t, float radius, float& out_t) {
    float rl2 = 0.0f;
    for (int d = 0; d < 3; ++d) rl2 += rd[d] * rd[d];
    if (rl2 <= 0.0f) return false;
    const float inv = 1.0f / rl2;
    const float r2 = radius * radius;
    const float eps = radius > 0.0f ? 0.0f : 1e-5f;
    float best = max_t;
    bool hit = false;
    for (const auto& p : pts) {
        float tn = 0.0f;
        for (int d = 0; d < 3; ++d) tn += (p.coords[d] - ro[d]) * rd[d];
        const float t = tn * inv;
        if (t < 0.0f || t >= best) continue;
        float d2 = 0.0f;
        for (int d = 0; d < 3; ++d) { const float px = ro[d] + t * rd[d]; const float diff = p.coords[d] - px; d2 += diff * diff; }
        if (d2 <= r2 + eps) { best = t; hit = true; }
    }
    out_t = best;
    return hit;
}

}  // namespace

TEST_CASE("C API: error handling and lifecycle") {
    kd3_error_t e = kd3_NotFound;
    CHECK(kd3_tree_create(nullptr, 0, &e) == nullptr);
    CHECK(e == kd3_EmptyInput);

    CPt one = {{0, 0, 0}, 7};
    CHECK(kd3_tree_create(&one, 0, &e) == nullptr);
    CHECK(e == kd3_EmptyInput);

    auto* t = kd3_tree_create(&one, 1, &e);
    REQUIRE(t != nullptr);
    CHECK(e == kd3_Ok);
    kd3_tree_destroy(t);
    kd3_tree_destroy(nullptr);  // must not crash
}

TEST_CASE("C API: 1-NN and distance2 match brute force") {
    std::mt19937 g(11);
    for (std::size_t n : {1u, 7u, 40u, 257u, 5000u}) {
        std::vector<CPt> orig = make_points(g, n);
        std::vector<CPt> pts = orig;  // tree_create mutates its input

        kd3_error_t e;
        auto* t = kd3_tree_create(pts.data(), pts.size(), &e);
        REQUIRE(t != nullptr);

        for (int q = 0; q < 200; ++q) {
            float target[3];
            std::uniform_real_distribution<float> d(-1000.0f, 1000.0f);
            for (float& x : target) x = d(g);

            const float b = brute_1nn(orig, target);

            kd3_knn_result_t r;
            REQUIRE(kd3_tree_query_1nn(t, target, &r) == kd3_Ok);
            CHECK(std::fabs(r.dist_sq - b) <= 1e-4f * std::max(1.0f, b));

            float d2;
            REQUIRE(kd3_tree_query_distance2(t, target, &d2) == kd3_Ok);
            CHECK(std::fabs(d2 - b) <= 1e-4f * std::max(1.0f, b));
        }
        kd3_tree_destroy(t);
    }
}

TEST_CASE("C API: k-NN matches brute force") {
    std::mt19937 g(12);
    std::uniform_real_distribution<float> d(-1000.0f, 1000.0f);

    std::vector<CPt> orig = make_points(g, 3000);
    std::vector<CPt> pts = orig;
    kd3_error_t e;
    auto* t = kd3_tree_create(pts.data(), pts.size(), &e);
    REQUIRE(t != nullptr);

    for (std::size_t k : {1u, 3u, 10u, 64u}) {
        for (int q = 0; q < 50; ++q) {
            float target[3];
            for (float& x : target) x = d(g);

            std::vector<kd3_knn_result_t> got(k);
            std::size_t kk = k;
            REQUIRE(kd3_tree_query_knn(t, target, got.data(), &kk) == kd3_Ok);

            const auto b = brute_knn(orig, target, k);
            REQUIRE(kk == b.size());
            for (std::size_t j = 0; j < b.size(); ++j)
                CHECK(std::fabs(got[j].dist_sq - b[j].dist_sq) <= 1e-4f * std::max(1.0f, b[j].dist_sq));
        }
    }
    kd3_tree_destroy(t);
}

TEST_CASE("C API: ray queries match brute force") {
    std::mt19937 g(13);
    std::uniform_real_distribution<float> d(-1000.0f, 1000.0f);

    std::vector<CPt> orig = make_points(g, 2000);
    std::vector<CPt> pts = orig;
    kd3_error_t e;
    auto* t = kd3_tree_create(pts.data(), pts.size(), &e);
    REQUIRE(t != nullptr);

    for (float radius : {0.0f, 5.0f}) {
        for (int q = 0; q < 400; ++q) {
            float ro[3] = {d(g), d(g), d(g)};
            float rd[3];
            float l = 0.0f;
            for (float& x : rd) { x = d(g); l += x * x; }
            l = std::sqrt(l);
            for (float& x : rd) x /= l;
            const float max_t = 4000.0f;

            float bt;
            const bool bhit = brute_ray(orig, ro, rd, max_t, radius, bt);

            kd3_ray_hit_t hit;
            const bool ghit = kd3_tree_query_ray(t, ro, rd, max_t, radius, &hit) == kd3_Ok;
            REQUIRE(ghit == bhit);
            if (bhit) CHECK(std::fabs(hit.t - bt) <= 1e-3f * std::max(1.0f, bt));

            float gd;
            const bool gdhit = kd3_tree_query_ray_distance(t, ro, rd, max_t, radius, &gd) == kd3_Ok;
            REQUIRE(gdhit == bhit);
            if (bhit) CHECK(std::fabs(gd - bt) <= 1e-3f * std::max(1.0f, bt));
        }
    }
    kd3_tree_destroy(t);
}

TEST_CASE("C API: GLSL generation smoke") {
    kd3_glsl_config_t gc = kd3_glsl_default_config();
    CHECK(gc.D == 3);
    char* s = kd3_glsl_generate_string(&gc);
    REQUIRE(s != nullptr);
    CHECK(std::string(s).find("kd3_query_1nn") != std::string::npos);
    kd3_free_string(s);
}
