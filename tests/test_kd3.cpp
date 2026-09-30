// doctest suite for kd3: API smoke, exactness vs brute force, edge cases, and
// a bounded differential fuzzer.
//
// Build target: `tests` (see xmake.lua). Compiled with -fno-tree-vectorize to
// dodge the known GCC auto-vectorizer bug on tiny trees (see README).
#include <doctest/doctest.h>

#include <kd3/kd3.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

namespace {

template <std::size_t D>
using Pt = std::array<float, D>;

using Limits2 = kd3::limits<float, 2>;
using Limits3 = kd3::limits<float, 3>;
using Limits4 = kd3::limits<float, 4>;

template <class Tree>
using PtOf = typename Tree::point_t;

template <class Tree>
static constexpr std::size_t Dim = std::tuple_size_v<typename Tree::point_t>;

static bool close(float a, float b, float rel = 1e-4f) {
    return std::fabs(a - b) <= rel * std::max(1.0f, std::fabs(a));
}

template <std::size_t D>
static std::vector<Pt<D>> make_coords(std::mt19937& g, std::size_t n, float lo = -1000.0f,
                                      float hi = 1000.0f) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<Pt<D>> v(n);
    for (auto& p : v)
        for (auto& x : p) x = d(g);
    return v;
}

using BruteKnn = std::vector<std::pair<float, std::uint32_t>>;

template <std::size_t D>
static BruteKnn brute_knn(const std::vector<Pt<D>>& c, const Pt<D>& t, std::size_t k) {
    std::vector<std::pair<float, std::uint32_t>> all(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) {
        float s = 0.0f;
        for (std::size_t d = 0; d < D; ++d) {
            const float diff = t[d] - c[i][d];
            s += diff * diff;
        }
        all[i] = {s, static_cast<std::uint32_t>(i)};
    }
    k = std::min(k, all.size());
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    all.resize(k);
    return all;
}

// Mirrors kd3::KdTreeView::ray_impl point test.
template <std::size_t D>
static bool brute_ray(const std::vector<Pt<D>>& c, const Pt<D>& ro, const Pt<D>& rd, float max_t,
                      float radius, float& out_t, std::uint32_t& out_id) {
    float rl2 = 0.0f;
    for (std::size_t d = 0; d < D; ++d) rl2 += rd[d] * rd[d];
    if (rl2 <= 0.0f) return false;
    const float inv = 1.0f / rl2;
    const float r2 = radius * radius;
    const float eps = radius > 0.0f ? 0.0f : 1e-5f;
    float best = max_t;
    std::uint32_t id = std::numeric_limits<std::uint32_t>::max();
    bool hit = false;
    for (std::size_t i = 0; i < c.size(); ++i) {
        float tn = 0.0f;
        for (std::size_t d = 0; d < D; ++d) tn += (c[i][d] - ro[d]) * rd[d];
        const float t = tn * inv;
        if (t < 0.0f || t >= best) continue;
        float d2 = 0.0f;
        for (std::size_t d = 0; d < D; ++d) {
            const float px = ro[d] + t * rd[d];
            const float diff = c[i][d] - px;
            d2 += diff * diff;
        }
        if (d2 <= r2 + eps) {
            best = t;
            id = static_cast<std::uint32_t>(i);
            hit = true;
        }
    }
    out_t = best;
    out_id = id;
    return hit;
}

template <class Tree>
static void check_1nn(const std::vector<PtOf<Tree>>& coords, const std::vector<PtOf<Tree>>& qs) {
    std::vector<typename Tree::FatPoint> fp(coords.size());
    for (std::size_t i = 0; i < coords.size(); ++i)
        fp[i] = {coords[i], static_cast<std::uint32_t>(i)};
    auto tree = Tree::build(fp);
    REQUIRE(tree.has_value());
    for (const auto& t : qs) {
        auto res = tree->query_1nn_inline(t);
        REQUIRE(res.has_value());
        const auto b = brute_knn<Dim<Tree>>(coords, t, 1);
        CHECK(close(res->dist_sq, b[0].first));
    }
}

template <class Tree>
static void check_knn(const std::vector<PtOf<Tree>>& coords, const std::vector<PtOf<Tree>>& qs,
                      std::size_t k) {
    std::vector<typename Tree::FatPoint> fp(coords.size());
    for (std::size_t i = 0; i < coords.size(); ++i)
        fp[i] = {coords[i], static_cast<std::uint32_t>(i)};
    auto tree = Tree::build(fp);
    REQUIRE(tree.has_value());
    std::vector<typename Tree::KnnResult> buf(k);
    for (const auto& t : qs) {
        auto res = tree->query_knn_inline(t, buf);
        REQUIRE(res.has_value());
        const auto b = brute_knn<Dim<Tree>>(coords, t, k);
        REQUIRE(res->size() == b.size());
        for (std::size_t j = 0; j < b.size(); ++j) CHECK(close((*res)[j].dist_sq, b[j].first));
    }
}

template <class Tree>
static void check_ray(const std::vector<PtOf<Tree>>& coords, std::mt19937& g, std::size_t nrays,
                      float radius) {
    using P = PtOf<Tree>;
    std::vector<typename Tree::FatPoint> fp(coords.size());
    for (std::size_t i = 0; i < coords.size(); ++i)
        fp[i] = {coords[i], static_cast<std::uint32_t>(i)};
    auto tree = Tree::build(fp);
    REQUIRE(tree.has_value());

    P mn, mx;
    for (std::size_t d = 0; d < Dim<Tree>; ++d) { mn[d] = 1e30f; mx[d] = -1e30f; }
    for (const auto& p : coords)
        for (std::size_t d = 0; d < Dim<Tree>; ++d) { mn[d] = std::min(mn[d], p[d]); mx[d] = std::max(mx[d], p[d]); }
    float diag = 0.0f;
    for (std::size_t d = 0; d < Dim<Tree>; ++d) diag += (mx[d]-mn[d])*(mx[d]-mn[d]);
    diag = std::sqrt(diag);
    const float max_t = diag;

    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (std::size_t r = 0; r < nrays; ++r) {
        P ro, rd;
        for (std::size_t d = 0; d < Dim<Tree>; ++d) ro[d] = mn[d] + u(g) * (mx[d] - mn[d]);
        float l = 0.0f;
        for (std::size_t d = 0; d < Dim<Tree>; ++d) { rd[d] = nd(g); l += rd[d]*rd[d]; }
        l = std::sqrt(l);
        if (l < 1e-6f) continue;
        for (std::size_t d = 0; d < Dim<Tree>; ++d) rd[d] /= l;

        float bt; std::uint32_t bid;
        const bool bhit = brute_ray<Dim<Tree>>(coords, ro, rd, max_t, radius, bt, bid);

        auto res = tree->query_ray_inline(ro, rd, max_t, radius);
        REQUIRE(res.has_value() == bhit);
        if (bhit) CHECK(close(res->t, bt, 1e-3f));

        auto resd = tree->query_ray_distance_inline(ro, rd, max_t, radius);
        REQUIRE(resd.has_value() == bhit);
        if (bhit) CHECK(close(*resd, bt, 1e-3f));
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// API / smoke
// ---------------------------------------------------------------------------

TEST_CASE("build rejects empty input") {
    kd3::KdTree<Limits3, {.leaf_size = 8}>::FatPoint dummy{};
    std::span<kd3::KdTree<Limits3, {.leaf_size = 8}>::FatPoint> empty(&dummy, 0);
    auto t = kd3::KdTree<Limits3, {.leaf_size = 8}>::build(empty);
    CHECK_FALSE(t.has_value());
    CHECK(t.error() == kd3::KdTree<Limits3, {.leaf_size = 8}>::error_t::EmptyInput);
}

TEST_CASE("README quick-start example") {
    using Tree = kd3::KdTree<kd3::limits<float>, {.leaf_size = 32}>;
    std::vector<Tree::FatPoint> pts = {
        {{0.0f, 0.0f, 0.0f}, 100},
        {{1.0f, 2.5f, -3.0f}, 101},
        {{4.2f, -1.0f, 0.0f}, 102},
    };
    auto t = Tree::build(pts);
    REQUIRE(t.has_value());
    auto r = t->query_1nn({1.0f, 2.0f, -2.0f});
    REQUIRE(r.has_value());
    CHECK(r->payload_id == 101);

    std::array<Tree::KnnResult, 2> buf{};
    auto knn = t->query_knn({1.0f, 2.0f, -2.0f}, buf);
    REQUIRE(knn.has_value());
    CHECK(knn->size() == 2);
}

// ---------------------------------------------------------------------------
// Exactness vs brute force (representative configs)
// ---------------------------------------------------------------------------

TEST_CASE("1-NN matches brute force across configs") {
    using T_plain = kd3::KdTree<Limits3, {.leaf_size = 8}>;
    using T_aabb = kd3::KdTree<Limits3, {.leaf_size = 8, .has_aabb = true}>;
    using T_leaves1 = kd3::KdTree<Limits3, {.leaf_size = 1}>;
    using T_leaves64 = kd3::KdTree<Limits3, {.leaf_size = 64, .has_aabb = true}>;

    std::mt19937 g(1);
    for (std::size_t n : {1u, 7u, 33u, 257u, 4096u}) {
        auto c = make_coords<3>(g, n);
        auto q = make_coords<3>(g, 50);
        check_1nn<T_plain>(c, q);
        check_1nn<T_aabb>(c, q);
        check_1nn<T_leaves1>(c, q);
        check_1nn<T_leaves64>(c, q);
    }
}

TEST_CASE("k-NN matches brute force across configs and k") {
    using T_plain = kd3::KdTree<Limits3, {.leaf_size = 16}>;
    using T_aabb = kd3::KdTree<Limits3, {.leaf_size = 16, .has_aabb = true}>;

    std::mt19937 g(2);
    for (std::size_t n : {1u, 5u, 40u, 1000u}) {
        auto c = make_coords<3>(g, n);
        auto q = make_coords<3>(g, 30);
        for (std::size_t k : {1u, 2u, 5u, 10u, 32u}) {
            check_knn<T_plain>(c, q, k);
            check_knn<T_aabb>(c, q, k);
        }
    }
}

TEST_CASE("distance2 (payload-free) matches brute force") {
    using T = kd3::KdTree<Limits3, {.leaf_size = 16, .has_payload = kd3::cfg_t::has_payload_t::NONE}>;
    std::mt19937 g(3);
    auto c = make_coords<3>(g, 777);
    auto q = make_coords<3>(g, 100);
    std::vector<T::FatPoint> fp(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) fp[i] = {c[i], static_cast<std::uint32_t>(i)};
    auto t = T::build(fp);
    REQUIRE(t.has_value());
    for (const auto& p : q) {
        auto r = t->query_distance2_inline(p);
        REQUIRE(r.has_value());
        CHECK(close(*r, brute_knn<3>(c, p, 1)[0].first));
    }
}

TEST_CASE("ray queries match brute force (radius 0 and >0)") {
    using T_plain = kd3::KdTree<Limits3, {.leaf_size = 8}>;
    using T_aabb = kd3::KdTree<Limits3, {.leaf_size = 8, .has_aabb = true}>;
    std::mt19937 g(4);
    auto c = make_coords<3>(g, 500);
    check_ray<T_plain>(c, g, 400, 0.0f);
    check_ray<T_aabb>(c, g, 400, 0.0f);
    check_ray<T_plain>(c, g, 400, 5.0f);
    check_ray<T_aabb>(c, g, 400, 5.0f);
}

TEST_CASE("dimensionality: 2D and 4D 1-NN match brute force") {
    using T2 = kd3::KdTree<Limits2, {.leaf_size = 8, .has_aabb = true}>;
    using T4 = kd3::KdTree<Limits4, {.leaf_size = 8, .has_aabb = true}>;
    std::mt19937 g(5);
    {
        auto c = make_coords<2>(g, 600);
        auto q = make_coords<2>(g, 80);
        check_1nn<T2>(c, q);
    }
    {
        auto c = make_coords<4>(g, 600);
        auto q = make_coords<4>(g, 80);
        check_1nn<T4>(c, q);
    }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST_CASE("degenerate clouds do not break queries") {
    using Tree = kd3::KdTree<Limits3, {.leaf_size = 16, .has_aabb = true}>;
    std::mt19937 g(7);

    // all-identical points
    {
        auto c = make_coords<3>(g, 200);
        std::fill(c.begin(), c.end(), c[0]);
        auto q = make_coords<3>(g, 50);
        check_1nn<Tree>(c, q);
        check_knn<Tree>(c, q, 8);
    }
    // collinear points
    {
        std::vector<Pt<3>> c(300);
        for (std::size_t i = 0; i < c.size(); ++i) c[i] = {static_cast<float>(i), 0.0f, 0.0f};
        auto q = make_coords<3>(g, 50);
        check_1nn<Tree>(c, q);
    }
}

// ---------------------------------------------------------------------------
// Differential fuzzer (bounded, deterministic)
// ---------------------------------------------------------------------------

template <class Tree>
static void fuzz_scenarios(std::uint32_t seed, int iters, std::size_t max_n) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<std::size_t> dn(1, max_n);
    std::uniform_int_distribution<int> kind(0, 3);

    for (int it = 0; it < iters; ++it) {
        const std::size_t n = dn(g);
        auto c = make_coords<Dim<Tree>>(g, n);
        switch (kind(g)) {
            case 0: if (n > 2) std::fill(c.begin(), c.end(), c[0]); break;            // duplicates
            case 1:                                                                  // collinear-ish
                for (std::size_t i = 0; i < n; ++i) { Pt<Dim<Tree>> p{}; p[0] = static_cast<float>(i); c[i] = p; }
                break;
            case 2: for (auto& p : c) for (auto& x : p) x = std::round(x); break;     // lattice / many ties
            default: break;
        }
        auto q = make_coords<Dim<Tree>>(g, 8);
        check_1nn<Tree>(c, q);
        check_knn<Tree>(c, q, 1 + (std::size_t)kind(g));
        check_ray<Tree>(c, g, 8, 10.0f);
    }
}

TEST_CASE("fuzz: differential vs brute force (leaf 8)") {
    using Tree = kd3::KdTree<Limits3, {.leaf_size = 8}>;
    fuzz_scenarios<Tree>(0xC0FFEEu, 150, 700);
}

TEST_CASE("fuzz: differential vs brute force (leaf 1, aabb)") {
    using Tree = kd3::KdTree<Limits3, {.leaf_size = 1, .has_aabb = true}>;
    fuzz_scenarios<Tree>(0xBEEFu, 120, 500);
}

TEST_CASE("fuzz: differential 2D (leaf 32, aabb)") {
    using Tree = kd3::KdTree<Limits2, {.leaf_size = 32, .has_aabb = true}>;
    fuzz_scenarios<Tree>(0x1234u, 120, 500);
}

TEST_CASE("fuzz: differential 4D (leaf 16)") {
    using Tree = kd3::KdTree<Limits4, {.leaf_size = 16}>;
    fuzz_scenarios<Tree>(0xABCDu, 120, 500);
}
