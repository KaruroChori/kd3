/**
 * @file render.cpp
 * @author karurochari
 * @brief A very condensed demo to show how this kd-tree can be used for rendering.
 * @version 0.1
 * @date 2026-05-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <raylib.h>
#include <rlgl.h>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <string>

#include <kd3/kd3.hpp>
#include <kd3/glsl.hpp>

using TreeType = kd3::KdTree<kd3::limits<float>,{.leaf_size=32, .has_aabb=true}>;

// ---------------------------------------------------------
// Core Math & Geometry
// ---------------------------------------------------------
struct Vec3 {
    float x, y, z;
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(float s) const { return {x/s, y/s, z/s}; }
    float dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    float length() const { return std::sqrt(dot(*this)); }
    Vec3 normalize() const { float l = length(); return l > 0.0f ? (*this * (1.0f / l)) : Vec3{0,0,0}; }
    Vec3 cross(const Vec3& o) const { return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x}; }

    operator TreeType::point_t() const { return {x,y,z}; }
};

struct AABB {
    Vec3 min{1e9f, 1e9f, 1e9f};
    Vec3 max{-1e9f, -1e9f, -1e9f};
    void grow(const Vec3& p) {
        min = {std::fmin(min.x, p.x), std::fmin(min.y, p.y), std::fmin(min.z, p.z)};
        max = {std::fmax(max.x, p.x), std::fmax(max.y, p.y), std::fmax(max.z, p.z)};
    }
    float dist_sq(const Vec3& p) const {
        float dx = (p.x < min.x) ? (min.x - p.x) : ((p.x > max.x) ? (p.x - max.x) : 0.0f);
        float dy = (p.y < min.y) ? (min.y - p.y) : ((p.y > max.y) ? (p.y - max.y) : 0.0f);
        float dz = (p.z < min.z) ? (min.z - p.z) : ((p.z > max.z) ? (p.z - max.z) : 0.0f);
        return dx*dx + dy*dy + dz*dz;
    }
};

struct Surfel { Vec3 position; Vec3 normal; float radius; };

// ---------------------------------------------------------
// Point-cloud loader: raw float32 `.bin` (x y z triples), or ASCII `.ply` /
// `.xyz` (honors the `element vertex N` count in PLY headers and skips faces).
// ---------------------------------------------------------
static bool load_points(const char* path, size_t max_points, std::vector<Surfel>& out) {
    out.clear();
    const std::string s = path;
    const bool is_bin = s.size() > 4 && s.compare(s.size() - 4, 4, ".bin") == 0;
    FILE* f = std::fopen(path, is_bin ? "rb" : "r");
    if (!f) { std::cerr << "cannot open " << path << "\n"; return false; }

    if (is_bin) {
        std::fseek(f, 0, SEEK_END);
        const long long bytes = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        long long n = bytes / (3 * (long long)sizeof(float));
        if (max_points && n > (long long)max_points) n = (long long)max_points;
        std::vector<float> raw(static_cast<size_t>(n) * 3);
        std::fread(raw.data(), sizeof(float), raw.size(), f);
        out.resize(static_cast<size_t>(n));
        for (long long i = 0; i < n; ++i)
            out[static_cast<size_t>(i)].position = {raw[i*3], raw[i*3+1], raw[i*3+2]};
    } else {
        char buf[512];
        long long limit = max_points ? (long long)max_points : -1;
        bool in_header = false;
        if (fgets(buf, sizeof(buf), f) && strncmp(buf, "ply", 3) == 0) in_header = true;
        else std::rewind(f);
        std::vector<Vec3> pts;
        while (fgets(buf, sizeof(buf), f)) {
            if (in_header) {
                if (strncmp(buf, "end_header", 10) == 0) { in_header = false; continue; }
                if (strncmp(buf, "element vertex", 14) == 0) {
                    const long long vc = atoll(buf + 14);
                    if (limit < 0 || vc < limit) limit = vc;
                }
                continue;
            }
            float x, y, z;
            if (sscanf(buf, "%f %f %f", &x, &y, &z) == 3) {
                pts.push_back({x, y, z});
                if (limit > 0 && (long long)pts.size() >= limit) break;
            }
        }
        out.resize(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) out[i].position = pts[i];
    }
    std::fclose(f);
    return true;
}

// ---------------------------------------------------------
// PCA normal estimation from a raw point cloud, using the kd-tree's own k-NN.
// ---------------------------------------------------------
// Jacobi rotation diagonalization of a symmetric 3x3; returns the eigenvector
// belonging to eigenvalue `which` (0 = largest, 2 = smallest).
inline Vec3 eigvec3(const float m[3][3], int which) {
    // Copy, jacobi-rotate until off-diagonals are negligible.
    float a[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) a[i][j] = m[i][j];
    float v[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    for (int iter = 0; iter < 64; ++iter) {
        int p = 0, q = 1;
        float mx = std::fabs(a[0][1]);
        for (int i = 0; i < 3; ++i) for (int j = i+1; j < 3; ++j)
            if (std::fabs(a[i][j]) > mx) { mx = std::fabs(a[i][j]); p = i; q = j; }
        if (mx < 1e-9f) break;
        const float app = a[p][p], aqq = a[q][q], apq = a[p][q];
        const float theta = 0.5f * (aqq - app) / apq;
        const float t = (theta >= 0 ? 1.0f : -1.0f) / (std::fabs(theta) + std::sqrt(theta*theta + 1.0f));
        const float c = 1.0f / std::sqrt(t*t + 1.0f);
        const float s = t * c;
        for (int i = 0; i < 3; ++i) {
            const float aip = a[i][p], aiq = a[i][q];
            a[i][p] = c*aip - s*aiq;
            a[i][q] = s*aip + c*aiq;
        }
        for (int j = 0; j < 3; ++j) {
            const float apj = a[p][j], aqj = a[q][j];
            a[p][j] = c*apj - s*aqj;
            a[q][j] = s*apj + c*aqj;
        }
        for (int i = 0; i < 3; ++i) {
            const float vip = v[i][p], viq = v[i][q];
            v[i][p] = c*vip - s*viq;
            v[i][q] = s*vip + c*viq;
        }
    }
    // Eigenvalues are now on the diagonal; pick the requested column.
    return {v[0][which], v[1][which], v[2][which]};
}

// Compute the PCA normal at query point q by looking at its k nearest neighbors.
// `surfels` gives the actual neighbor coordinates (via payload_id). Returns the
// outward-ish normal (oriented toward +z for terrain) and the local spacing.
inline Vec3 pca_normal(const Vec3& q, const std::vector<Surfel>& surfels,
                       const kd3::KdTreeView<kd3::limits<float>, TreeType::cfg>& view,
                       float& out_spacing) {
    std::array<TreeType::KnnResult, 1> nb;
    TreeType::point_t qt = q; // via Vec3::operator point_t()
    auto res = view.query_knn(qt, nb);
    if (!res || res->empty()) { out_spacing = 1.0f; return {0,0,1}; }

    float cx = 0, cy = 0, cz = 0;
    for (auto& r : *res) {
        const Vec3& p = surfels[r.payload_id].position;
        cx += p.x; cy += p.y; cz += p.z;
    }
    const float n = static_cast<float>(res->size());
    cx /= n; cy /= n; cz /= n;

    // Covariance matrix.
    float m[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    out_spacing = 0.0f;
    for (auto& r : *res) {
        const Vec3& p = surfels[r.payload_id].position;
        const float dx = p.x - cx, dy = p.y - cy, dz = p.z - cz;
        m[0][0] += dx*dx; m[0][1] += dx*dy; m[0][2] += dx*dz;
        m[1][1] += dy*dy; m[1][2] += dy*dz;
        m[2][2] += dz*dz;
        out_spacing += std::sqrt(r.dist_sq);
    }
    m[1][0] = m[0][1]; m[2][0] = m[0][2]; m[2][1] = m[1][2];
    out_spacing /= n;

    Vec3 normal = eigvec3(m, 2).normalize();
    return normal;
}

// ---------------------------------------------------------
// GPU Data Layout
// ---------------------------------------------------------
struct alignas(16) GpuSurfel {
    float pos[3]; float radius;
    float norm[3]; float pad1;
};

struct QueryResult { float dist; Vec3 normal; };

// Elevation ramp for terrain coloring: low = water/teal, mid = green, high = tan.
inline Vec3 elevation_color(float z, float zmin, float zmax) {
    const float t = std::clamp((z - zmin) / (zmax - zmin + 1e-6f), 0.0f, 1.0f);
    // stops: (low) deep green-blue -> (mid) green -> (high) sandy
    Vec3 c0 = {0.10f, 0.35f, 0.45f};
    Vec3 c1 = {0.25f, 0.70f, 0.35f};
    Vec3 c2 = {0.85f, 0.65f, 0.40f};
    if (t < 0.5f) {
        const float u = t / 0.5f;
        return { c0.x + (c1.x - c0.x)*u, c0.y + (c1.y - c0.y)*u, c0.z + (c1.z - c0.z)*u };
    } else {
        const float u = (t - 0.5f) / 0.5f;
        return { c1.x + (c2.x - c1.x)*u, c1.y + (c2.y - c1.y)*u, c1.z + (c2.z - c1.z)*u };
    }
}

QueryResult evaluate_surfel(const Vec3& p, float dist_to_point_sq, int idx, const std::vector<Surfel>& surfels) {
    if (idx == -1) return {1000.0f, {0,1,0}};
    float exact_dist = std::sqrt(dist_to_point_sq) - (surfels[idx].radius * 0.65f);
    return { exact_dist, surfels[idx].normal };
}


// ---------------------------------------------------------
// Application
// ---------------------------------------------------------
int main(int argc, char** argv) {
    bool shot_mode = false;
    const char* dataset = "private/autzen.bin"; // kept out of datasets/ so the benchmark ignores it
    size_t max_points = 0; // 0 = load all
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shot") shot_mode = true;
        else if (a == "--max" && i + 1 < argc) max_points = static_cast<size_t>(atoll(argv[++i]));
        else dataset = argv[i];
    }
    const int W = 1280, H = 720;
    InitWindow(W, H, "kd3 demo renderer");
    SetTargetFPS(0); // VSYNC off to benchmark "raw" render time

    std::cout << "Loading " << dataset << " ...\n" << std::flush;
    std::vector<Surfel> master_points;
    if (!load_points(dataset, max_points, master_points) || master_points.empty()) {
        std::cerr << "failed to load " << dataset << "\n";
        return 1;
    }
    std::cout << "Loaded " << master_points.size() << " points\n" << std::flush;

    // Center on the point centroid (the cloud is lopsided; bbox center is far
    // from where the mass actually is), scale so the largest horizontal extent
    // is ~2, and exaggerate the vertical (Z) so LiDAR relief reads clearly.
    // Accumulate the centroid in double: 5M coords ~1e5 summed into float loses
    // catastrophic precision (float ULP ~5e5 at that magnitude).
    double sx = 0, sy = 0, sz = 0;
    for (auto& s : master_points) { sx += s.position.x; sy += s.position.y; sz += s.position.z; }
    const double inv = 1.0 / static_cast<double>(master_points.size());
    const Vec3 centroid{ static_cast<float>(sx * inv),
                         static_cast<float>(sy * inv),
                         static_cast<float>(sz * inv) };
    AABB root_aabb;
    for (auto& s : master_points) root_aabb.grow(s.position);
    Vec3 extent = root_aabb.max - root_aabb.min;
    const float scale = 2.0f / std::max(extent.x, extent.y); // fit the largest horizontal extent
    for (auto& s : master_points) {
        Vec3 p = (s.position - centroid) * scale;
        s.position = p;
        s.normal = {0,0,0};
        s.radius = 0.0f;
    }
    root_aabb = AABB();
    for (auto& s : master_points) root_aabb.grow(s.position);
    std::cout << "Normalized to AABB diag " << (root_aabb.max - root_aabb.min).length() << "\n\n";

    // 1. Prepare points for kd3
    std::vector<TreeType::FatPoint> build_points(master_points.size());
    for (size_t i = 0; i < master_points.size(); ++i) {
        build_points[i].coords[0] = master_points[i].position.x;
        build_points[i].coords[1] = master_points[i].position.y;
        build_points[i].coords[2] = master_points[i].position.z;
        build_points[i].payload_id = i;
    }

    // 2. Build KdTree natively utilizing LEAF_SIZE = 8 for 1:1 GLSL SSBO matching
    auto t1 = std::chrono::steady_clock::now();
    auto tree_expected = TreeType::build(build_points);
    if (!tree_expected) {
        std::cerr << "Failed to build tree! Empty input?\n";
        return 1;
    }
    TreeType kdtree = std::move(tree_expected.value());
    auto t2 = std::chrono::steady_clock::now();
    std::cout << "KdTree Built in: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() 
              << " ms\n\n";

    // Obtain Trivially-Copyable View
    auto view = kdtree.view();

    // One-time PCA pass: recover normals + local spacing for every surfel,
    // using the kd-tree's own k-NN. Parallelized with OpenMP.
    {
        const size_t k = 16;
        auto t3 = std::chrono::steady_clock::now();
        std::vector<float> spacing(master_points.size(), 0.0f);
        #pragma omp parallel for schedule(dynamic, 10000)
        for (size_t i = 0; i < master_points.size(); ++i) {
            Vec3 n = pca_normal(master_points[i].position, master_points, view, spacing[i]);
            if (n.z < 0.0f) n = n * -1.0f; // point "up" (+Z, toward the scanner) for terrain
            master_points[i].normal = n;
        }
        // Radius sized from local spacing so splats overlap into a solid surface.
        const float K = 1.5f; // tunable splat multiplier
        for (size_t i = 0; i < master_points.size(); ++i)
            master_points[i].radius = spacing[i] * K;
        auto t4 = std::chrono::steady_clock::now();
        std::cout << "PCA normals for " << master_points.size() << " points in " << std::flush
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count()
                  << " ms (k=" << k << ")\n\n";
    }

    // Generate GPU GLSL Code matching our C++ template exactly
    constexpr kd3::GlslConfig glsl_cfg {
        .binding_vals = 1,
        .binding_dims = 2,
        .binding_bks = 3,
        .binding_boxes = 4,
        .max_stack_depth = 128
    };

    std::string glsl_kd3 = kd3::generate_glsl<kd3::limits<float>, TreeType::cfg, glsl_cfg>();

    const std::string gpu_shader_code = std::string{} + R"(
#version 430 core
#extension GL_ARB_gpu_shader_int64 : require

out vec4 finalColor;

uniform vec3 ro;
uniform vec3 uu;
uniform vec3 vv;
uniform vec3 ww;
uniform float W;
uniform float H;

// Added for the Root BBox optimization
uniform vec3 root_bmin;
uniform vec3 root_bmax;

// --- Structs & Buffers ---
struct Surfel {
    vec3 pos; float radius;
    vec3 norm; float pad;
};

layout(std430, binding = 0) readonly buffer SurfelBuffer { Surfel surfels[]; };

)" + glsl_kd3 + R"(

// --- Raymarching ---
struct QueryResult { float dist; vec3 normal; };

float dist_sq_aabb(vec3 p, vec3 bmin, vec3 bmax) {
    float dx = max(0.0, max(bmin.x - p.x, p.x - bmax.x));
    float dy = max(0.0, max(bmin.y - p.y, p.y - bmax.y));
    float dz = max(0.0, max(bmin.z - p.z, p.z - bmax.z));
    return dx*dx + dy*dy + dz*dz;
}

QueryResult query(vec3 p) {
    // HUGE OPTIMIZATION: Check global bbox to step past empty space!
    float root_dist = dist_sq_aabb(p, root_bmin, root_bmax);
    if (root_dist > 0.001) return QueryResult(sqrt(root_dist), vec3(0,1,0));

    Kd3KnnResult knn;
    if (kd3_query_1nn(p, knn)) {
        float exact_dist = sqrt(knn.dist_sq) - (surfels[knn.payload_id].radius * 0.65);
        return QueryResult(exact_dist, surfels[knn.payload_id].norm);
    }
    return QueryResult(1000.0, vec3(0,1,0));
}

void main() {
    float u = (2.0 * gl_FragCoord.x - W) / H;
    float v = (2.0 * gl_FragCoord.y - H) / H;
    vec3 rd = normalize(uu * u + vv * v + ww);

    float t = 0.0;
    bool hit = false;
    vec3 n = vec3(0.0);
    vec3 p;

    for (int i = 0; i < 64; ++i) {
        p = ro + rd * t;
        QueryResult res = query(p);

        if (res.dist < 0.005) { hit = true; n = res.normal; break; }
        if (t > 8.0) break;
        t += res.dist;
    }

    if (hit) {
        float diff = max(0.1, dot(n, normalize(vec3(0.4, 0.4, 1.0))));
        float zt = clamp((p.z - root_bmin.z) / (root_bmax.z - root_bmin.z + 1e-6), 0.0, 1.0);
        vec3 c0 = vec3(0.10, 0.35, 0.45);
        vec3 c1 = vec3(0.25, 0.70, 0.35);
        vec3 c2 = vec3(0.85, 0.65, 0.40);
        vec3 c = zt < 0.5 ? mix(c0, c1, zt / 0.5) : mix(c1, c2, (zt - 0.5) / 0.5);
        finalColor = vec4(c * diff, 1.0);
    } else {
        finalColor = vec4(30.0/255.0, 30.0/255.0, 45.0/255.0, 1.0);
    }
}
)";

    // 3. Setup GPU Environment
    Shader gpu_shader = LoadShaderFromMemory(nullptr, gpu_shader_code.c_str());
    int loc_ro = GetShaderLocation(gpu_shader, "ro");
    int loc_uu = GetShaderLocation(gpu_shader, "uu");
    int loc_vv = GetShaderLocation(gpu_shader, "vv");
    int loc_ww = GetShaderLocation(gpu_shader, "ww");
    int loc_W = GetShaderLocation(gpu_shader, "W");
    int loc_H = GetShaderLocation(gpu_shader, "H");
    int loc_root_bmin = GetShaderLocation(gpu_shader, "root_bmin");
    int loc_root_bmax = GetShaderLocation(gpu_shader, "root_bmax");

    // Convert surfels to GPU format
    std::vector<GpuSurfel> g_surfels(master_points.size());
    for(size_t i=0; i<master_points.size(); ++i) {
        g_surfels[i].pos[0] = master_points[i].position.x; 
        g_surfels[i].pos[1] = master_points[i].position.y; 
        g_surfels[i].pos[2] = master_points[i].position.z;
        g_surfels[i].norm[0] = master_points[i].normal.x;  
        g_surfels[i].norm[1] = master_points[i].normal.y;  
        g_surfels[i].norm[2] = master_points[i].normal.z;
        g_surfels[i].radius = master_points[i].radius; 
        g_surfels[i].pad1 = 0.0f;
    }

    // Allocate & populate SSBOs
    unsigned int surfel_ssbo = rlLoadShaderBuffer(g_surfels.size() * sizeof(GpuSurfel), g_surfels.data(), RL_DYNAMIC_DRAW);
    unsigned int vals_ssbo   = rlLoadShaderBuffer(view.split_vals.size() * sizeof(float), (void*)view.split_vals.data(), RL_DYNAMIC_DRAW);
    unsigned int dims_ssbo   = rlLoadShaderBuffer(view.split_dims.size() * sizeof(uint64_t), (void*)view.split_dims.data(), RL_DYNAMIC_DRAW);
    unsigned int bks_ssbo    = rlLoadShaderBuffer(view.buckets.size() * sizeof(TreeType::LeafBucket), (void*)view.buckets.data(), RL_DYNAMIC_DRAW);
    unsigned int boxes_ssbo  = 0;
    if constexpr (TreeType::cfg.has_aabb) {
        boxes_ssbo = rlLoadShaderBuffer(view.node_boxes.size() * sizeof(typename TreeType::NodeBox), (void*)view.node_boxes.data(), RL_DYNAMIC_DRAW);
    }

    std::vector<Color> fb(W * H);
    Image img = { fb.data(), W, H, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };
    Texture2D tex = LoadTextureFromImage(img);

    bool use_gpu = false;
    double last_render_ms = 0.0;

    // Camera state: orbit around the terrain center.
    Vec3 cam_target = {0,0,0};
    float cam_yaw   = 0.7f;
    float cam_pitch = 1.0f;
    float cam_dist  = 1.9f;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_G)) use_gpu = !use_gpu;

        // --- camera controls ---
        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            Vector2 d = GetMouseDelta();
            cam_yaw   -= d.x * 0.008f;
            cam_pitch += d.y * 0.008f;
            cam_pitch = std::clamp(cam_pitch, -1.4f, 1.4f);
        }
        cam_dist *= std::exp(-GetMouseWheelMove() * 0.12f);
        cam_dist  = std::clamp(cam_dist, 0.35f, 25.0f);

        // Position on a sphere, world "up" = +Z (LiDAR elevation).
        Vec3 ro = cam_target + Vec3{ std::cos(cam_pitch) * std::sin(cam_yaw),
                                     std::cos(cam_pitch) * std::cos(cam_yaw),
                                     std::sin(cam_pitch) } * cam_dist;
        Vec3 ww = (cam_target - ro).normalize();              // toward target
        Vec3 uu = (Vec3{0,0,1}.cross(ww)).normalize();         // right
        Vec3 vv = ww.cross(uu);                                // up on screen

        // Pan the target (W/S = forward/back in depth, A/D = strafe, R/F = up/down).
        // Scale by GetFrameTime() so movement speed is FPS-independent.
        const float pan = 1.2f * cam_dist * GetFrameTime();
        Vec3 fwd_h = ww; fwd_h.z = 0.0f;                 // forward projected on the ground plane
        const float fl = fwd_h.length();
        if (fl > 1e-6f) fwd_h = fwd_h * (1.0f / fl);
        if (IsKeyDown(KEY_W)) cam_target = cam_target + fwd_h * pan;
        if (IsKeyDown(KEY_S)) cam_target = cam_target - fwd_h * pan;
        if (IsKeyDown(KEY_D)) cam_target = cam_target + uu * pan;
        if (IsKeyDown(KEY_A)) cam_target = cam_target - uu * pan;
        if (IsKeyDown(KEY_R)) cam_target = cam_target + Vec3{0,0,1} * pan;
        if (IsKeyDown(KEY_F)) cam_target = cam_target - Vec3{0,0,1} * pan;
        if (IsKeyPressed(KEY_HOME)) { cam_target = {0,0,0}; cam_yaw = 0.7f; cam_pitch = 1.0f; cam_dist = 1.9f; }

        if (use_gpu) {
            BeginDrawing();
            ClearBackground(BLACK);
            BeginShaderMode(gpu_shader);
            
            float ro_arr[3] = {ro.x, ro.y, ro.z}, uu_arr[3] = {uu.x, uu.y, uu.z}, vv_arr[3] = {vv.x, vv.y, vv.z}, ww_arr[3] = {ww.x, ww.y, ww.z};
            SetShaderValue(gpu_shader, loc_ro, ro_arr, SHADER_UNIFORM_VEC3);
            SetShaderValue(gpu_shader, loc_uu, uu_arr, SHADER_UNIFORM_VEC3);
            SetShaderValue(gpu_shader, loc_vv, vv_arr, SHADER_UNIFORM_VEC3);
            SetShaderValue(gpu_shader, loc_ww, ww_arr, SHADER_UNIFORM_VEC3);
            
            float fW = W, fH = H;
            SetShaderValue(gpu_shader, loc_W, &fW, SHADER_UNIFORM_FLOAT);
            SetShaderValue(gpu_shader, loc_H, &fH, SHADER_UNIFORM_FLOAT);

            float bmin_arr[3] = {root_aabb.min.x, root_aabb.min.y, root_aabb.min.z};
            float bmax_arr[3] = {root_aabb.max.x, root_aabb.max.y, root_aabb.max.z};
            SetShaderValue(gpu_shader, loc_root_bmin, bmin_arr, SHADER_UNIFORM_VEC3);
            SetShaderValue(gpu_shader, loc_root_bmax, bmax_arr, SHADER_UNIFORM_VEC3);

            rlBindShaderBuffer(surfel_ssbo, 0);
            rlBindShaderBuffer(vals_ssbo, 1);
            rlBindShaderBuffer(dims_ssbo, 2);
            rlBindShaderBuffer(bks_ssbo, 3);
            if constexpr (TreeType::cfg.has_aabb) rlBindShaderBuffer(boxes_ssbo, 4);
            
            DrawRectangle(0, 0, W, H, WHITE);
            EndShaderMode();

            last_render_ms = GetFrameTime() * 1000.0; 

        } else {
            double cpu_start = GetTime();

            #pragma omp parallel for schedule(dynamic, 4)
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float u = (2.0f * (x + 0.5f) - W) / H;
                    float v = (2.0f * ((H - y - 1) + 0.5f) - H) / H;
                    Vec3 rd = (uu*u + vv*v + ww).normalize();

                    float t = 0.0f; bool hit = false; Vec3 p, n;
                    for (int i = 0; i < 64; ++i) {
                        p = ro + rd * t;
                        
                        QueryResult res;
                        
                        // HUGE CPU OPTIMIZATION: Check root BBOX first
                        float root_dist_sq = root_aabb.dist_sq(p);
                        if (root_dist_sq > 0.001f) {
                            res = { std::sqrt(root_dist_sq), {0,1,0} };
                        } else {
                            TreeType::point_t target = {p.x, p.y, p.z};
                            auto knn_opt = view.query_1nn(target);
                            
                            if (knn_opt.has_value()) {
                                res = evaluate_surfel(p, knn_opt->dist_sq, knn_opt->payload_id, master_points);
                            } else {
                                res = {1000.0f, {0,1,0}};
                            }
                        }

                        if (res.dist < 0.005f) { hit = true; n = res.normal; break; }
                        if (t > 8.0f) break;
                        t += res.dist;
                    }

                    if (hit) {
                        Vec3 nv = n;
                        float diff = std::fmax(0.1f, nv.dot(Vec3{0.4f, 0.4f, 1.0f}.normalize()));
                        Vec3 c = elevation_color(p.z, root_aabb.min.z, root_aabb.max.z);
                        fb[y * W + x] = {(unsigned char)(c.x*diff*255), (unsigned char)(c.y*diff*255), (unsigned char)(c.z*diff*255), 255};
                    } else { fb[y * W + x] = {30, 30, 45, 255}; }
                }
            }
            last_render_ms = (GetTime() - cpu_start) * 1000.0;

            UpdateTexture(tex, fb.data());
            BeginDrawing();
            ClearBackground(BLACK);
            DrawTexture(tex, 0, 0, WHITE);
        }
        
        //The basic UI overlay.
        if (!shot_mode) {
            DrawRectangle(5, 5, 580, 130, Fade(BLACK, 0.8f));
            DrawText(TextFormat("ENGINE: [G] %s", use_gpu ? "GPU Compute (GLSL)" : "CPU Compute (OpenMP)"), 15, 15, 10, ORANGE);
            DrawText("ACTIVE: kd3::KdTree<32> (Autzen LiDAR, 5M pts)", 15, 32, 14, GREEN);
            DrawText("Drag = orbit  |  Wheel = zoom  |  WASD = pan  |  R/F = up/down  |  HOME = reset", 15, 50, 12, LIGHTGRAY);

            if (use_gpu) DrawText(TextFormat("GPU Frame Time: %.1f ms (%d FPS)", last_render_ms, GetFPS()), 15, 66, 14, YELLOW);
            else DrawText(TextFormat("CPU Render Time: %.1f ms (%d FPS)", last_render_ms, GetFPS()), 15, 66, 14, YELLOW);

            int num_points = (int)master_points.size();
            int num_splits = (int)view.split_vals.size();
            int num_leaves = (int)view.buckets.size();
            int total_nodes = num_splits + num_leaves;

            DrawText(TextFormat("Points: %d  |  Nodes: %d (%d splits, %d leaves)", num_points, total_nodes, num_splits, num_leaves), 15, 84, 12, WHITE);
            DrawText(TextFormat("Distance: %.2f  Elev: %.2f rad", cam_dist, cam_pitch), 15, 100, 12, WHITE);
        }

        EndDrawing();

        if (shot_mode) {
            TakeScreenshot("autzen_shot.png");
            break;
        }
    }

    CloseWindow(); 
    return 0;
}
