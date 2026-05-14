/**
 * @file render.cpp
 * @author karurochari (adapted for KD)
 * @brief A very condensed demo to show how kd3 can be used for direct raytracing.
 * @version 0.1
 * @date 2026-05-11
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <raylib.h>
#include <rlgl.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <string>

#include <kd3/kd3.hpp>

using TreeType = kd3::KdTree<kd3::limits<float>,{.LeafSize=32}>;

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

    operator TreeType::point_t(){return {x,y,z};}
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
// Twisted Box SDF (Used just to scatter our primitives)
// ---------------------------------------------------------
float sdf_box(Vec3 p, Vec3 b) {
    Vec3 d = {std::abs(p.x) - b.x, std::abs(p.y) - b.y, std::abs(p.z) - b.z};
    return std::sqrt(std::fmax(d.x, 0.f)*std::fmax(d.x, 0.f) + std::fmax(d.y, 0.f)*std::fmax(d.y, 0.f) + std::fmax(d.z, 0.f)*std::fmax(d.z, 0.f)) + std::fmin(std::fmax(d.x, std::fmax(d.y, d.z)), 0.f);
}
float sdf_twisted_box(Vec3 p) {
    float k = 2.0f; 
    float c = std::cos(k * p.y), s = std::sin(k * p.y);
    return sdf_box({c * p.x - s * p.z, p.y, s * p.x + c * p.z}, {0.8f, 1.2f, 0.8f});
}
Vec3 sdf_gradient(Vec3 p) {
    float e = 0.005f;
    return Vec3{sdf_twisted_box({p.x+e, p.y, p.z}) - sdf_twisted_box({p.x-e, p.y, p.z}),
                sdf_twisted_box({p.x, p.y+e, p.z}) - sdf_twisted_box({p.x, p.y-e, p.z}),
                sdf_twisted_box({p.x, p.y, p.z+e}) - sdf_twisted_box({p.x, p.y, p.z-e})}.normalize();
}

// ---------------------------------------------------------
// GPU Data Layout
// ---------------------------------------------------------
struct alignas(16) GpuSurfel {
    float pos[3]; float radius;
    float norm[3]; float pad1;
};

// ---------------------------------------------------------
// GLSL Fragment Shader for GPU Traversal
// ---------------------------------------------------------

constexpr char basics[] = {
#embed "../include/kd3/query.glsl"
, 0
};

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

// --- kd3 Config ---
#define KD3_LEAF_SIZE 32 
#define KD3_MAX_K 8
#define KD3_STACK_SIZE 128
#define KD3_INF2 1e29

#define KD3_BINDING_VALS 1
#define KD3_BINDING_DIMS 2
#define KD3_BINDING_BKS  3

struct Surfel {
    vec3 pos; float radius;
    vec3 norm; float pad;
};

layout(std430, binding = 0) readonly buffer SurfelBuffer { Surfel surfels[]; };

)" + basics +

R"(

float dist_sq_aabb(vec3 p, vec3 bmin, vec3 bmax) {
    float dx = max(0.0, max(bmin.x - p.x, p.x - bmax.x));
    float dy = max(0.0, max(bmin.y - p.y, p.y - bmax.y));
    float dz = max(0.0, max(bmin.z - p.z, p.z - bmax.z));
    return dx*dx + dy*dy + dz*dz;
}

void main() {
    float u = (2.0 * gl_FragCoord.x - W) / H;
    float v = (2.0 * gl_FragCoord.y - H) / H;
    vec3 rd = normalize(uu * u + vv * v + ww);

    Kd3RayHit res;

    //float root_dist = dist_sq_aabb(p, root_bmin, root_bmax);
    //if (root_dist > 0.001) res = Kd3RayHit(sqrt(root_dist), vec3(0,1,0));

    // Direct Raytracing against the BVH AABBs!
    if (kd3_query_ray(ro, rd, 1000.0, 0.01, res)) {
        vec3 n = surfels[res.payload_id].norm;
        float diff = max(0.1, dot(n, normalize(vec3(0.5, 1.0, -0.5))));
        vec3 c = vec3(0.3, 0.6, 0.9); // Blue theme for Direct Raytracing
        finalColor = vec4(c * diff, 1.0);
    } else {
        finalColor = vec4(30.0/255.0, 30.0/255.0, 45.0/255.0, 1.0);
    }
}
)";

// ---------------------------------------------------------
// Application
// ---------------------------------------------------------
int main() {
    const int W = 640, H = 480;
    InitWindow(W, H, "bvh3 Direct Raytracing demo");
    SetTargetFPS(0); // VSYNC off to benchmark "raw" render time

    std::cout << "Sampling SDF Surface...\n";
    std::vector<Surfel> master_points;
    AABB root_aabb; // Track global limits

    const float step = 0.01f;
    for(float x = -2.0f; x <= 2.0f; x += step) {
        for(float y = -2.0f; y <= 2.0f; y += step) {
            for(float z = -2.0f; z <= 2.0f; z += step) {
                if (std::abs(sdf_twisted_box({x,y,z})) < step * 0.8f) {
                    master_points.push_back({{x,y,z}, sdf_gradient({x,y,z}), step * 1.5f});
                    root_aabb.grow({x, y, z});
                }
            }
        }
    }
    std::cout << "Primitives generated: " << master_points.size() << "\n\n";

    // 1. Prepare points for bvh3 AABBs (bounding each surfel sphere)
    std::vector<TreeType::FatPoint> build_points(master_points.size());
    for (size_t i = 0; i < master_points.size(); ++i) {
        build_points[i].coords[0] = master_points[i].position.x;
        build_points[i].coords[1] = master_points[i].position.y;
        build_points[i].coords[2] = master_points[i].position.z;
        build_points[i].payload_id = i;
    }

    // 2. Build BvhTree
    auto t1 = std::chrono::high_resolution_clock::now();
    auto tree_expected = TreeType::build(build_points);
    if (!tree_expected) {
        std::cerr << "Failed to build tree! Empty input?\n";
        return 1;
    }
    TreeType kdtree = std::move(tree_expected.value());
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "BvhTree Built in: " 
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() 
              << " ms\n\n";
    kdtree.set_bbox(root_aabb.min, root_aabb.max);
    auto view = kdtree.view();

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

    std::vector<Color> fb(W * H);
    Image img = { fb.data(), W, H, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };
    Texture2D tex = LoadTextureFromImage(img);

    bool use_gpu = false;
    float time = 0.0f;
    double last_render_ms = 0.0;

    while (!WindowShouldClose()) {
        time += GetFrameTime();
        if (IsKeyPressed(KEY_G)) use_gpu = !use_gpu;

        Vec3 ro = { std::sin(time * 0.5f) * 3.5f, 2.0f, std::cos(time * 0.5f) * 3.5f };
        Vec3 ww = (Vec3{0,0,0} - ro).normalize();
        Vec3 uu = (Vec3{0,1,0}.cross(ww)).normalize();
        Vec3 vv = ww.cross(uu);

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

                    TreeType::point_t ro_arr = {ro.x, ro.y, ro.z};
                    TreeType::point_t rd_arr = {rd.x, rd.y, rd.z};

                    // Direct CPU Raycast traversal! No distance sphere-marching needed!
                    auto hit_opt = view.query_ray(ro_arr, rd_arr, 1000.0f, 0.01f);

                    if (hit_opt.has_value()) {
                        Vec3 n = master_points[hit_opt->payload_id].normal;
                        float diff = std::fmax(0.1f, n.dot(Vec3{0.5f, 1.0f, -0.5f}.normalize()));
                        Vec3 c = {0.3f, 0.6f, 0.9f}; // Blue theme matching shader
                        fb[y * W + x] = {(unsigned char)(c.x*diff*255), (unsigned char)(c.y*diff*255), (unsigned char)(c.z*diff*255), 255};
                    } else { 
                        fb[y * W + x] = {30, 30, 45, 255}; 
                    }
                }
            }
            last_render_ms = (GetTime() - cpu_start) * 1000.0;

            UpdateTexture(tex, fb.data());
            BeginDrawing();
            ClearBackground(BLACK);
            DrawTexture(tex, 0, 0, WHITE);
        }

        DrawRectangle(5, 5, 380, 110, Fade(BLACK, 0.8f));
        DrawText(TextFormat("ENGINE: [G] %s", use_gpu ? "GPU Compute (GLSL)" : "CPU Compute (OpenMP)"), 15, 15, 10, ORANGE);
        DrawText("ACTIVE: kd3::KdTree<32> (Direct Raytracing)", 15, 35, 16, SKYBLUE);

        if (use_gpu) DrawText(TextFormat("GPU Frame Time: %.1f ms (%d FPS)", last_render_ms, GetFPS()), 15, 60, 16, YELLOW);
        else DrawText(TextFormat("CPU Render Time: %.1f ms (%d FPS)", last_render_ms, GetFPS()), 15, 60, 16, YELLOW);

        int num_prims = (int)master_points.size();
        DrawText(TextFormat("Total Primitives: %d", num_prims), 15, 85, 16, WHITE);

        EndDrawing();
    }

    CloseWindow(); 
    return 0;
}
