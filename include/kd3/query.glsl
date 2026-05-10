/**
 * @file query.glsl
 * @author karurochari
 * @brief GLSL shader implementing the same logic as the baseline C++ code.
 * @date 2026-05-06
 * 
 * @copyright Copyright (c) 2026
 * 
 */

// Warning: it requires at least GLSL 4.30 for SSBO support
//          it requires 64bit extension GL_ARB_gpu_shader_int64

// ---------------------------------------------------------
// CONFIGURATION MACROS
// Override these by defining them before including this code
// ---------------------------------------------------------

// MUST match SIMD_PARALLELISM / LeafSize used in C++
#ifndef KD3_LEAF_SIZE
#define KD3_LEAF_SIZE 16 
#endif

// The maximum number of elements for KNN queries. This mess is slooow on GPU
#ifndef KD3_MAX_K
#define KD3_MAX_K 8
#endif

// 48 is more than enough to traverse a tree
#ifndef KD3_STACK_SIZE
#define KD3_STACK_SIZE 48
#endif

#define KD3_INF2 1e29

// --- Memory Layout Constants ---
// Replicates `alignas(std::min<size_t>(64, LeafSize*8))` in 32-bit words
#define KD3_ALIGN_WORDS (min(16u, uint(KD3_LEAF_SIZE) * 2u))

// Distance in words between x, y, and z arrays (rounded up to nearest alignment)
#define KD3_ARRAY_STRIDE (((uint(KD3_LEAF_SIZE) + KD3_ALIGN_WORDS - 1u) / KD3_ALIGN_WORDS) * KD3_ALIGN_WORDS)

// Total words per bucket (x, y, z, and ids, with final padding to alignment boundary)
#define KD3_BUCKET_WORDS (((2u * KD3_ARRAY_STRIDE + 2u * uint(KD3_LEAF_SIZE) + KD3_ALIGN_WORDS - 1u) / KD3_ALIGN_WORDS) * KD3_ALIGN_WORDS)

// ---------------------------------------------------------
// BUFFERS
// ---------------------------------------------------------
#ifndef KD3_BINDING_VALS
#define KD3_BINDING_VALS 0
#endif
#ifndef KD3_BINDING_DIMS
#define KD3_BINDING_DIMS 1
#endif
#ifndef KD3_BINDING_BKS
#define KD3_BINDING_BKS  2
#endif

layout(std430, binding = KD3_BINDING_VALS) readonly buffer KdTreeSplitVals { float kd3_split_vals[]; };
layout(std430, binding = KD3_BINDING_DIMS) readonly buffer KdTreeSplitDims { uint64_t kd3_split_dims[]; };
layout(std430, binding = KD3_BINDING_BKS)  readonly buffer KdTreeBuckets   { uint kd3_buckets[]; };

// ---------------------------------------------------------
// DATA STRUCTURES
// ---------------------------------------------------------
struct Kd3KnnResult {
    float dist_sq;
    uint payload_id;
};

// ---------------------------------------------------------
// INTERNAL HELPERS
// ---------------------------------------------------------
uint kd3_get_dim(uint i) {
    uint idx = i / 32u;               // 16 dimensions fit inside a 32-bit word
    uint offset = (i % 32u) * 2u;
    return uint(kd3_split_dims[idx] >> offset) & 3u;
}

// ---------------------------------------------------------
// API FUNCTIONS
// ---------------------------------------------------------

/**
 * @brief Find the single nearest neighbor.
 * @return true if found, false if tree is empty.
 */
bool kd3_query_1nn(vec3 target, out Kd3KnnResult result) {
    uint num_buckets = uint(kd3_buckets.length()) / KD3_BUCKET_WORDS;
    if (num_buckets == 0u) return false;

    float min_dist_sq = 1e38; // Max float
    uint best_id = 0u;

    uint stack_idx[KD3_STACK_SIZE];
    float stack_dist[KD3_STACK_SIZE];
    uint stack_sz = 0u;

    stack_idx[stack_sz] = 0u;
    stack_dist[stack_sz] = 0.0;
    stack_sz++;

    uint LEAF_THRESHOLD = num_buckets - 1u;

    while (stack_sz > 0u) {
        stack_sz--;
        uint curr = stack_idx[stack_sz];
        float node_min_dist_sq = stack_dist[stack_sz];

        if (node_min_dist_sq >= min_dist_sq) continue;

        if (curr >= LEAF_THRESHOLD) {
            uint bucket_idx = curr - LEAF_THRESHOLD;
            uint base = bucket_idx * KD3_BUCKET_WORDS;
            
            uint x_base = base;
            uint y_base = base + KD3_ARRAY_STRIDE;
            uint z_base = base + 2u * KD3_ARRAY_STRIDE;
            uint id_base = z_base + uint(KD3_LEAF_SIZE);

            for (uint i = 0u; i < uint(KD3_LEAF_SIZE); ++i) {
                float bx = uintBitsToFloat(kd3_buckets[x_base + i]);
                float by = uintBitsToFloat(kd3_buckets[y_base + i]);
                float bz = uintBitsToFloat(kd3_buckets[z_base + i]);
                
                float dx = target.x - bx;
                float dy = target.y - by;
                float dz = target.z - bz;
                float dist_sq = dx*dx + dy*dy + dz*dz;

                if (dist_sq < min_dist_sq) {
                    min_dist_sq = dist_sq;
                    best_id = kd3_buckets[id_base + i];
                }
            }
            continue;
        }

        uint dim = kd3_get_dim(curr);
        float split = kd3_split_vals[curr];

        // GLSL vectors nicely support array-style indexing
        float axis_dist = target[dim] - split;
        float axis_dist_sq = axis_dist * axis_dist;

        uint left = 2u * curr + 1u;
        uint right = 2u * curr + 2u;

        uint first = (axis_dist < 0.0) ? left : right;
        uint second = (axis_dist < 0.0) ? right : left;

        if (axis_dist_sq < min_dist_sq) {
            stack_idx[stack_sz] = second;
            stack_dist[stack_sz] = axis_dist_sq;
            stack_sz++;
        }
        stack_idx[stack_sz] = first;
        stack_dist[stack_sz] = 0.0;
        stack_sz++;
    }

    if (min_dist_sq >= float(KD3_INF2)) return false;

    result.dist_sq = min_dist_sq;
    result.payload_id = best_id;
    return true;
}

/**
 * @brief Find the K nearest neighbors. 
 * @return The number of valid valid matches appended into `results` array. Sorted closest to furthest.
 */
uint kd3_query_knn(vec3 target, uint k, out Kd3KnnResult results[KD3_MAX_K]) {
    uint num_buckets = uint(kd3_buckets.length()) / KD3_BUCKET_WORDS;
    if (num_buckets == 0u || k == 0u) return 0u;

    uint count = 0u;
    uint stack_idx[KD3_STACK_SIZE]; float stack_dist[KD3_STACK_SIZE]; uint stack_sz = 0u;

    stack_idx[stack_sz] = 0u; stack_dist[stack_sz] = 0.0; stack_sz++;
    uint LEAF_THRESHOLD = num_buckets - 1u;

    while (stack_sz > 0u) {
        stack_sz--;
        uint curr = stack_idx[stack_sz]; 
        float node_min_dist = stack_dist[stack_sz];

        // Prune if the node is further than our Kth best point
        if (count == k && node_min_dist >= results[k - 1u].dist_sq) continue;

        if (curr >= LEAF_THRESHOLD) {
            uint base = (curr - LEAF_THRESHOLD) * KD3_BUCKET_WORDS;
            uint x_base = base;
            uint y_base = base + KD3_ARRAY_STRIDE;
            uint z_base = base + 2u * KD3_ARRAY_STRIDE;
            uint id_base = z_base + uint(KD3_LEAF_SIZE);

            for (uint i = 0u; i < uint(KD3_LEAF_SIZE); ++i) {
                float bx = uintBitsToFloat(kd3_buckets[x_base + i]);
                float by = uintBitsToFloat(kd3_buckets[y_base + i]);
                float bz = uintBitsToFloat(kd3_buckets[z_base + i]);
                
                float dx = target.x - bx; float dy = target.y - by; float dz = target.z - bz;
                float dist_sq = dx*dx + dy*dy + dz*dz;

                // Inline, register-friendly insertion sort!
                if (count < k || dist_sq < results[k - 1u].dist_sq) {
                    uint pos = count;
                    if (pos == k) pos = k - 1u;
                    else count++;
                    
                    // Shift worse elements to the right
                    while (pos > 0u && results[pos - 1u].dist_sq > dist_sq) {
                        results[pos] = results[pos - 1u];
                        pos--;
                    }
                    results[pos] = Kd3KnnResult(dist_sq, kd3_buckets[id_base + i]);
                }
            }
            continue;
        }

        uint dim = kd3_get_dim(curr);
        float axis_dist = target[dim] - kd3_split_vals[curr];
        float axis_dist_sq = axis_dist * axis_dist;
        uint first = (axis_dist < 0.0) ? (2u * curr + 1u) : (2u * curr + 2u);
        uint second = (axis_dist < 0.0) ? (2u * curr + 2u) : (2u * curr + 1u);

        if (count < k || axis_dist_sq < results[k - 1u].dist_sq) {
            stack_idx[stack_sz] = second; stack_dist[stack_sz] = axis_dist_sq; stack_sz++;
        }
        stack_idx[stack_sz] = first; stack_dist[stack_sz] = 0.0; stack_sz++;
    }

    // Filter padded elements
    uint valid_results = 0u;
    for (uint i = 0u; i < count; ++i) if (results[i].dist_sq < float(KD3_INF2)) valid_results++;
    return valid_results;
}
