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

// MUST match SIMD_PARALLELISM / LEAF_SIZE used in C++
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
// Replicates `alignas(std::min<size_t>(64, LEAF_SIZE*8))` in 32-bit words
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

struct Kd3RayHit {
    float t;
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
 * @brief Cast a ray!
 * @return true if found, false if tree is empty.
 */
bool kd3_query_ray(vec3 ro, vec3 rd, float max_t, float epsilon, out Kd3RayHit result) {
  uint num_buckets = uint(kd3_buckets.length()) / KD3_BUCKET_WORDS;
  if (num_buckets == 0u) return false;

  // OPTIMIZATION 1: Branchless Zero-Division Prevention (Identical to your BVH)
  vec3 inv_rd = 1.0 / mix(rd, vec3(1e-8), equal(rd, vec3(0.0)));
  vec3 abs_inv_rd = abs(inv_rd);

  // Define the thickness of the points (analogous to the BVH voxel width)
  vec3 voxel_extent = epsilon * abs_inv_rd;

  vec3 t1_aabb = (root_bmin - ro) * inv_rd;
  vec3 t2_aabb = (root_bmax - ro) * inv_rd;

  float tN = max(max(min(t1_aabb.x, t2_aabb.x), min(t1_aabb.y, t2_aabb.y)), max(min(t1_aabb.z, t2_aabb.z), 0.0));
  float tF = min(min(max(t1_aabb.x, t2_aabb.x), max(t1_aabb.y, t2_aabb.y)), min(max(t1_aabb.z, t2_aabb.z), max_t));

  // 2. Instantly cull any ray that misses the scene entirely (e.g., Skybox rays)
  if (tN > tF) return false;

  uint stack_idx[KD3_STACK_SIZE];
  float stack_t_min[KD3_STACK_SIZE];
  float stack_t_max[KD3_STACK_SIZE];
  uint stack_sz = 0u;

  // 3. FAST-FORWARD OPTIMIZATION: 
  // We initialize the traversal strictly within the AABB bounds!
  // The ray skips all empty space between the camera and the bounding box.
  stack_idx[0] = 0u; 
  stack_t_min[0] = tN; 
  stack_t_max[0] = tF; 
  stack_sz++;

  stack_idx[0] = 0u; stack_t_min[0] = 0.0; stack_t_max[0] = max_t; stack_sz++;

  float best_t = max_t;
  uint best_id = 0xFFFFFFFFu;
  uint LEAF_THRESHOLD = num_buckets - 1u;

  while (stack_sz > 0u) {
    stack_sz--;
    uint curr = stack_idx[stack_sz];
    float t_min = stack_t_min[stack_sz];
    float node_t_max = stack_t_max[stack_sz];

    if (t_min >= best_t) continue;

    if (curr >= LEAF_THRESHOLD) {
      uint base = (curr - LEAF_THRESHOLD) * KD3_BUCKET_WORDS;
      uint x_base = base;
      uint y_base = base + KD3_ARRAY_STRIDE;
      uint z_base = base + 2u * KD3_ARRAY_STRIDE;
      uint id_base = z_base + uint(KD3_LEAF_SIZE);

      for (uint i = 0u; i < uint(KD3_LEAF_SIZE); ++i) {
        vec3 b = vec3(
          uintBitsToFloat(kd3_buckets[x_base + i]),
          uintBitsToFloat(kd3_buckets[y_base + i]),
          uintBitsToFloat(kd3_buckets[z_base + i])
        );

        // OPTIMIZATION 3: Implicit Slab Test. 
        // This identically mimics the BVH AABB test, but because we know the voxel 
        // is perfectly centered on the point, we mathematically guarantee t1 <= t2. 
        // This lets us skip 6 min/max instructions per point compared to the BVH!
        vec3 C1 = (b - ro) * inv_rd;
        vec3 t1 = C1 - voxel_extent;
        vec3 t2 = C1 + voxel_extent;

        float tn = max(max(t1.x, t1.y), max(t1.z, 0.0));
        float tf = min(min(t2.x, t2.y), min(t2.z, best_t));

        if (tn <= tf && tn < best_t) {
          best_t = tn;
          best_id = kd3_buckets[id_base + i];
        }
      }
      continue;
    }

    uint dim = kd3_get_dim(curr);
    float split = kd3_split_vals[curr];

    float ro_dim = (dim == 0u) ? ro.x : ((dim == 1u) ? ro.y : ro.z);
    float rd_dim = (dim == 0u) ? rd.x : ((dim == 1u) ? rd.y : rd.z);
    float inv_rd_dim = (dim == 0u) ? inv_rd.x : ((dim == 1u) ? inv_rd.y : inv_rd.z);

    uint left_child = 2u * curr + 1u;
    uint right_child = 2u * curr + 2u;

    uint first = (rd_dim >= 0.0) ? left_child : right_child;
    uint second = (rd_dim >= 0.0) ? right_child : left_child;

    float t_split = (split - ro_dim) * inv_rd_dim;
    float t_margin = epsilon * abs(inv_rd_dim);

    float t_enter_second = t_split - t_margin;
    float t_leave_first  = t_split + t_margin;

    bool visit_first  = t_min <= t_leave_first;
    bool visit_second = node_t_max >= t_enter_second;

    // OPTIMIZATION 4: Flattened branching logic. 
    if (visit_first && visit_second) {
      stack_idx[stack_sz] = second; stack_t_min[stack_sz] = max(t_min, t_enter_second); stack_t_max[stack_sz] = node_t_max; stack_sz++;
      stack_idx[stack_sz] = first;  stack_t_min[stack_sz] = t_min; stack_t_max[stack_sz] = min(node_t_max, t_leave_first); stack_sz++;
    } else {
      stack_idx[stack_sz] = visit_first ? first : second;
      stack_t_min[stack_sz] = t_min;
      stack_t_max[stack_sz] = node_t_max;
      stack_sz++;
    }
  }

  if (best_id != 0xFFFFFFFFu) {
    result.t = best_t;
    result.payload_id = best_id;
    return true;
  }

  return false;
}

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
