> How do I run the benchmarks?

```sh
xmake f --with_evaluation=true
xmake run fetch-datasets       # optional: real-world clouds into ./datasets
xmake run benchmarks.comparative  # synthetics + ./datasets, stock and has_aabb trees
```

Every cloud measures build time and k-NN throughput under two access patterns, validated against brute force:

- `bbox-volume`: queries uniform inside the cloud bounding box (raymarch-like);
- `on-cloud`: queries sampled from the cloud itself with ~0.2% jitter (kNN-graph-like).

Both tree configurations are compared per cloud - stock median splits vs `cfg_t::has_aabb` subtree boxes - and rows stream into `kd3_datasets_report.csv` next to the built binaries. With `--with_demo=true`, nanoflann joins as a fixed reference column. LAS/LAZ loading requires LASlib/LASzip from `--with_evaluation`; without it only `.bin`, `.xyz` and `.ply` files load.

`benchmarks.fast` separately micro-benchmarks the payload-free tree (`has_payload = NONE`, distance-only queries) across scalar types.

> When should cfg_t::has_aabb be enabled?

Queries far from any stored point prune poorly against plain median splits: cells covering sparse regions span huge volumes. `has_aabb` stores one tight bounding box per subtree (+`48/leaf_size` bytes per point, about +9% at the default leaf size) and restores competitive pruning there.

Typical k=10 throughput on the `bbox-volume` distribution (5950X, single thread, 50K queries):

| cloud | stock | has_aabb | nanoflann |
|---|---|---|---|
| autzen LiDAR scan | 53K q/s | **252K q/s** | 119K q/s |
| dragon scan | 87K q/s | **285K q/s** | 167K q/s |
| sphere-shell surface | 17K q/s | **112K q/s** | 44K q/s |
| anisotropic ellipsoid | 2.3K q/s | **107K q/s** | 106K q/s |
| uniform solid cube | **763K q/s** | 705K q/s | 333K q/s |
| power-law halo | 426K q/s | 598K q/s | **1,215K q/s** |

Near-data access (`on-cloud`) favors kd3 by roughly 2-3x over nanoflann in either configuration; the stock tree suffices there. A strongly halo-skewed power-law density remains nanoflann's best case even with boxes enabled.

Guidance:

- dense volumetric data or near-data access → stock tree;
- scans, shells, or query patterns spread through empty space → `has_aabb`;
- memory-tight deployments on small leaves: box overhead scales as `48/leaf_size` bytes per point (+37% at leaf 8, +9% at leaf 32, +2% at leaf 128).

> How to optimize for speed?

- use smaller data types, for example fixed point integers;
- tweak the leaf size based on available registers and tree size; use the [plot](../examples/plot.cpp) to characterize you system[^1];
- use the force inlined versions of all functions if you are worried about your compiler not doing it by itself; it can boost performance up to 10% in our tests;
- if distance is all you need, set the flag `has_index` to false; most functions will no longer work, but you will gain a 25% boost in throughput according to my testing;


[^1]: Benchmark you system if you seek maximum powaaa! The presence of AVX2 vs AVX512 and 3D-cache do have a major impact in determining the best configuration as visible from the [sample results](../assets/results/).
