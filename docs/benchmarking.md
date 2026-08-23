> How do I run the benchmarks? Can I trust your claim?

Never, always check, and ideally bring your own benchmarks.

```sh
xmake f --with_evaluation=true      # To guard off heavy dependencies
xmake run fetch-datasets            # optional if you want to test against REAL datasets
xmake run benchmarks.comparative    # single pass for 1-NN and 10-NN, on a matrix of options
```

Every batch measures build time and k-NN throughput under two query patterns, validated against brute force:

- `bbox-volume`: queries uniform inside the cloud bounding box;
- `on-cloud`: queries sampled from the cloud itself with ~0.2% jitter, like very close to the original points.

Four configurations are compared per cloud, spanning {full-query, fast-query} x {basic split, bboxes}.  
Rows stream into `kd3_datasets_report.csv` and a Plotly comparison graph (`kd3_comparative_report.html`) is written next to the built binaries.

You can check those I got on one of my workstations [here](../assets/results/5950X).

> When should cfg_t::has_aabb be enabled?

Queries far from any stored point prune poorly against plain median splits: cells covering sparse regions span huge volumes. `has_aabb` stores one tight bounding box per subtree and restores competitive pruning there at the cost of some extra space.

Guidance:

- dense volumetric data or near-data access → `has_aabb: false`;
- same if you have very little memory and you care less about speed.
- scans, shells, or query patterns spread through empty space → `has_aabb: true`;

> How to optimize for speed?

- use smaller data types, for example fixed point integers;
- tweak the leaf size based on available registers and tree size; use the [sweep](../benchmarks/sweep.cpp) to characterize your system[^1];
- use the force inlined versions of all functions if you are worried about your compiler not doing it by itself; it can boost performance up to 10% in our tests;
- if distance for the 1-NN is all you need, set the flag `has_index` to false; most functions will no longer work, but there is a meaningful improvement for the remaining specialized ones.


[^1]: Benchmark you system if you seek maximum powaaa! The presence of AVX2 vs AVX512 and 3D-cache do have a major impact in determining the best configuration as visible from the [sample results](../assets/results/).
