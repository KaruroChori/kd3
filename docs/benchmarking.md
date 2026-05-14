> How to benchmark it against nanoflann?

```sh
xmake f --with_demo=true
xmake run nanoflann
```

Example output on my 5950x:

```
--- KD-Tree Benchmark --- [simd: 8, parallelism: 32]
Generating 5000000 random points...

[ BUILD PHASE ]
kd3 Build Time:       152.139 ms
nanoflann Build Time: 892.225 ms

[ QUERY PHASE ] - 100000 queries
kd3 Query Time:       1059.85 ms (94352.9 QPS)
nanoflann Query Time: 2318.7 ms (43127.6 QPS)

[ VALIDATION PHASE ]
Validating correctness against linear scan...
Linear Scan Time (Subset):       416.774 ms
kd3 matches brute force:         YES
nanoflann matches brute force:   YES
```

On all my tested machines, more or less regardless of parameters, I measured a x2 performance boost for `kd3`.  
Also, our build construction is parallelized, but even disabling `openmp` it is still faster compared to the one offered by nanoflann.  

Clearly `nanoflann` is more flexibile in several ways, but if you don't need that flexibility, `kd3` is a meaningful improvement.