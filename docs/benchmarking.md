> How to benchmark it against nanoflann?

```sh
xmake f --with_demo=true
xmake run benchmarks.nanoflann
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

Regardless of the specifics for my test-bench, on all machines I tried this one, more or less regardless of parameters, I always measured `kd3` has ~2.2 times the throughput of `nanoflann`.  
Furthermore, our tree construction is parallelized; still, even disabling OpenMP support, `kd3` would be faster.  

Clearly `nanoflann` is more flexibile in several ways; but if you don't need that flexibility, `kd3` provides a meaningful improvement.  

> How to optimize for speed?

- use smaller data types, for example fixed point integers;
- tweak the leaf size based on available registers and tree size; use the [plot](../examples/plot.cpp) to characterize you system[^1];
- use the force inlined versions of all functions if you are worried about your compiler not doing it by itself; it can boost performance up to 10% in our tests;
- if distance is all you need, set the flag `has_index` to false; most functions will no longer work, but you will gain a 25% boost in throughput according to my testing;


[^1]: Benchmark you system if you seek maximum powaaa! The presence of AVX2 vs AVX512 and 3D-cache do have a major impact in determining the best configuration as visible from the [sample results](../assets/results/).