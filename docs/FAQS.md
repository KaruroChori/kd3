> What's the difference between `kd3` and `kd3-fast`?
 
Same data structure at its core, but the index to the original point has been removed. Operations on it are also a bit different, as the general kNN makes no sense.  
For many tasks (like when rendering SDFs), the distance of the closest point is totally sufficient if we don't need extra data like normals or material ID to be baked in there.  
It is always possible to perform the very last pass on a different data structure or even with the source SDF, so if no baking is needed the index can be ignored.  
Doing so leads to a great boost in performance!

> What about BVH tree / Octrees?

I am working on similar implementations for the other tree types.  
They will be shipped as separate libraries, but same basics and similar interfaces.
