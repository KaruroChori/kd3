# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- A first attempt at writing some test via doctest, LLM-generated as I honestly cannot be bothered.

### Changed

- Query path rewritten for SIMD: 1-NN and k-NN throughput improved by roughly
  17–50% depending on the workload and tree configuration. The payload-free
  `fast` variants are unaffected.
- The tree builder no longer zero-initializes its storage, which gives a
  measurable build-time speed-up for bigger trees.
- Ray queries are now vectorized when SIMD is available.

### Removed

- The preordered/`build_from_ordered` builder. It was a leftover from an earlier dead-end design. Technically it is a breaking feature.

## [1.3.1] - 2026-09-23

### Added

- Header-only install target for easier distribution.
- `render.autzen` scene demo.

### Changed

- CPU tuning (`-march=native`, `-ffast-math`) is now opt-in via the `native`
  build option instead of being forced on in release builds.

- Fixed a bug in the benchmarks which made nanoflann appear slighly worse compared to their real baseline.

## [1.3.0] - 2026-08-26

### Added

- Optional per-subtree bounding boxes (`cfg_t::has_aabb`), which restore
  competitive pruning for queries far from the stored points.
- Plots and support for the payload-free `fast` query variants.
- Rewritten comparative benchmark suite against nanoflann, covering real-world
  point clouds (autzen, bunny, cities, dragon) and a range of synthetic
  distributions; results are emitted as CSV and Plotly reports.

### Changed

- Query performance is now much more uniform across spatial distributions.
- OpenMP is optional: the library can be built without a hard dependency on it.

### Fixed

- Ray query edge cases; the CPU path now agrees with the GLSL implementation.
