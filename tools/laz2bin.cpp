// Convert a LAS/LAZ file to a raw float32 XYZ .bin (12 bytes/point), capped at MAX.
#include <LASlib/lasreader.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <input.laz> <output.bin> [max_points]\n", argv[0]);
        return 1;
    }
    const size_t MAX = argc > 3 ? strtoull(argv[3], nullptr, 10) : 5000000;

    LASreadOpener opener;
    opener.set_file_name(argv[1]);
    LASreader* reader = opener.open();
    if (!reader) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    FILE* out = std::fopen(argv[2], "wb");
    if (!out) { std::fprintf(stderr, "cannot write %s\n", argv[2]); return 1; }

    // deterministic stride subsample so we keep <= MAX points evenly spread
    const long long total = static_cast<long long>(reader->header.number_of_point_records);
    const double stride = total > 0 ? static_cast<double>(total) / static_cast<double>(MAX) : 1.0;
    size_t kept = 0;
    for (long long i = 0; i < total && kept < MAX; ++i) {
        reader->read_point();
        const long long want = static_cast<long long>(static_cast<double>(kept) * stride);
        if (i < want) continue;
        const LASpoint& p = reader->point;
        float f[3] = { static_cast<float>(p.get_x()),
                       static_cast<float>(p.get_y()),
                       static_cast<float>(p.get_z()) };
        std::fwrite(f, sizeof(float), 3, out);
        ++kept;
    }
    reader->close();
    delete reader;
    std::fclose(out);
    std::printf("wrote %zu points -> %s\n", kept, argv[2]);
    return 0;
}
