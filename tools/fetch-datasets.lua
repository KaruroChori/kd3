--! @file fetch-datasets.lua
--! Downloads and prepares real-world point-cloud datasets into ./datasets
--! for use by the benchmarks.comparative target. Invoked via:
--!
--!     xmake run fetch-datasets
--!
--! Files already present are skipped, so re-running is cheap and offline-safe.
--! All sources are freely redistributable for research/benchmarking purposes:
--!   - bunny / dragon : Stanford 3D Scanning Repository
--!   - cities         : GeoNames (CC-BY 4.0), converted from TSV to lon-lat-elev XYZ
--!   - autzen         : PDAL sample LiDAR scan (LASzip-compressed)
--!
--! All logic lives inside on_run: xmake sandboxes each script scope differently,
--! and only there are the os/net/archive APIs guaranteed to be bound correctly.
--! Deliberately avoids post-5.1 syntax (goto, integer division, ...) for
--! compatibility with older embedded Lua runtimes.

local datasets_root = path.join(os.projectdir(), "datasets")
local tmp_root = path.join(datasets_root, ".tmp")

-- Ordered list so repeated runs produce stable output.
local manifest = {
    {
        name = "bunny",
        urls = {"https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"},
        archive = true,
        keep = "bun_zipper.ply",
        out = "bunny.ply"
    },
    {
        name = "dragon",
        urls = {"https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz"},
        archive = true,
        keep = "dragon_vrip.ply",
        out = "dragon.ply"
    },
    {
        name = "cities",
        urls = {"https://download.geonames.org/export/dump/cities500.zip"},
        archive = true,
        keep = "cities500.txt",
        out = "cities.xyz",
        geonames = true
    },
    {
        name = "autzen",
        urls = {"https://media.githubusercontent.com/media/PDAL/data/main/autzen/autzen.laz"},
        out = "autzen.laz"
    }
}

target("fetch-datasets")
    set_kind("phony")
    set_default(false)
    on_run(function()
        local http = import("net.http", {anonymous = true})
        local archive = import("utils.archive", {anonymous = true})

        -- Converts a GeoNames TSV dump (tab separated, lat=field5, lon=field6,
        -- elev=field16) into a plain "lon lat elev" XYZ file loadable by kdbench.
        local function geonames_to_xyz(src, dst)
            local fin = io.open(src, "r")
            if not fin then
                raise("cannot read %s", src)
            end
            local fout = io.open(dst, "w")
            local count = 0
            for line in fin:lines() do
                local fields = {}
                line:gsub("([^\t]*)\t", function(c)
                    fields[#fields + 1] = c
                end)
                local lat = tonumber(fields[5])
                local lon = tonumber(fields[6])
                if lat and lon then
                    local elev = tonumber(fields[16]) or 0.0
                    fout:write(string.format("%f %f %f\n", lon, lat, elev))
                    count = count + 1
                end
            end
            fout:close()
            fin:close()
            print(format("  -> converted %d GeoNames records to XYZ", count))
        end

        local function locate(extractdir, filename)
            local found = {}
            for _, p in ipairs(os.filedirs(path.join(extractdir, "**"))) do
                if path.filename(p) == filename then
                    table.insert(found, p)
                end
            end
            return #found > 0 and found[1] or nil
        end

        -- Returns the path of a successfully downloaded file, or nil.
        local function download(urls)
            for _, url in ipairs(urls) do
                -- Keep the source basename so utils.archive can infer the format
                -- from a proper .tar.gz / .zip / .laz extension.
                local candidate = path.join(tmp_root, string.match(url, "[^/]+$"))
                os.rm(candidate)
                try {
                    function()
                        http.download(url, candidate)
                    end
                }
                if os.isfile(candidate) and os.filesize(candidate) > 4096 then
                    -- Files below 4KB are treated as failures: they are typically
                    -- Git-LFS pointers or HTML error pages rather than real data.
                    return candidate
                end
                os.rm(candidate)
            end
            return nil
        end        -- Returns true on success or benign skip; false marks the dataset as failed.
        local function fetch_one(spec)
            local dst = path.join(datasets_root, spec.out)

            if os.isfile(dst) then
                print(format("[skip] %-8s (%s already present)", spec.name, spec.out))
                return true
            end

            print(format("[fetch] %s ...", spec.name))
            os.mkdir(tmp_root)

            local raw = download(spec.urls)
            local ok = false
            if raw then
                ok = true
            else
                print(format("[warn] could not download %s; skipping (network issue?)", spec.name))
            end

            if ok and spec.archive then
                local extractdir = path.join(tmp_root, "extracted")
                os.rm(extractdir)
                archive.extract(raw, extractdir)
                local inner = locate(extractdir, spec.keep)
                if not inner then
                    print(format("[warn] %s missing expected member %s", spec.name, spec.keep))
                    ok = false
                elseif spec.geonames then
                    geonames_to_xyz(inner, dst)
                else
                    os.cp(inner, dst)
                end
                os.rm(extractdir)
            elseif ok then
                os.mv(raw, dst)
            end

            if ok then
                print(format("[ok]   %-8s -> %s", spec.name, dst))
            end
            return ok
        end

        os.mkdir(datasets_root)

        local failures = {}
        for _, spec in ipairs(manifest) do
            if not fetch_one(spec) then
                table.insert(failures, spec.name)
            end
        end

        os.rm(tmp_root)

        if #failures > 0 then
            print(format("\nDone with %d failure(s): %s", #failures, table.concat(failures, ", ")))
            print("Re-run `xmake run fetch-datasets` to retry the missing ones.")
        else
            print("\nAll datasets ready under " .. datasets_root)
            print("Now run: xmake run benchmarks.comparative")
        end
    end)
target_end()
