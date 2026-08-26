/**
 * @file benchmarks.comparative.cpp
 * @brief Dataset-driven comparative benchmarking of kd3 configurations against
 *        structured synthetic distributions and real-world point clouds
 *        (see tools/fetch-datasets.lua).
 *
 * Two test kinds run over every cloud, each spanning the configuration matrix
 * where the node type is semantically valid:
 *
 *   knn (k = --knn, default 10, payload-bearing trees only):
 *       base : fat nodes, median splits
 *       aabb : fat nodes + per-subtree bounding boxes (cfg_t::has_aabb)
 *
 *   nn1 (exact 1-NN, minimum squared distance - the only query the
 *        lightweight nodes support):
 *       base / aabb            : through query_1nn
 *       fast / fast-aabb       : payload-free nodes (cfg_t::has_payload = NONE)
 *                                through query_distance2_inline
 *
 * Query distributions for both kinds:
 *   - bbox-volume : queries uniform inside the cloud bounding box (raymarch-like access)
 *   - on-cloud    : queries sampled from the cloud itself, small jitter (kNN-graph-like access)
 *
 * All rows are validated against brute force. nanoflann (when compiled in via
 * --with_demo) is measured once per cloud/distribution at matching k and
 * stamped as a fixed reference. Results stream into kd3_datasets_report.csv
 * and a Plotly comparison graph next to the binaries.
 */

#include "datasets.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#ifdef KD3_BENCH_NANOFLANN
#include <nanoflann.hpp>
#endif

using TreeType       = kd3::KdTree<kd3::limits<float>, {.leaf_size = 32}>;
using TreeTypeAabb   = kd3::KdTree<kd3::limits<float>, {.leaf_size = 32, .has_aabb = true}>;
using TreeTypeFast   = kd3::KdTree<kd3::limits<float>,
                                   {.leaf_size = 32,
                                    .has_payload = kd3::cfg_t::has_payload_t::NONE}>;
using TreeTypeFastAabb = kd3::KdTree<kd3::limits<float>,
                                     {.leaf_size = 32,
                                      .has_payload = kd3::cfg_t::has_payload_t::NONE,
                                      .has_aabb = true}>;

namespace {

struct Options {
    size_t n_queries = 100'000;
    size_t k         = 10;
    std::string datasets_dir_arg;
};

Options parse_options(int argc, char** argv) {
    Options opts;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> const char* {
            const size_t eq = arg.find('=');
            return eq == std::string::npos ? "" : arg.c_str() + eq + 1;
        };
        if (arg.rfind("--queries=", 0) == 0) opts.n_queries = std::strtoull(value(), nullptr, 10);
        else if (arg.rfind("--knn=", 0) == 0) opts.k = std::strtoull(value(), nullptr, 10);
        else if (arg.rfind("--dir=", 0) == 0) opts.datasets_dir_arg = value();
    }
    if (opts.n_queries == 0 || opts.k == 0 || opts.k > 128) {
        std::cerr << "Invalid options: queries and knn must be > 0, knn <= 128\n";
        std::exit(1);
    }
    return opts;
}

/// The datasets folder may not be the process working directory when launched
/// through `xmake run`, so resolution order is:
/// explicit --dir -> $KD3_DATASETS_DIR -> ./datasets -> ancestors of the executable.
std::filesystem::path resolve_datasets_dir(const Options& opts, const char* argv0) {
    if (!opts.datasets_dir_arg.empty()) return opts.datasets_dir_arg;
    if (const char* env = std::getenv("KD3_DATASETS_DIR"); env && *env) return env;
    std::error_code ec;
    if (std::filesystem::exists("datasets", ec)) return "datasets";
    if (argv0 && *argv0) {
        if (auto exe = std::filesystem::canonical(argv0, ec); !ec) {
            auto dir = exe.parent_path();
            for (int depth = 0; depth < 6 && !dir.empty(); ++depth) {
                const auto candidate = dir / "datasets";
                if (std::filesystem::exists(candidate, ec)) return candidate;
                dir = dir.parent_path();
            }
        }
    }
    return "datasets";
}

using LimitsF = kd3::limits<float>;

std::vector<LimitsF::KnnResult> linear_scan(const float* target, size_t k,
                                            std::span<const LimitsF::FatPoint> points) {
    std::vector<LimitsF::KnnResult> heap;
    heap.reserve(k + 1);
    for (const auto& p : points) {
        const float dx = target[0] - p.coords[0];
        const float dy = target[1] - p.coords[1];
        const float dz = target[2] - p.coords[2];
        const float dist_sq = dx * dx + dy * dy + dz * dz;
        if (heap.size() < k) {
            heap.push_back({dist_sq, p.payload_id});
            std::push_heap(heap.begin(), heap.end());
        } else if (dist_sq < heap.front().dist_sq) {
            std::pop_heap(heap.begin(), heap.end());
            heap.back() = {dist_sq, p.payload_id};
            std::push_heap(heap.begin(), heap.end());
        }
    }
    std::sort(heap.begin(), heap.end());
    return heap;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------
// Result collection & formatting
// ---------------------------------------------------------

struct Row {
    std::string dataset, source, distribution, config;
    /// "knn" (k nearest, payload-bearing trees) or "nn1" (exact 1-NN).
    const char* test = "knn";
    size_t n_points = 0, n_queries = 0, k = 0;
    double build_s = 0.0, kd3_s = 0.0, kd3_qps = 0.0;
    double nf_build_s = -1.0, nf_s = -1.0, nf_qps = -1.0;
    /// -1 = not applicable, 0 = failed, 1 = passed.
    int kd3_ok = -1, nf_ok = -1;
};

std::string thousands(uint64_t v) {
    std::string s = std::to_string(v);
    int insert = static_cast<int>(s.length()) - 3;
    while (insert > 0) {
        s.insert(static_cast<size_t>(insert), ",");
        insert -= 3;
    }
    return s;
}

template <typename T>
std::string fixed(T value, int precision) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

const char* ok_label(int v) { return v < 0 ? "n/a" : (v ? "YES" : "NO"); }

void write_csv_header(std::ofstream& csv) {
    csv << "dataset,source,distribution,test,config,n_points,n_queries,k,"
           "build_s,kd3_s,kd3_qps,nf_build_s,nf_s,nf_qps,kd3_ok,nf_ok\n";
}

/// Rows are appended as soon as they exist (and flushed per cloud) so that
/// interrupted runs still leave usable partial reports behind.
void append_csv_row(std::ofstream& csv, const Row& r) {
    csv << r.dataset << ',' << r.source << ',' << r.distribution << ',' << r.test << ','
        << r.config << ',' << r.n_points << ',' << r.n_queries << ',' << r.k << ',' << r.build_s
        << ',' << r.kd3_s << ',' << r.kd3_qps << ',' << r.nf_build_s << ',' << r.nf_s << ','
        << r.nf_qps << ',' << r.kd3_ok << ',' << r.nf_ok << '\n';
}

#ifdef KD3_BENCH_NANOFLANN
constexpr bool kHasNanoflann = true;
#else
constexpr bool kHasNanoflann = false;
#endif

/// Aligned summary table; one row per (cloud x configuration x test x distribution).
void print_table(const std::vector<Row>& rows, size_t k) {
    const int W_CLOUD = 24, W_SRC = 10, W_PTS = 11, W_DIST = 20, W_CFG = 10;
    const int W_MS = 10, W_QPS = 12;

    const auto rule = [&] {
        std::cout << std::string(W_CLOUD + W_SRC + W_PTS + W_DIST + W_CFG +
                                 ((kHasNanoflann ? 6 : 3) * (W_MS + W_QPS + 1)) + 24, '-')
                  << "\n";
    };

    const auto dist_label = [&](const Row& r) -> std::string {
        return r.distribution + (r.test[0] == 'n' ? " /1nn" : " /" + std::to_string(k) + "nn");
    };

    std::cout << "\n";
    rule();
    std::cout << std::left << std::setw(W_CLOUD) << "cloud" << std::setw(W_SRC) << "source"
              << std::right << std::setw(W_PTS) << "points" << "  " << std::left
              << std::setw(W_DIST) << "query-set" << std::setw(W_CFG) << "config" << std::right
              << std::setw(W_MS) << "build_ms" << std::setw(W_MS) << "kd3_ms" << std::setw(W_QPS)
              << "kd3_q/s";
    if constexpr (kHasNanoflann) {
        std::cout << std::setw(W_MS) << "nf_bld_ms" << std::setw(W_MS) << "nf_ms"
                  << std::setw(W_QPS) << "nf_q/s";
    }
    std::cout << "  ok\n";
    rule();

    for (const auto& r : rows) {
        std::cout << std::left << std::setw(W_CLOUD) << r.dataset << std::setw(W_SRC) << r.source
                  << std::right << std::setw(W_PTS) << thousands(r.n_points) << "  " << std::left
                  << std::setw(W_DIST) << dist_label(r) << std::setw(W_CFG) << r.config
                  << std::right << std::setw(W_MS) << fixed(r.build_s * 1000.0, 1)
                  << std::setw(W_MS) << fixed(r.kd3_s * 1000.0, 1) << std::setw(W_QPS)
                  << thousands(static_cast<uint64_t>(r.kd3_qps));
        if constexpr (kHasNanoflann) {
            const bool has_nf = r.nf_qps >= 0;
            std::cout << std::setw(W_MS)
                      << (has_nf ? fixed(r.nf_build_s * 1000.0, 1) : std::string("-"))
                      << std::setw(W_MS) << (has_nf ? fixed(r.nf_s * 1000.0, 1) : std::string("-"))
                      << std::setw(W_QPS)
                      << (has_nf ? thousands(static_cast<uint64_t>(r.nf_qps)) : std::string("-"));
            std::cout << "  " << ok_label(r.kd3_ok) << "/" << ok_label(r.nf_ok);
        } else {
            std::cout << "  " << ok_label(r.kd3_ok);
        }
        std::cout << "\n";
    }
    rule();
}

// ---------------------------------------------------------
// Plotly comparison graphs
// ---------------------------------------------------------

/// Emits one grouped-bar query chart per test kind (traces = configurations),
/// plus a build-time chart. Categories are labelled cloud + distribution + test.
void write_html_report(const std::filesystem::path& path, const std::vector<Row>& rows,
                       size_t k) {
    const auto cat_of = [&](const Row& r) {
        return r.dataset + " " + r.distribution +
               (r.test[0] == 'n' ? " /1nn" : " /" + std::to_string(k) + "nn");
    };

    std::vector<std::string> knn_cats, nn1_cats;
    for (const auto& r : rows) {
        auto& v = (r.test[0] == 'n') ? nn1_cats : knn_cats;
        const std::string cat = cat_of(r);
        if (std::find(v.begin(), v.end(), cat) == v.end()) v.push_back(cat);
    }

    struct TraceDef {
        const char* name;
        const char* color;
        const char* config;
        bool nn1;
    };
    const TraceDef qdefs[] = {
        {"base", "#636efa", "base", false},
        {"aabb", "#ef553b", "aabb", false},
        {"nanoflann", "#ab63fa", "base", false}, // stamped reference lives on base rows
        {"base", "#636efa", "base", true},
        {"aabb", "#ef553b", "aabb", true},
        {"fast", "#00cc96", "fast", true},
        {"fast-aabb", "#ff7f0e", "fast-aabb", true},
        {"nanoflann", "#ab63fa", "base", true},
    };

    std::ostringstream charts;
    const char* chart_ids[] = {"q_knn", "q_1nn"};
    for (int kind = 0; kind < 2; ++kind) {
        const bool nn1 = kind == 1;
        const char* cats_var = kind == 0 ? "cats_knn" : "cats_1nn";
        std::ostringstream cats_json, traces;
        const auto& cats = nn1 ? nn1_cats : knn_cats;
        for (size_t i = 0; i < cats.size(); ++i)
            cats_json << (i ? "," : "") << "'" << cats[i] << "'";

        for (const auto& td : qdefs) {
            if (td.nn1 != nn1) continue;
            traces << "{name:'" << td.name << "',type:'bar',marker:{color:'" << td.color
                   << "'},x:" << cats_var << ",y:[";
            const bool is_nf_trace = std::strcmp(td.name, "nanoflann") == 0;
            for (size_t i = 0; i < cats.size(); ++i) {
                double v = -1;
                for (const auto& r : rows) {
                    const bool r_nn1 = r.test[0] == 'n';
                    if (r_nn1 != nn1 || r.config != td.config) continue;
                    if (cat_of(r) != cats[i]) continue;
                    if (is_nf_trace) {
                        if (r.nf_qps < 0) continue;
                        v = r.nf_qps;
                    } else {
                        v = r.kd3_qps;
                    }
                    break;
                }
                traces << (i ? "," : "") << (v >= 0 ? std::to_string(v) : std::string("null"));
            }
            traces << "]},";
        }

        charts << "const " << cats_var << "=[" << cats_json.str() << "];\n"
               << "Plotly.newPlot('" << chart_ids[kind]
               << "',[" << traces.str() << "],"
               << "{barmode:'group',margin:{b:140},title:'"
               << (nn1 ? "Exact 1-NN throughput" : "k-NN throughput (k=")
               << (nn1 ? "'" : std::to_string(k) + ")'")
               << ",xaxis:{title:'',tickangle:-40,automargin:true},"
               << "yaxis:{type:'log',title:'queries/s'}});\n";
    }

    // build-time chart: one category per cloud, one trace per configuration
    std::vector<std::string> clouds;
    for (const auto& r : rows)
        if (std::find(clouds.begin(), clouds.end(), r.dataset) == clouds.end())
            clouds.push_back(r.dataset);

    struct BuildSeries {
        const char* name;
        const char* color;
        const char* config;
        bool nf;
    };
    const BuildSeries bs[] = {
        {"base", "#636efa", "base", false},   {"aabb", "#ef553b", "aabb", false},
        {"fast", "#00cc96", "fast", false},   {"fast-aabb", "#ff7f0e", "fast-aabb", false},
        {"nanoflann", "#ab63fa", "base", true},
    };
    std::ostringstream build_traces, clouds_json;
    for (const auto& series : bs) {
        build_traces << "{name:'" << series.name << "',type:'bar',marker:{color:'"
                     << series.color << "'},x:clouds,y:[";
        for (size_t i = 0; i < clouds.size(); ++i) {
            double v = -1;
            for (const auto& r : rows) {
                if (r.dataset != clouds[i]) continue;
                if (series.nf) {
                    if (r.nf_build_s >= 0) { v = r.nf_build_s * 1000.0; break; }
                } else if (r.config == series.config) {
                    v = r.build_s * 1000.0;
                    break;
                }
            }
            if (i) build_traces << ",";
            build_traces << (v >= 0 ? std::to_string(v) : std::string("null"));
        }
        build_traces << "]},";
    }
    for (size_t i = 0; i < clouds.size(); ++i)
        clouds_json << (i ? "," : "") << "'" << clouds[i] << "'";

    std::ofstream html(path, std::ios::trunc);
    if (!html) return;
    html << "<!doctype html><html><head><meta charset='utf-8'>"
         << "<title>kd3 comparative report</title>"
         << "<script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script></head><body>"
         << "<div id='q_knn' style='width:100%;height:480px'></div>"
         << "<div id='q_1nn' style='width:100%;height:480px'></div>"
         << "<div id='b' style='width:100%;height:420px'></div><script>\n"
         << charts.str()
         << "const clouds=[" << clouds_json.str() << "];\n"
         << "Plotly.newPlot('b',[" << build_traces.str() << "],"
         << "{barmode:'group',title:'build time',margin:{b:80},"
         << "xaxis:{title:'',tickangle:-30,automargin:true},yaxis:{title:'ms'}});\n"
         << "</script></body></html>";
}

/// Marketing view: throughput as % of nanoflann, split volume/surface,
/// one color per configuration. Requires nanoflann reference rows.
void write_relative_report(const std::filesystem::path& path,
                           const std::vector<Row>& rows, size_t k) {
    // Panels mirror the reference report: two test kinds (k-NN, 1-NN) x two query
    // distributions, each rendered as a grouped bar chart with one trace per config.
    struct Panel { const char* test; const char* dist; const char* title; const char* id; };
    const Panel panels[] = {
        {"knn", "bbox-volume", "Volume queries (bbox-volume)", "knn_vol"},
        {"knn", "on-cloud",    "Surface queries (on-cloud)",   "knn_surf"},
        {"nn1", "bbox-volume", "Volume queries (bbox-volume)", "nn1_vol"},
        {"nn1", "on-cloud",    "Surface queries (on-cloud)",   "nn1_surf"},
    };
    const char* cfg_names[]  = {"base", "aabb", "fast", "fast-aabb"};
    const char* cfg_colors[] = {"#636efa", "#ef553b", "#00cc96", "#ff7f0e"};
    // k-NN needs payload trees, so only base/aabb appear there; 1-NN shows every config.
    const bool cfg_in_nn1[] = {true, true, true, true};
    const bool cfg_in_knn[] = {true, true, false, false};

    // X-axis category, e.g. "autzen bbox-volume /10nn" (same convention as the main report).
    const auto label_of = [&](const Row& r) {
        return r.dataset + " " + r.distribution +
               (r.test[0] == 'n' ? " /1nn" : " /" + std::to_string(k) + "nn");
    };

    struct Trace {
        const char* cfg;
        const char* color;
        std::vector<double> rel;    // kd3 q/s as % of nanoflann (-1 if absent)
        std::vector<double> nf_qps; // nanoflann q/s for hover (-1 if absent)
    };
    struct Block {
        const Panel* p;
        std::vector<std::string> cats;
        std::vector<Trace> traces;
    };

    // ---- build per-panel blocks (categories = first-seen row order) ----
    std::vector<Block> blocks;
    for (const auto& p : panels) {
        const bool nn1 = p.test[0] == 'n';
        Block b;
        b.p = &p;

        for (const auto& r : rows) {
            if (r.test != p.test || r.distribution != p.dist) continue;
            const std::string lab = label_of(r);
            if (std::find(b.cats.begin(), b.cats.end(), lab) == b.cats.end())
                b.cats.push_back(lab);
        }
        if (b.cats.empty()) continue;

        for (size_t ci = 0; ci < 4; ++ci) {
            if (!(nn1 ? cfg_in_nn1[ci] : cfg_in_knn[ci])) continue;
            Trace t;
            t.cfg   = cfg_names[ci];
            t.color = cfg_colors[ci];
            t.rel.resize(b.cats.size(), -1.0);
            t.nf_qps.resize(b.cats.size(), -1.0);
            for (const auto& r : rows) {
                if (r.test != p.test || r.distribution != p.dist) continue;
                if (r.config != t.cfg || r.nf_qps < 0 || r.kd3_qps <= 0) continue;
                const auto it = std::find(b.cats.begin(), b.cats.end(), label_of(r));
                const size_t i = static_cast<size_t>(it - b.cats.begin());
                t.rel[i]    = r.kd3_qps / r.nf_qps * 100.0;
                t.nf_qps[i] = r.nf_qps;
            }
            b.traces.push_back(std::move(t));
        }
        blocks.push_back(std::move(b));
    }

    // ---- emit: two sections, each a 2-col grid of <div> + <script> panels ----
    std::ostringstream body;
    const char* last_kind = nullptr; // nullptr | "knn" | "nn1"
    for (const auto& b : blocks) {
        const char* kind = (b.p->test[0] == 'n') ? "nn1" : "knn";
        if (last_kind && std::strcmp(last_kind, kind) != 0) body << "</div>\n";
        if (!last_kind || std::strcmp(last_kind, kind) != 0) {
            const bool nn1 = kind[0] == 'n';
            const std::string heading =
                nn1 ? "k-NN (k=1) with fast queries supported"
                    : "k-NN (k=" + std::to_string(k) + ")";
            body << "<h2 style='margin:10px 0 4px'>" << heading << "</h2>\n"
                 << "<div class='row'>\n";
            last_kind = kind;
        }

        std::ostringstream xarr;
        for (size_t i = 0; i < b.cats.size(); ++i)
            xarr << (i ? "," : "") << '"' << b.cats[i] << '"';

        std::ostringstream traces;
        for (size_t t = 0; t < b.traces.size(); ++t) {
            const Trace& tr = b.traces[t];
            if (t) traces << ", ";
            traces << "{\"name\": \"" << tr.cfg << "\", \"x\": [" << xarr.str() << "], \"y\": [";
            for (size_t i = 0; i < tr.rel.size(); ++i)
                traces << (i ? ", " : "") << (tr.rel[i] >= 0 ? fixed(tr.rel[i], 1) : "null");
            traces << "], \"marker\": {\"color\": \"" << tr.color
                   << "\"}, \"customdata\": [";
            for (size_t i = 0; i < tr.nf_qps.size(); ++i)
                traces << (i ? ", " : "")
                       << (tr.nf_qps[i] >= 0 ? fixed(tr.nf_qps[i], 0) : "null");
            traces << "], \"type\": \"bar\", \"text\": [";
            for (size_t i = 0; i < tr.rel.size(); ++i) {
                if (i) traces << ", ";
                if (tr.rel[i] < 0) { traces << "\"null\""; continue; }
                traces << "\"" << (tr.rel[i] >= 100.0 ? "\u25b2" : "\u25bc")
                       << static_cast<long long>(tr.rel[i] + 0.5) << "%\"";
            }
            traces << "], \"textposition\": \"outside\", \"cliponaxis\": false, "
                      "\"textfont\": {\"color\": \"#333\", \"size\": 12}, \"hovertemplate\": "
                      "\"%{x}<br>%{fullData.name}: %{y:.1f}% of nanoflann "
                      "(%{customdata} q/s)<extra></extra>\"}";
        }

        body << "      <div style='min-width:0'>\n"
             << "        <h3 style='margin:6px 0 2px'>" << b.p->title << "</h3>\n"
             << "        <div id='" << b.p->id << "' style='width:100%;height:460px'></div>\n"
             << "        <script>\n"
             << "        Plotly.newPlot('" << b.p->id << "', [" << traces.str() << "],\n"
             << "          {barmode:'group', height:460,\n"
             << "            xaxis:{tickangle:-30, automargin:true},\n"
             << "            yaxis:{title:'% of nanoflann q/s', zeroline:false, range:[0, null]},\n"
             << "            shapes:[{type:'line', xref:'paper', x0:0, x1:1, y0:100, y1:100,\n"
             << "                     line:{color:'#888', dash:'dot'}}],\n"
             << "            legend:{title:{text:'configuration'}, orientation:'h', y:1.12},\n"
             << "            margin:{t:40}, plot_bgcolor:'#fafafa'}, {responsive:true});\n"
             << "        </script>\n"
             << "      </div>\n";
    }
    if (last_kind) body << "</div>\n";

    std::ofstream html(path, std::ios::trunc);
    if (!html) return;
    html << "<!doctype html><html><head><meta charset='utf-8'>\n"
         << "<title>kd3 vs nanoflann - relative throughput</title>\n"
         << "<script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script>\n"
         << "<style>body{font-family:sans-serif;margin:24px;background:#fff}\n"
         << ".row{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}\n"
         << ".key{color:#555;margin:6px 0 14px}</style></head><body>\n"
         << "<h1>kd3 vs nanoflann \u2014 relative throughput</h1>\n"
         << "<p class='key'>Single-threaded queries. Bar height = kd3 throughput as a percentage "
            "of nanoflann's on identical clouds and query sets.\n"
         << "<b>&#9650;</b> at or above nanoflann \u00b7 <b>&#9660;</b> below it \u00b7 dashed "
            "line marks baseline.</p>\n"
         << body.str() << "</body></html>";
}

// ---------------------------------------------------------
// Benchmarking core
// ---------------------------------------------------------

#ifdef KD3_BENCH_NANOFLANN
template <typename TreeType>
struct Kd3PointAdaptor {
    const std::vector<typename TreeType::FatPoint>& pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
        return pts[idx].coords[dim];
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const {
        return false;
    }
};
template <typename TreeType>
using NanoflannTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, Kd3PointAdaptor<TreeType>>, Kd3PointAdaptor<TreeType>, 3,
    uint32_t>;
#endif

/// Tie-tolerant k-NN comparison: real-world clouds often contain duplicated or
/// equidistant points, so result ORDER between implementations is not defined.
bool knn_results_match(std::span<const LimitsF::KnnResult> got,
                       const std::vector<LimitsF::KnnResult>& brute) {
    if (got.size() != brute.size()) return false;
    for (size_t j = 0; j < brute.size(); ++j) {
        const float tol = 1e-4f * std::max(1.0f, brute[j].dist_sq);
        if (std::fabs(got[j].dist_sq - brute[j].dist_sq) > tol) return false;
    }
    return true;
}

/// one_nn=false -> k-NN comparison (payload trees only).
/// one_nn=true  -> exact 1-NN distance comparison (every configuration).
template <typename TreeType>
bool validate_against_brute_force(const TreeType& tree, const kdbench::QuerySet& queries,
                                  const std::vector<typename TreeType::FatPoint>& reference,
                                  const Options& opts, bool one_nn) {
    const size_t checks = std::min<size_t>(queries.size(), 100);
    for (size_t c = 0; c < checks; ++c) {
        const auto& q = queries[(c * 997) % queries.size()];
        if (!one_nn) {
            std::vector<typename TreeType::KnnResult> buffer(opts.k);
            const auto got = tree.query_knn(q, buffer);
            if (!got) return false;
            if (!knn_results_match(*got, linear_scan(q.data(), opts.k, reference))) return false;
        } else {
            double got = -1;
            if constexpr (TreeType::cfg.has_payload == kd3::cfg_t::has_payload_t::INDEX) {
                const auto res = tree.query_1nn(q);
                if (!res) return false;
                got = res->dist_sq;
            } else {
                const auto res = tree.query_distance2(q);
                if (!res) return false;
                got = *res;
            }
            const auto brute = linear_scan(q.data(), 1, reference);
            if (std::fabs(got - brute.front().dist_sq) >
                1e-4f * std::max(1.0f, brute.front().dist_sq))
                return false;
        }
    }
    return true;
}

/// nanoflann has no kd3-config dependence: measured once per cloud and stamped
/// on the fat-node rows so every configuration stays directly comparable.
struct NfData {
    bool present = false;
    double build_s = -1.0;
    /// Indexed like the shared workload list (bbox-volume, on-cloud).
    struct Dist {
        double s = -1.0, qps = -1.0;
        int ok = -1;
    };
    Dist knn[2] = {}; // k = opts.k
    Dist nn1[2] = {}; // k = 1
};

void measure_nanoflann(const kdbench::PointCloud& reference,
                       const std::array<std::pair<const char*, kdbench::QuerySet>, 2>& workloads,
                       const Options& opts, NfData& out) {
#if defined(KD3_BENCH_NANOFLANN)
    Kd3PointAdaptor<TreeType> adaptor(reference);
    NanoflannTree<TreeType> nf_tree(3, adaptor, {10});
    auto t0 = std::chrono::steady_clock::now();
    nf_tree.buildIndex();
    out.present = true;
    out.build_s = seconds_since(t0);
    std::cout << "[nanoflann] build " << fixed(out.build_s * 1000.0, 1) << " ms\n";

    for (int variant = 0; variant < 2; ++variant) {
        const size_t kk = variant == 0 ? opts.k : 1;
        for (size_t w = 0; w < workloads.size(); ++w) {
            std::vector<uint32_t> nf_indices(kk);
            std::vector<float> nf_dists(kk);
            volatile uint64_t sink = 0;
            t0 = std::chrono::steady_clock::now();
            for (const auto& q : workloads[w].second) {
                nf_tree.knnSearch(q.data(), kk, nf_indices.data(), nf_dists.data());
                sink += nf_indices[0];
            }
            NfData::Dist& dst = variant == 0 ? out.knn[w] : out.nn1[w];
            dst.s   = seconds_since(t0);
            dst.qps = static_cast<double>(workloads[w].second.size()) / dst.s;

            bool ok = true;
            const size_t checks = std::min<size_t>(workloads[w].second.size(), 100);
            for (size_t c = 0; c < checks; ++c) {
                const auto& q = workloads[w].second[(c * 997) % workloads[w].second.size()];
                nf_tree.knnSearch(q.data(), kk, nf_indices.data(), nf_dists.data());
                const auto brute = linear_scan(q.data(), kk, reference);
                for (size_t j = 0; j < kk; ++j) {
                    if (nf_indices[j] != brute[j].payload_id &&
                        nf_dists[j] != brute[j].dist_sq) { ok = false; break; }
                }
                if (!ok) break;
            }
            dst.ok = ok ? 1 : 0;
            std::cout << "[nanoflann/" << workloads[w].first << (variant == 0 ? " k=" : " 1nn k=")
                      << kk << "] " << fixed(dst.s * 1000.0, 1) << " ms "
                      << thousands(static_cast<uint64_t>(dst.qps)) << " q/s | ok "
                      << ok_label(dst.ok) << "\n";
        }
    }
#else
    (void)reference;
    (void)workloads;
    (void)opts;
    (void)out;
#endif
}

/// Runs one kd3 configuration over the shared workload sets.
/// one_nn=false -> k-NN test (payload-bearing trees only);
/// one_nn=true  -> exact 1-NN test (every configuration).
template <typename TreeType>
void run_config(const std::string& name, const std::string& source,
                kdbench::PointCloud&& raw_points,
                const std::array<std::pair<const char*, kdbench::QuerySet>, 2>& workloads,
                const kdbench::PointCloud& reference_cloud, const NfData& nf,
                const Options& opts, bool one_nn, std::vector<Row>& rows) {
    constexpr bool kIsAabb = TreeType::cfg.has_aabb;
    constexpr bool kFat    = TreeType::cfg.has_payload != kd3::cfg_t::has_payload_t::NONE;
    const char* cfg_name   = !kFat ? (kIsAabb ? "fast-aabb" : "fast")
                                   : (kIsAabb ? "aabb" : "base");
    if (!one_nn && !kFat) return; // lightweight nodes cannot answer k-NN

    Row base_row;
    base_row.dataset   = name;
    base_row.source    = source;
    base_row.config    = cfg_name;
    base_row.test      = one_nn ? "nn1" : "knn";
    base_row.n_points  = raw_points.size();
    base_row.n_queries = opts.n_queries;
    base_row.k         = one_nn ? 1 : opts.k;

    std::vector<typename TreeType::FatPoint> points_copy = raw_points;

    auto t0 = std::chrono::steady_clock::now();
    auto tree_result = TreeType::build(raw_points);
    base_row.build_s = seconds_since(t0);
    if (!tree_result) {
        std::cerr << "kd3 build failed!\n";
        std::exit(1);
    }
    const auto& tree = *tree_result;
    std::cout << "[" << cfg_name << (one_nn ? "/1nn]" : "/knn]") << " build kd3 "
              << fixed(base_row.build_s * 1000.0, 1) << " ms\n";

    for (size_t w = 0; w < workloads.size(); ++w) {
        Row row = base_row;
        row.distribution = workloads[w].first;
        const auto& queries = workloads[w].second;

        volatile uint64_t sink = 0;
        t0 = std::chrono::steady_clock::now();

        if (one_nn) {
            for (const auto& q : queries) {
                if constexpr (kFat) {
                    const auto res = tree.query_1nn_inline(q);
                    sink += res->payload_id;
                } else {
                    const auto res = tree.query_distance2_inline(q);
                    sink += static_cast<uint64_t>(res.value());
                }
            }
        } else {
            std::vector<typename TreeType::KnnResult> buffer(opts.k);
            for (const auto& q : queries) {
                const auto res = tree.query_knn_inline(q, buffer);
                sink += res->front().payload_id;
            }
        }
        row.kd3_s   = seconds_since(t0);
        row.kd3_qps = static_cast<double>(queries.size()) / row.kd3_s;
        row.kd3_ok =
            validate_against_brute_force(tree, queries, points_copy, opts, one_nn) ? 1 : 0;

        if (nf.present && (one_nn || kFat)) {
            const NfData::Dist& ref = one_nn ? nf.nn1[w] : nf.knn[w];
            row.nf_build_s = nf.build_s;
            row.nf_s       = ref.s;
            row.nf_qps     = ref.qps;
            row.nf_ok      = ref.ok;
        }

        std::cout << "[" << cfg_name << "/" << row.distribution
                  << (one_nn ? " 1nn]" : " knn]") << " kd3 " << fixed(row.kd3_s * 1000.0, 1)
                  << " ms " << thousands(static_cast<uint64_t>(row.kd3_qps)) << " q/s | ok "
                  << ok_label(row.kd3_ok);
        if (nf.present && (one_nn || kFat))
            std::cout << " | nanoflann " << fixed(row.nf_s * 1000.0, 1) << " ms "
                      << thousands(static_cast<uint64_t>(row.nf_qps)) << " q/s";
        std::cout << "\n";
        rows.push_back(row);
    }
}

} // namespace

int main(int argc, char** argv) {
    const Options opts = parse_options(argc, argv);
    const std::filesystem::path datasets_dir = resolve_datasets_dir(opts, argc > 0 ? argv[0] : "");

    std::filesystem::path out_dir;
    {
        std::error_code ec;
        if (argc > 0 && argv[0]) {
            const auto exe = std::filesystem::canonical(argv[0], ec);
            if (!ec) out_dir = exe.parent_path();
        }
        if (out_dir.empty()) out_dir = std::filesystem::current_path();
    }
    const std::string csv_path  = (out_dir / "kd3_datasets_report.csv").string();
    const std::string html_path = (out_dir / "kd3_comparative_report.html").string();

    std::cout << "kd3 comparative benchmark  [simd: " << TreeType::cfg.simd_parallelism
              << "  leaf_size: " << TreeType::cfg.leaf_size
              << "  tests: knn(k=" << opts.k << ", fat trees)+nn1(all trees)"
              << "  queries/row: " << opts.n_queries << "]\n";
    std::cout << "datasets dir: " << datasets_dir.string() << "\n\n";
    std::cout << "clouds  : every dataset file found above, plus fixed synthetic generators;\n"
                 "          each cloud indexes identically for all compared implementations.\n"
                 "queries : bbox-volume = uniform inside the cloud bounding box (raymarch-like)\n"
                 "          on-cloud    = sampled from the cloud itself, ~0.2% jitter (kNN-like)\n"
                 "tests   : knn = k nearest neighbours, fat trees only (base, aabb)\n"
                 "          nn1 = exact 1-NN minimum distance, every configuration\n"
                 "                (base, aabb, fast, fast-aabb)\n"
                 "ok      : results match brute force within tie/fp tolerance\n\n";

    std::ofstream csv(csv_path, std::ios::trunc);
    if (!csv) {
        std::cerr << "Cannot open " << csv_path << " for writing\n";
        return 1;
    }
    write_csv_header(csv);
    csv.flush();

    // ---------------------------------------------------------
    // Real datasets discovered in the datasets folder
    // ---------------------------------------------------------
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    if (std::filesystem::exists(datasets_dir, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(datasets_dir, ec)) {
            if (!entry.is_regular_file()) continue;
            const auto path = entry.path();
            const auto ext = path.extension().string();
            if (ext == ".bin" || ext == ".xyz" || ext == ".pts" || ext == ".ply" ||
                ext == ".laz" || ext == ".las") {
                files.push_back(path);
            }
        }
        std::sort(files.begin(), files.end());
    }

    constexpr size_t N_POINTS = 1'000'000;
    const struct {
        const char* name;
        kdbench::PointCloud (*generate)(size_t, size_t);
    } synthetics[] = {
        {"synthetic-solid", kdbench::uniform_solid},
        {"synthetic-mixture", kdbench::gaussian_mixture},
        {"synthetic-anisotropic", kdbench::anisotropic_gaussian},
        {"synthetic-sphere-shell", kdbench::sphere_shell},
        {"synthetic-torus", kdbench::torus},
        {"synthetic-blobs", kdbench::sdf_blobs},
        {"synthetic-powerlaw", kdbench::power_law_radial},
    };

    auto derive_workloads = [&](const std::string& name, const kdbench::PointCloud& reference) {
        return std::array<std::pair<const char*, kdbench::QuerySet>, 2>{
            std::pair{"bbox-volume",
                      kdbench::queries_uniform_in_bbox(reference, opts.n_queries,
                                                       kdbench::seed_from_name(name + "/uniform"))},
            std::pair{"on-cloud",
                      kdbench::queries_from_dataset(reference, opts.n_queries,
                                                    kdbench::seed_from_name(name + "/dataset"))}};
    };

    auto emit_cloud = [&](const std::string& name, const std::string& source,
                          kdbench::PointCloud&& base, std::vector<Row>& rows,
                          std::ofstream& csv_out) {
        const kdbench::PointCloud reference = base; // unsorted copy for queries & NF
        const auto workloads = derive_workloads(name, reference);

        NfData nf;
        measure_nanoflann(reference, workloads, opts, nf);

        const size_t first = rows.size();
        {
            auto c = base;
            run_config<TreeType>(name, source, std::move(c), workloads, reference, nf, opts,
                                 false, rows);
        }
        {
            auto c = base;
            run_config<TreeTypeAabb>(name, source, std::move(c), workloads, reference, nf, opts,
                                     false, rows);
        }
        {
            auto c = base;
            run_config<TreeType>(name, source, std::move(c), workloads, reference, nf, opts,
                                 true, rows);
        }
        {
            auto c = base;
            run_config<TreeTypeAabb>(name, source, std::move(c), workloads, reference, nf, opts,
                                     true, rows);
        }
        {
            auto c = base;
            run_config<TreeTypeFast>(name, source, std::move(c), workloads, reference, nf, opts,
                                     true, rows);
        }
        {
            auto c = base;
            run_config<TreeTypeFastAabb>(name, source, std::move(c), workloads, reference, nf,
                                         opts, true, rows);
        }
        for (size_t i = first; i < rows.size(); ++i) append_csv_row(csv_out, rows[i]);
        csv_out.flush();
    };

    std::vector<Row> rows;

    for (const auto& file : files) {
        auto loaded = kdbench::load_dataset(file);
        if (!loaded) {
            std::cerr << "[skip] " << loaded.error() << "\n";
            continue;
        }
        std::cout << "\n[load] " << file.stem().string() << " -> " << thousands(loaded->size())
                  << " points\n";
        emit_cloud(file.stem().string(), "file", std::move(*loaded), rows, csv);
    }
    if (files.empty()) {
        std::cout << "no dataset files found (run `xmake run fetch-datasets`, or pass --dir=PATH)\n";
    }

    for (const auto& s : synthetics) {
        std::cout << "\n[load] " << s.name << " (generator) -> " << thousands(N_POINTS)
                  << " points\n";
        emit_cloud(s.name, "generator", s.generate(N_POINTS, kdbench::seed_from_name(s.name)),
                   rows, csv);
    }

    print_table(rows, opts.k);
    write_html_report(html_path, rows, opts.k);
#ifdef KD3_BENCH_NANOFLANN
    const std::string rel_path = (out_dir / "kd3_relative_report.html").string();
    write_relative_report(rel_path, rows, opts.k);
#endif
    std::cout << "\nCSV report:   " << csv_path << "\n"
              << "HTML graphs:  " << html_path << "\n";
#ifdef KD3_BENCH_NANOFLANN
    std::cout << "Relative view: " << rel_path << "\n";
#endif
    return 0;
}
