// Reference run for the SIFT1M comparison, same parameters as examples/sift.rs.
//
//   git clone https://github.com/nmslib/hnswlib
//   g++ -O3 -march=native -std=c++17 -static -I hnswlib benches/compare/hnswlib_sift.cpp -o hnswlib_sift
//   ./hnswlib_sift <dir with sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs> [threads] [passes]
//
// HNSW_INDEX=<file>: load the index from it if it exists, otherwise build and save it there.
// Queries run on a loaded index in a fresh process, the same protocol as SIFT_SNAPSHOT in
// examples/sift.rs, and the fastest of `passes` passes is reported.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "hnswlib/hnswlib.h"

using Clock = std::chrono::steady_clock;

template <class T>
static std::vector<T> read_vecs(const std::string &path, size_t &n, size_t &d) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { perror(path.c_str()); exit(1); }
    int dim;
    if (fread(&dim, 4, 1, f) != 1) exit(1);
    d = dim;
    fseek(f, 0, SEEK_END);
    n = ftell(f) / (4 + 4 * d);
    rewind(f);
    std::vector<T> out(n * d);
    for (size_t i = 0; i < n; i++) {
        int dd;
        if (fread(&dd, 4, 1, f) != 1 || fread(&out[i * d], 4, d, f) != d) exit(1);
    }
    fclose(f);
    return out;
}

int main(int argc, char **argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    int threads = argc > 2 ? atoi(argv[2]) : 8;
    int passes = argc > 3 ? atoi(argv[3]) : 3;
    const char *cache = getenv("HNSW_INDEX");
    size_t n, d, nq, dq, ng, kg;
    auto base = read_vecs<float>(dir + "/sift_base.fvecs", n, d);
    auto qs = read_vecs<float>(dir + "/sift_query.fvecs", nq, dq);
    auto gt = read_vecs<int>(dir + "/sift_groundtruth.ivecs", ng, kg);

    hnswlib::L2Space space(d);
    hnswlib::HierarchicalNSW<float> *idxp;
    FILE *probe = cache ? fopen(cache, "rb") : nullptr;
    if (probe) {
        fclose(probe);
        auto t0 = Clock::now();
        idxp = new hnswlib::HierarchicalNSW<float>(&space, cache, false, n);
        printf("hnswlib: loaded in %.1fs\n", std::chrono::duration<double>(Clock::now() - t0).count());
    } else {
        idxp = new hnswlib::HierarchicalNSW<float>(&space, n, 16, 200);
        auto t0 = Clock::now();
        std::atomic<size_t> next{0};
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; t++)
            pool.emplace_back([&] {
                for (size_t i; (i = next++) < n;) idxp->addPoint(&base[i * d], i);
            });
        for (auto &th : pool) th.join();
        double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        printf("hnswlib: n=%zu dim=%zu build=%.1fs (%.0f inserts/s on %d threads)\n", n, d, secs, n / secs, threads);
        if (cache) idxp->saveIndex(cache);
    }
    auto &idx = *idxp;

    printf("%6s %10s %10s %10s\n", "ef", "recall@10", "p50 us", "QPS");
    for (size_t ef : {16, 32, 64, 128, 256, 512}) {
        idx.setEf(ef);
        double best_total = 1e300, best_p50 = 0, recall = 0;
        for (int pass = 0; pass < passes; pass++) {
            std::vector<double> lat;
            size_t hit = 0;
            for (size_t q = 0; q < nq; q++) {
                auto s = Clock::now();
                auto res = idx.searchKnn(&qs[q * d], 10);
                lat.push_back(std::chrono::duration<double, std::micro>(Clock::now() - s).count());
                while (!res.empty()) {
                    int id = (int)res.top().second;
                    res.pop();
                    for (size_t j = 0; j < 10; j++)
                        if (gt[q * kg + j] == id) { hit++; break; }
                }
            }
            std::sort(lat.begin(), lat.end());
            double total = 0;
            for (double l : lat) total += l;
            if (total < best_total) { best_total = total; best_p50 = lat[lat.size() / 2]; }
            recall = hit / (nq * 10.0);
        }
        printf("%6zu %10.4f %10.0f %10.0f\n", ef, recall, best_p50, nq / (best_total / 1e6));
    }
}
