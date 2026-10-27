// Reference run for the SIFT1M comparison, same parameters as examples/sift.rs.
//
//   git clone https://github.com/nmslib/hnswlib
//   g++ -O3 -march=native -std=c++17 -I hnswlib benches/compare/hnswlib_sift.cpp -o hnswlib_sift
//   ./hnswlib_sift <dir with sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs> [threads]
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
    size_t n, d, nq, dq, ng, kg;
    auto base = read_vecs<float>(dir + "/sift_base.fvecs", n, d);
    auto qs = read_vecs<float>(dir + "/sift_query.fvecs", nq, dq);
    auto gt = read_vecs<int>(dir + "/sift_groundtruth.ivecs", ng, kg);

    hnswlib::L2Space space(d);
    hnswlib::HierarchicalNSW<float> idx(&space, n, 16, 200);

    auto t0 = Clock::now();
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++)
        pool.emplace_back([&] {
            for (size_t i; (i = next++) < n;) idx.addPoint(&base[i * d], i);
        });
    for (auto &th : pool) th.join();
    double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    printf("hnswlib: n=%zu dim=%zu build=%.1fs (%.0f inserts/s on %d threads)\n", n, d, secs, n / secs, threads);

    printf("%6s %10s %10s %10s\n", "ef", "recall@10", "p50 us", "QPS");
    for (size_t ef : {16, 32, 64, 128, 256, 512}) {
        idx.setEf(ef);
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
        printf("%6zu %10.4f %10.0f %10.0f\n", ef, hit / (nq * 10.0), lat[lat.size() / 2], nq / (total / 1e6));
    }
}
