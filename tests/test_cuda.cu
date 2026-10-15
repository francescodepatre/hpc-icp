// Verifica che la versione CUDA dia gli stessi risultati della sequenziale,
// per diverse dimensioni di blocco e per i casi limite (n non multiplo del blocco, nessuna coppia).
//
// Tolleranze piu' larghe di test_omp: sulla GPU nvcc puo' fondere moltiplicazioni e somme (FMA)
// in modo diverso dalla CPU, e due vicini quasi a pari distanza possono scambiarsi.
#include <cmath>
#include <cstdio>
#include <random>

#include "grid.h"
#include "icp.h"
#include "icp_cuda.h"
#include "kabsch.h"
#include "metrics.h"

static int failures = 0;
#define CHECK(cond, ...)                              \
    do {                                              \
        if (!(cond)) {                                \
            ++failures;                               \
            std::printf("  FALLITO: %s  -> ", #cond); \
            std::printf(__VA_ARGS__);                 \
            std::printf("\n");                        \
        }                                             \
    } while (0)

static Rigid make_rigid(double ax, double ay, double az, double deg, double tx, double ty, double tz) {
    const double n = std::sqrt(ax * ax + ay * ay + az * az);
    ax /= n; ay /= n; az /= n;
    const double a = deg * 3.14159265358979323846 / 180.0, c = std::cos(a), s = std::sin(a), C = 1 - c;
    Rigid T;
    T.R[0] = c + ax * ax * C;      T.R[1] = ax * ay * C - az * s; T.R[2] = ax * az * C + ay * s;
    T.R[3] = ay * ax * C + az * s; T.R[4] = c + ay * ay * C;      T.R[5] = ay * az * C - ax * s;
    T.R[6] = az * ax * C - ay * s; T.R[7] = az * ay * C + ax * s; T.R[8] = c + az * az * C;
    T.t[0] = tx; T.t[1] = ty; T.t[2] = tz;
    return T;
}

static PointCloud make_surface(size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0, 1);
    PointCloud pc;
    for (size_t i = 0; i < n; ++i) {
        const double th = std::acos(1 - 2 * u(rng)), ph = 2 * 3.14159265358979323846 * u(rng);
        const double r = 1 + 0.25 * std::sin(3 * th) * std::cos(2 * ph) + 0.1 * std::cos(5 * ph);
        pc.push_back(static_cast<float>(r * std::sin(th) * std::cos(ph)),
                     static_cast<float>(0.6 * r * std::sin(th) * std::sin(ph)),
                     static_cast<float>(0.3 * r * std::cos(th)));
    }
    return pc;
}


static PointCloud make_source(const PointCloud& base, const Rigid& G, unsigned seed, double sigma) {
    PointCloud src;
    const Rigid Gi = rigid_inverse(G);
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0, sigma);
    for (size_t i = 0; i < base.size(); ++i) {
        double o[3];
        rigid_apply(Gi, base.x[i], base.y[i], base.z[i], o);
        src.push_back(static_cast<float>(o[0] + noise(rng)), static_cast<float>(o[1] + noise(rng)),
                      static_cast<float>(o[2] + noise(rng)));
    }
    return src;
}

static void confronta(const char* nome, const PointCloud& src, const PointCloud& tgt, const IcpParams& prm,
                      const CudaOptions& opt) {
    IcpResult rs = icp_seq(src, tgt, prm);
    CudaInfo ci;
    IcpResult rc = icp_cuda(src, tgt, prm, rigid_identity(), opt, &ci);
    double rot, tr;
    pose_error(rc.T, rs.T, rot, tr);
    const double drmse = std::fabs(rc.rmse.back() - rs.rmse.back());
    const long long dm = std::llabs(static_cast<long long>(rc.matches.back()) - static_cast<long long>(rs.matches.back()));
    std::printf("  %-28s blocchi nn=%4d other=%4d | seq %.3f s, cuda %.3f s | diff: rot %.1e gradi, trasl %.1e, rmse %.1e, coppie %zu vs %zu\n",
                nome, opt.block_nn, opt.block_other, rs.tm.total, rc.tm.total, rot, tr, drmse, rc.matches.back(),
                rs.matches.back());
    CHECK(rc.rmse.size() == rs.rmse.size(), "numero di iterazioni diverso: %zu vs %zu", rc.rmse.size(), rs.rmse.size());
    CHECK(rot < 1e-3 && tr < 1e-4, "T diversa dalla sequenziale (%s)", nome);
    CHECK(drmse < 1e-4, "RMSE diverso (%s)", nome);
    CHECK(dm <= static_cast<long long>(rs.matches.back() / 1000 + 2), "numero di coppie diverso (%s)", nome);
}

int main() {
    std::printf("[ICP CUDA vs sequenziale]\n");
    PointCloud tgt = make_surface(10000, 11), base = make_surface(10000, 12);
    const Rigid G = make_rigid(1, 2, 3, 6.0, 0.05, -0.03, 0.02);
    PointCloud src = make_source(base, G, 21, 0.002);

    IcpParams prm;
    prm.dmax = 0.15f;
    prm.iters = 30;

    const int blocks[] = {32, 64, 128, 256, 1024};
    for (int b : blocks) {
        CudaOptions opt;
        opt.block_nn = b;
        opt.block_other = b;
        confronta("superficie 10000 punti", src, tgt, prm, opt);
    }

    {   // n non multiplo del blocco, nuvola piu' piccola della griglia di blocchi
        PointCloud t2 = make_surface(1237, 5), b2 = make_surface(1001, 6);
        PointCloud s2 = make_source(b2, G, 7, 0.002);
        confronta("n dispari (1001 / 1237)", s2, t2, prm, CudaOptions());
    }

    {   // un cluster molto denso: celle con carichi molto diversi
        PointCloud t3 = tgt, s3 = src;
        std::mt19937 rng(9);
        std::normal_distribution<float> g(0.2f, 0.02f);
        for (int i = 0; i < 8000; ++i) { t3.push_back(g(rng), g(rng), g(rng)); s3.push_back(g(rng), g(rng), g(rng)); }
        confronta("con cluster denso", s3, t3, prm, CudaOptions());
    }

    {   // convergenza verso la trasformazione vera
        IcpResult rc = icp_cuda(src, tgt, prm);
        double rot, tr;
        pose_error(rc.T, G, rot, tr);
        std::printf("  errore vs verita': rot %.4f gradi, trasl %.5f\n", rot, tr);
        CHECK(rot < 0.2 && tr < 0.01, "ICP CUDA non converge alla trasformazione vera");
    }

    {   // nessuna coppia valida: sorgente lontanissima. Seq e CUDA devono fermarsi alla prima iterazione.
        PointCloud far;
        for (size_t i = 0; i < src.size(); ++i) far.push_back(src.x[i] + 1000.f, src.y[i], src.z[i]);
        IcpResult rs = icp_seq(far, tgt, prm);
        IcpResult rc = icp_cuda(far, tgt, prm);
        std::printf("  nessuna coppia: seq %zu iter, cuda %zu iter, coppie %zu / %zu\n", rs.rmse.size(), rc.rmse.size(),
                    rs.matches.back(), rc.matches.back());
        CHECK(rc.rmse.size() == rs.rmse.size() && rc.matches.back() == 0, "caso senza coppie gestito male");
    }

    {   // parametri invalidi: devono dare un'eccezione, non un crash
        bool thrown = false;
        CudaOptions bad;
        bad.block_nn = 100;
        try { icp_cuda(src, tgt, prm, rigid_identity(), bad); } catch (const std::exception&) { thrown = true; }
        CHECK(thrown, "block_nn=100 non rifiutato");
    }

    std::printf("\n%s (%d controlli falliti)\n", failures ? "TEST FALLITI" : "TUTTI I TEST OK", failures);
    return failures ? 1 : 0;
}
