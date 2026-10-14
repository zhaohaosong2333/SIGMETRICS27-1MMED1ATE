// 1MM (IMM) SEAL bench: sec_comp + ONE diagonal matmul (mirrors imm_protocol.py).
// Emits collector-schema JSON (experiment_results_SEAL_...), asserts decrypted winning-leaf == golden.
#include "immed1ate/imm_core.h"
#include "immed1ate/io.h"
#include "immed1ate/params.h"
#include <chrono>
#include <iostream>
#include <string>
#include <cstring>
using namespace imm;
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int main(int argc, char** argv) {
  std::string exported, config = "OD", out = ".", scheme = "bgv";
  long threads = 1, m_override = -1;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--exported") && i + 1 < argc) exported = argv[++i];
    else if (!std::strcmp(argv[i], "--config") && i + 1 < argc) config = argv[++i];
    else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atol(argv[++i]);
    else if (!std::strcmp(argv[i], "--m") && i + 1 < argc) m_override = std::atol(argv[++i]);
    else if (!std::strcmp(argv[i], "--scheme") && i + 1 < argc) scheme = argv[++i];
  }
  if (exported.empty()) { std::cerr << "usage: imm_bench --exported <json> --config OD|UD --out <dir>\n"; return 2; }
  (void)threads;   // SEAL core is single-threaded

  ModelData md = load_model(exported);
  int scbits = (md.p <= 8) ? 8 : 16;
  FheParams pp = pick_params("IMM", config, md.p);
  if (m_override > 0) pp.m = m_override;
  try {
    Backend be(pp, scheme != "bfv");
    auto t_enc0 = clk::now();
    auto ct_dthresh = be.encrypt_matrix(md.imm.dthresh_bits);
    auto ct_dfacts  = be.encrypt_matrix(md.imm.dfacts_bits);
    auto ct_em      = be.encrypt_matrix(md.imm.em_diag);
    double t_enc = ms(t_enc0, clk::now());

    auto t0 = clk::now(); auto ct_dec = be.sec_comp(ct_dfacts, ct_dthresh, md.imm.sc_length, scbits); double t_sc = ms(t0, clk::now());
    auto t1 = clk::now(); auto ct_leafs = be.mat_mul(ct_em, ct_dec, md.imm.mm_colnum,
                                                                  be.replication_rounds((long)md.imm.em_diag[0].size(), md.imm.mm_colnum)); double t_mm = ms(t1, clk::now());
    int nb_final = be.noise_budget(ct_leafs);
    auto t2 = clk::now(); auto dec = be.decrypt(ct_leafs, md.L); double t_dec = ms(t2, clk::now());

    Row res(md.L, 0); for (long i = 0; i < md.L; ++i) res[i] = (dec[i] == md.imm.answer) ? 1 : 0;
    bool ok = (res == md.golden);
    double peak = peak_rss_mb();
    std::map<std::string, double> tm = {
      {"encryption", t_enc}, {"SecComp", t_sc}, {"MatMul", t_mm}, {"Reordering", 0.0}, {"Reduction", 0.0},
      {"decryption", t_dec}, {"evaluation", t_sc + t_mm}, {"total", t_enc + t_sc + t_mm + t_dec} };
    write_result_json(out, "IMM", config, md.name, tm, peak);

    long wf = 0, wg = 0; for (long x : res) wf += x; for (long x : md.golden) wg += x;
    std::cout << "IMM " << md.name << " [" << config << "] golden=" << (ok ? "MATCH" : "MISMATCH") << " nb=" << nb_final
              << " sec=" << be.security() << " (N=" << be.ring() << ",bits=" << be.totbits() << ",depth=" << pp.depth << ")"
              << " enc=" << t_enc << " sc=" << t_sc << " mm=" << t_mm << " peakMB=" << peak
              << " win_fhe=" << wf << " win_gold=" << wg << std::endl;
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "IMM " << md.name << " [" << config << "] ERROR: " << e.what() << std::endl;
    return 4;
  }
}
