// VESTA SEAL bench: sec_comp + d unified-level matmuls (+masks) + product reduction
// (mirrors vesta_protocol.py). Emits collector-schema JSON, checks golden.
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
  if (exported.empty()) { std::cerr << "usage: vesta_bench --exported <json> --config OD|UD --out <dir>\n"; return 2; }
  (void)threads;

  ModelData md = load_model(exported);
  int scbits = (md.p <= 8) ? 8 : 16;
  FheParams pp = pick_params("VESTA", config, md.p);
  if (m_override > 0) pp.m = m_override;
  const VestaT& V = md.vesta;
  try {
    Backend be(pp, scheme != "bfv");
    auto t_enc0 = clk::now();
    auto ct_thresh = be.encrypt_matrix(V.thresh_bits);
    auto ct_facts  = be.encrypt_matrix(V.facts_bits);
    std::vector<std::vector<imm::Ctxt>> ct_ulvls;
    for (const auto& ul : V.ulvls_diag) ct_ulvls.push_back(be.encrypt_matrix(ul));
    auto ct_masks = be.encrypt_matrix(V.masks);
    double t_enc = ms(t_enc0, clk::now());

    auto t0 = clk::now(); auto ct_dec = be.sec_comp(ct_facts, ct_thresh, V.sc_length, scbits); double t_sc = ms(t0, clk::now());

    auto t1 = clk::now();
    std::vector<imm::Ctxt> lvl_results;
    for (size_t i = 0; i < ct_ulvls.size(); ++i) {
      auto cur = be.mat_mul(ct_ulvls[i], ct_dec, V.mm_colnum,
                              be.replication_rounds((long)V.ulvls_diag[i][0].size(), V.mm_colnum));
      be.add_inplace(cur, ct_masks[i]);
      lvl_results.push_back(std::move(cur));
    }
    double t_lvl = ms(t1, clk::now());

    auto t2 = clk::now(); auto ct_leafs = be.reduction(lvl_results); double t_red = ms(t2, clk::now());
    int nb_final = be.noise_budget(ct_leafs);
    auto t3 = clk::now(); auto dec = be.decrypt(ct_leafs, md.L); double t_dec = ms(t3, clk::now());

    Row res(md.L, 0); for (long i = 0; i < md.L; ++i) res[i] = (dec[i] == V.answer) ? 1 : 0;
    bool ok = (res == md.golden);
    double peak = peak_rss_mb();
    std::map<std::string, double> tm = {
      {"encryption", t_enc}, {"SecComp", t_sc}, {"Reordering", 0.0}, {"LvlProcessing", t_lvl},
      {"Reduction", t_red}, {"decryption", t_dec},
      {"evaluation", t_sc + t_lvl + t_red}, {"total", t_enc + t_sc + t_lvl + t_red + t_dec} };
    write_result_json(out, "VESTA", config, md.name, tm, peak);

    std::cout << "VESTA " << md.name << " [" << config << "] golden=" << (ok ? "MATCH" : "MISMATCH") << " nb=" << nb_final
              << " sec=" << be.security() << " (N=" << be.ring() << ",bits=" << be.totbits() << ",depth=" << pp.depth << ")"
              << " enc=" << t_enc << " sc=" << t_sc << " lvl=" << t_lvl << " red=" << t_red
              << " peakMB=" << peak << std::endl;
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "VESTA " << md.name << " [" << config << "] ERROR: " << e.what() << std::endl;
    return 4;
  }
}
