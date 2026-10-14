#include "immed1ate/io.h"
#include "json.hpp"
#include <fstream>
#include <sstream>
#include <cstdio>

namespace imm {
using nlohmann::json;

static Mat as_mat(const json& j) { return j.get<Mat>(); }
static std::vector<Mat> as_mats(const json& j) {
  std::vector<Mat> out; out.reserve(j.size());
  for (const auto& m : j) out.push_back(m.get<Mat>());
  return out;
}

ModelData load_model(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  json j; f >> j;

  ModelData m;
  m.name = j.at("name").get<std::string>();
  m.p = j.at("p"); m.d = j.at("d"); m.b = j.at("b");
  m.q = j.at("q"); m.k = j.at("k"); m.L = j.at("L");
  m.golden = j.at("golden").get<Row>();

  const auto& ji = j.at("imm");
  m.imm.answer = ji.at("answer"); m.imm.sc_length = ji.at("sc_length"); m.imm.mm_colnum = ji.at("mm_colnum");
  m.imm.dfacts_bits = as_mat(ji.at("dfacts_bits"));
  m.imm.dthresh_bits = as_mat(ji.at("dthresh_bits"));
  m.imm.em_diag = as_mat(ji.at("em_diag"));

  const auto& jv = j.at("vesta");
  m.vesta.answer = jv.at("answer"); m.vesta.sc_length = jv.at("sc_length"); m.vesta.mm_colnum = jv.at("mm_colnum");
  m.vesta.facts_bits = as_mat(jv.at("facts_bits"));
  m.vesta.thresh_bits = as_mat(jv.at("thresh_bits"));
  m.vesta.ulvls_diag = as_mats(jv.at("ulvls_diag"));
  m.vesta.masks = as_mat(jv.at("masks"));

  const auto& jc = j.at("copse");
  m.copse.answer = jc.at("answer"); m.copse.sc_length = jc.at("sc_length");
  m.copse.reshuf_colnum = jc.at("reshuf_colnum"); m.copse.lvl_colnum = jc.at("lvl_colnum");
  m.copse.facts_bits = as_mat(jc.at("facts_bits"));
  m.copse.thresh_bits = as_mat(jc.at("thresh_bits"));
  m.copse.reshuf_diag = as_mat(jc.at("reshuf_diag"));
  m.copse.lvls_diag = as_mats(jc.at("lvls_diag"));
  m.copse.masks = as_mat(jc.at("masks"));
  return m;
}

double peak_rss_mb() {
  std::ifstream s("/proc/self/status");
  std::string line;
  while (std::getline(s, line)) {
    if (line.rfind("VmHWM:", 0) == 0) {
      std::istringstream is(line.substr(6));
      long kb = 0; is >> kb;
      return kb / 1024.0;
    }
  }
  return -1.0;
}

void write_result_json(const std::string& dir, const std::string& proto,
                       const std::string& config, const std::string& model,
                       const std::map<std::string, double>& timings, double peak_mb) {
  json entry;
  entry["model_name"] = model;
  json r;
  for (const auto& kv : timings) r[kv.first] = kv.second;
  r["peak_memory_mb"] = peak_mb;
  entry["results"][proto] = r;
  json arr = json::array({entry});

  std::string path = dir + "/experiment_results_SEAL_" + config + "_" + model + "_" + proto + ".json";
  std::ofstream o(path);
  o << arr.dump(2) << "\n";
}

}  // namespace imm
