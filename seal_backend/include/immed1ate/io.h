// Exported-tensor reader + result-JSON writer (schema consumed unchanged by
// reproduce/collect_speed_csv.py and collect_memory_wide_csv.py) + peak-RSS.
#pragma once
#include <string>
#include <vector>
#include <map>

namespace imm {

using Row = std::vector<long>;
using Mat = std::vector<std::vector<long>>;

struct ImmT   { long answer, sc_length, mm_colnum;
                Mat dfacts_bits, dthresh_bits, em_diag; };
struct VestaT { long answer, sc_length, mm_colnum;
                Mat facts_bits, thresh_bits, masks; std::vector<Mat> ulvls_diag; };
struct CopseT { long answer, sc_length, reshuf_colnum, lvl_colnum;
                Mat facts_bits, thresh_bits, reshuf_diag, masks; std::vector<Mat> lvls_diag; };

struct ModelData {
  std::string name;
  long p, d, b, q, k, L;
  Row golden;
  ImmT imm; VestaT vesta; CopseT copse;
};

ModelData load_model(const std::string& json_path);

// peak resident set size (MB) from /proc/self/status VmHWM
double peak_rss_mb();

// Writes experiment_results_HELIB_<config>_<model>_<PROTO>.json into `dir`, in the exact
// merge-able schema the collectors parse. `proto` in {IMM,VESTA,COPSE}. timings in ms.
void write_result_json(const std::string& dir, const std::string& proto,
                       const std::string& config, const std::string& model,
                       const std::map<std::string, double>& timings, double peak_mb);

}  // namespace imm
