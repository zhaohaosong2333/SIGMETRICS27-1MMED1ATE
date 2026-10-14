// BGV parameter selection for the SEAL backend. SEAL is RNS-BGV: ~40 modulus-bits/level, and its
// SEALContext validates every parameter set against sec_level_type::tc128, so EVERY config is
// >=128-bit by construction. We keep the SAME multiplicative-depth formula as OpenFHE
// (config.py::calculate_fhe_depth) so OD/UD mean the same thing across backends, and let SEAL pick
// the smallest ring that holds depth+1 chain primes at 128-bit -- so 1MM-OD lands on a smaller ring
// (its OD advantage survives), exactly as in OpenFHE.
#pragma once
#include <string>
namespace imm {

// m := poly_modulus_degree N (0 = auto-pick from depth). bits/c advisory. depth := #levels.
// spare := extra modulus primes beyond `depth` (noise headroom). 1MM (no product-tree reduction)
// needs few -> keeps its shallow OD circuit on the small ring N=16384; VESTA/COPSE reduce a product
// of deep, big-model operands and need more mod-switch resets, so they get more spare (they sit on
// N=32768 anyway, where there is budget for it).
struct FheParams { long m, bits, c; int depth; int spare; };

inline FheParams pick_params(const std::string& proto, const std::string& config, long p_precision) {
  int pbl = (p_precision <= 8) ? 4 : 5;   // p.bit_length(): 8->4, 16->5   (mirror OpenFHE)
  int dbl = 3;                            // d.bit_length() for d in [4..7]
  int depth;
  if (config == "UD")        depth = pbl + dbl + 2;   // unified = COPSE's depth
  else if (proto == "IMM")   depth = pbl + 1;
  else if (proto == "VESTA") depth = pbl + dbl + 1;
  else                       depth = pbl + dbl + 2;   // COPSE OD
  int spare = (proto == "IMM") ? 4 : 5;   // VESTA/COPSE reductions need one more level of headroom
  return {0 /*auto N*/, 0, 0, depth, spare};
}

}  // namespace imm
