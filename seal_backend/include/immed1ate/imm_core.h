// 1MMED1ATE SEAL (Microsoft SEAL 4.1, BGV) backend core.
// Reimplements src/immed1ate/fhe/bgv.py::BGVScheme against
// SEAL's RNS-BGV. Common Backend interface, so imm_bench/vesta_bench/copse_bench and
// io.* are reused unchanged except the ciphertext typedef (imm::Ctxt) and the two 'cur += mask' sites
// (-> add_inplace, since seal::Ciphertext has no operator+=).
//
// SEAL, unlike OpenFHE, does NOT auto-track levels: every binary op needs operands at the same
// modulus level, so mul()/add_inplace() explicitly mod-switch to align (see .cpp). We mod-switch after
// every multiply (standard BGV noise management). SEALContext validates tc128 => every context is
// >=128-bit by construction.
#pragma once
#include <seal/seal.h>
#include "immed1ate/params.h"
#include <vector>
#include <map>
#include <memory>

namespace imm {

using Row  = std::vector<long>;
using Mat  = std::vector<std::vector<long>>;
using Ctxt = seal::Ciphertext;                 // benches say imm::Ctxt / Ctxt

class Backend {
public:
  Backend(const FheParams& pp, bool bgv = true, long p = 65537);   // bgv=false -> BFV scheme

  long   nslots()  const { return nslots_; }
  double security() const { return sec_; }     // 128.0 if tc128-valid (always), else -1
  long   ring()    const { return N_; }
  long   totbits() const { return totbits_; }
  int    nprimes() const { return nprimes_; }

  std::vector<Ctxt> encrypt_matrix(const Mat& mat) const;   // one Ctxt per row
  Ctxt              encrypt_vector(const Row& v)   const;
  std::vector<long> decrypt(const Ctxt& ct, long len) const;

  Ctxt eval_rotate(Ctxt ct, long amount) const;             // OpenFHE-EvalRotate semantics
  void add_inplace(Ctxt& a, const Ctxt& b) const;           // benches' 'cur += ct_masks[i]'

  // bit-sliced a<b over `bits` bit-planes (verbatim structure of bgv.py::sec_comp_{8,16}bit)
  Ctxt sec_comp(const std::vector<Ctxt>& a, const std::vector<Ctxt>& b, long length, int bits) const;
  // diagonal ctxt x ctxt matrix-vector product (bgv.py::mat_mul)
  Ctxt mat_mul(const std::vector<Ctxt>& M, Ctxt v, long colnum, int recursive_time = 2) const;
  // rotate-and-add rounds (copies - 1) the diagonal MatMul needs for a rows x colnum operand; throws if
  // the copies do not fit one rotation row (N/2 slots), instead of letting rotate_rows wrap them around
  int replication_rounds(long rows, long colnum) const;
  // remaining invariant noise budget (bits) of a ciphertext; 0 => decryption is no longer reliable
  int noise_budget(const Ctxt& c) const;
  // balanced binary-tree product (operations.py::fhe_reduction)
  Ctxt reduction(std::vector<Ctxt> cts) const;

private:
  int  level_of(const Ctxt& c) const;                       // chain index (0 = decryption level)
  void align(Ctxt& a, Ctxt& b) const;                       // mod-switch both to min level
  Ctxt mul(const Ctxt& a, const Ctxt& b) const;             // align, multiply, relin, mod-switch-next
  void add_ones(Ctxt& ct, long length) const;               // ct + [1]*length (plaintext)
  Ctxt eq(const Ctxt& a, const Ctxt& b, long length) const; // 1-(b-a)^2
  Ctxt lt(const Ctxt& a, const Ctxt& b, long length) const; // (1-a)*b

  seal::SEALContext ctx_;
  std::unique_ptr<seal::BatchEncoder> benc_;
  std::unique_ptr<seal::Encryptor>    encr_;
  std::unique_ptr<seal::Decryptor>    decr_;
  std::unique_ptr<seal::Evaluator>    eval_;
  seal::PublicKey pk_;
  seal::SecretKey sk_;
  seal::RelinKeys rlk_;
  seal::GaloisKeys glk_;
  long N_ = 0, nslots_ = 0, totbits_ = 0;
  int  nprimes_ = 0;
  double sec_ = -1;
  bool bgv_ = true;

  static seal::SEALContext build_ctx(const FheParams& pp, long p, bool bgv, long& N_out,
                                     long& totbits_out, int& nprimes_out);
};

}  // namespace imm
