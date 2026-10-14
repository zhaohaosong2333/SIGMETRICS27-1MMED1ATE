#include "immed1ate/imm_core.h"
#include <stdexcept>
#include <cstdlib>
#include <algorithm>

namespace imm {
using namespace seal;

// OpenFHE EvalRotate(ct,k) == SEAL rotate_rows(ct, ROT_SIGN*k). The diagonal matmul is designed for
// OpenFHE semantics; ROT_SIGN is fixed by tests/rotation_test + matmul_test (flip if a gate fails).
static constexpr int ROT_SIGN = 1;

// ---- parameter selection: keep OpenFHE's multiplicative-depth, let SEAL pick the smallest 128-bit ring ----
SEALContext Backend::build_ctx(const FheParams& pp, long p, bool bgv, long& N_out, long& totbits_out, int& nprimes_out) {
  int depth = pp.depth;
  int spare = pp.spare > 0 ? pp.spare : 4;   // protocol-set headroom (1MM=4 keeps N=16384; V/C=5)
  int nprimes = depth + spare;               // (depth+spare-1) chain + 1 special => depth+spare-1 mod-switches

  // pick the SMALLEST 128-bit ring that holds nprimes ~40-bit primes -- so 1MM-OD (shallow) lands on a
  // small ring (N=16384) while VESTA/COPSE/UD (deep) need N=32768, exactly the OpenFHE OD ring split.
  long N = pp.m;
  if (N <= 0) {
    N = 8192;
    while (N < 32768 && CoeffModulus::MaxBitCount(N, sec_level_type::tc128) < 40 * nprimes) N <<= 1;
  }
  long maxbits = CoeffModulus::MaxBitCount(N, sec_level_type::tc128);
  if (maxbits < 30 * nprimes)
    throw std::runtime_error("depth too large for p=65537 batching ring (<=32768) at 128-bit");

  // Prefer FEWER, LARGER (60-bit) primes: per-level noise headroom (prime size) matters more for the
  // product-tree than the number of levels. Only when 60-bit primes cannot supply the minimum chain
  // (small ring, e.g. 1MM-OD at N=16384) do we shrink the primes to fit `nprimes`. SEAL caps a prime
  // at 60 bits; keeping them at 60 is what makes VESTA/COPSE decrypt at both OD and UD on N=32768.
  int min_primes = depth + 2;                     // minimum chain for the circuit's mod-switches
  int maxprimes60 = (int)(maxbits / 60);
  if (nprimes > maxprimes60 && maxprimes60 >= min_primes) nprimes = maxprimes60;
  int pbits = (int)(maxbits / nprimes); if (pbits > 60) pbits = 60;
  std::vector<int> bits(nprimes, pbits);
  // diagnostic only (default off): IMM_EXTRA_PRIME_BITS=k appends one k-bit prime if it still fits the
  // tc128 bit budget, i.e. one more mod-switch of noise headroom for the deep VESTA/COPSE circuits
  if (const char* e = std::getenv("IMM_EXTRA_PRIME_BITS")) {
    int k = std::atoi(e);
    if (k > 0 && pbits * nprimes + k <= maxbits) { bits.push_back(k); ++nprimes; }
  }

  EncryptionParameters parms(bgv ? scheme_type::bgv : scheme_type::bfv);
  parms.set_poly_modulus_degree(N);
  parms.set_coeff_modulus(CoeffModulus::Create(N, bits));
  parms.set_plain_modulus(p);                                   // 65537: batches for N<=32768
  SEALContext ctx(parms, /*expand_mod_chain=*/true, sec_level_type::tc128);
  N_out = N; totbits_out = 0; for (int b : bits) totbits_out += b; nprimes_out = nprimes;
  return ctx;
}

Backend::Backend(const FheParams& pp, bool bgv, long p) : ctx_(build_ctx(pp, p, bgv, N_, totbits_, nprimes_)), bgv_(bgv) {
  if (!ctx_.parameters_set())
    throw std::runtime_error(std::string("SEAL params invalid: ") + ctx_.parameter_error_message());
  KeyGenerator keygen(ctx_);
  sk_ = keygen.secret_key();
  keygen.create_public_key(pk_);
  keygen.create_relin_keys(rlk_);
  keygen.create_galois_keys(glk_);                             // all power-of-two rotations + columns
  benc_ = std::make_unique<BatchEncoder>(ctx_);
  encr_ = std::make_unique<Encryptor>(ctx_, pk_);
  decr_ = std::make_unique<Decryptor>(ctx_, sk_);
  eval_ = std::make_unique<Evaluator>(ctx_);
  nslots_ = (long)benc_->slot_count();
  sec_ = ctx_.parameters_set() ? 128.0 : -1.0;                 // tc128-validated => >=128-bit
  { auto cd = ctx_.key_context_data();                         // report the real ring/modulus
    N_ = (long)cd->parms().poly_modulus_degree();
    const auto& cm = cd->parms().coeff_modulus();
    nprimes_ = (int)cm.size(); totbits_ = 0;
    for (const auto& q : cm) totbits_ += (long)q.bit_count(); }
}

// ---- encode/encrypt/decrypt ----
std::vector<Ctxt> Backend::encrypt_matrix(const Mat& mat) const {
  std::vector<Ctxt> out; out.reserve(mat.size());
  for (const auto& row : mat) out.push_back(encrypt_vector(row));
  return out;
}
Ctxt Backend::encrypt_vector(const Row& row) const {
  std::vector<int64_t> v(nslots_, 0);
  for (size_t i = 0; i < row.size() && (long)i < nslots_; ++i) v[i] = (int64_t)row[i];
  Plaintext pt; benc_->encode(v, pt);
  Ctxt ct; encr_->encrypt(pt, ct);
  return ct;
}
std::vector<long> Backend::decrypt(const Ctxt& ct, long len) const {
  Plaintext pt; decr_->decrypt(ct, pt);
  std::vector<int64_t> v; benc_->decode(pt, v);
  std::vector<long> out(v.begin(), v.end());
  if (len >= 0 && len < (long)out.size()) out.resize(len);
  return out;
}

Ctxt Backend::eval_rotate(Ctxt ct, long amount) const {
  if (amount == 0) return ct;
  eval_->rotate_rows_inplace(ct, (int)(ROT_SIGN * amount), glk_);
  return ct;
}

// ---- level management (SEAL is NOT auto-leveled) ----
int Backend::level_of(const Ctxt& c) const {
  return (int)ctx_.get_context_data(c.parms_id())->chain_index();
}
void Backend::align(Ctxt& a, Ctxt& b) const {
  int la = level_of(a), lb = level_of(b);
  while (la > lb) { eval_->mod_switch_to_next_inplace(a); --la; }
  while (lb > la) { eval_->mod_switch_to_next_inplace(b); --lb; }
}
Ctxt Backend::mul(const Ctxt& a, const Ctxt& b) const {
  Ctxt x = a, y = b; align(x, y);
  Ctxt r; eval_->multiply(x, y, r);
  eval_->relinearize_inplace(r, rlk_);
  // BGV: mod-switch after each multiply (noise mgmt + drop a level). BFV: no mod-switch -- all
  // ciphertexts stay at the top level, so align() naturally becomes a no-op.
  if (bgv_ && level_of(r) > 0) eval_->mod_switch_to_next_inplace(r);
  return r;
}
void Backend::add_inplace(Ctxt& a, const Ctxt& b) const {
  Ctxt bb = b; align(a, bb);
  eval_->add_inplace(a, bb);
}
void Backend::add_ones(Ctxt& ct, long length) const {
  std::vector<int64_t> v(nslots_, 0);
  for (long i = 0; i < length && i < nslots_; ++i) v[i] = 1;
  Plaintext pt; benc_->encode(v, pt);
  eval_->add_plain_inplace(ct, pt);                            // BGV plaintext add (level-agnostic)
}

// ---- secure comparison helpers (bgv.py::_fhe_eq / _fhe_lt) ----
Ctxt Backend::eq(const Ctxt& a, const Ctxt& b, long length) const {
  Ctxt d; { Ctxt bb = b, aa = a; align(bb, aa); eval_->sub(bb, aa, d); }   // b - a
  Ctxt sq = mul(d, d);                                                     // (b-a)^2  (mod-switched)
  eval_->negate_inplace(sq);                                              // -(b-a)^2
  add_ones(sq, length);                                                    // 1 - (b-a)^2
  return sq;
}
Ctxt Backend::lt(const Ctxt& a, const Ctxt& b, long length) const {
  Ctxt na = a; eval_->negate_inplace(na); add_ones(na, length);           // 1 - a
  return mul(na, b);                                                       // (1-a)*b
}

// ---- bit-sliced comparison trees (verbatim port of bgv.py::sec_comp_{8,16}bit) ----
Ctxt Backend::sec_comp(const std::vector<Ctxt>& a, const std::vector<Ctxt>& b,
                       long length, int bits) const {
  if (bits != 8 && bits != 16) throw std::runtime_error("sec_comp: bits must be 8 or 16");
  std::vector<Ctxt> eqr, ltr; eqr.reserve(bits); ltr.reserve(bits);
  eqr.push_back(lt(a[0], b[0], length));            // eqr[0] placeholder (unused)
  ltr.push_back(lt(a[0], b[0], length));            // ltr[0] = lt(a0,b0)
  for (int i = 1; i < bits; ++i) { eqr.push_back(eq(a[i], b[i], length)); ltr.push_back(lt(a[i], b[i], length)); }

  if (bits == 8) {
    Ctxt e76 = mul(eqr[7], eqr[6]), e5l4 = mul(eqr[5], ltr[4]), e54 = mul(eqr[5], eqr[4]), e7654 = mul(e76, e54);
    Ctxt e3l2 = mul(eqr[3], ltr[2]), e32 = mul(eqr[3], eqr[2]), e32l1 = mul(e32, ltr[1]),
         e1l0 = mul(eqr[1], ltr[0]), e321l0 = mul(e32, e1l0);
    ltr[6] = mul(eqr[7], ltr[6]); ltr[5] = mul(e76, ltr[5]); ltr[4] = mul(e76, e5l4);
    ltr[3] = mul(e7654, ltr[3]); ltr[2] = mul(e7654, e3l2); ltr[1] = mul(e7654, e32l1); ltr[0] = mul(e7654, e321l0);
    Ctxt r = ltr[0]; for (int i = 1; i < 8; ++i) add_inplace(r, ltr[i]); return r;
  }

  Ctxt e1514 = mul(eqr[15], eqr[14]), e13l12 = mul(eqr[13], ltr[12]), e1312 = mul(eqr[13], eqr[12]),
       e15141312 = mul(e1514, e1312);
  Ctxt e11l10 = mul(eqr[11], ltr[10]), e1110 = mul(eqr[11], eqr[10]), e1110l9 = mul(e1110, ltr[9]),
       e9l8 = mul(eqr[9], ltr[8]), e11109l8 = mul(e1110, e9l8), e98 = mul(eqr[9], eqr[8]),
       e111098 = mul(e1110, e98), e15141312111098 = mul(e15141312, e111098);
  Ctxt e7l6 = mul(eqr[7], ltr[6]), e76 = mul(eqr[7], eqr[6]), e76l5 = mul(e76, ltr[5]),
       e5l4 = mul(eqr[5], ltr[4]), e765l4 = mul(e76, e5l4), e54 = mul(eqr[5], eqr[4]), e7654 = mul(e76, e54),
       e7654l3 = mul(e7654, ltr[3]), e3l2 = mul(eqr[3], ltr[2]), e76543l2 = mul(e7654, e3l2),
       e32 = mul(eqr[3], eqr[2]), e32l1 = mul(e32, ltr[1]), e765432l1 = mul(e7654, e32l1),
       e1l0 = mul(eqr[1], ltr[0]), e321l0 = mul(e32, e1l0), e7654321l0 = mul(e7654, e321l0);
  ltr[14] = mul(eqr[15], ltr[14]); ltr[13] = mul(e1514, ltr[13]); ltr[12] = mul(e1514, e13l12);
  ltr[11] = mul(e15141312, ltr[11]); ltr[10] = mul(e15141312, e11l10); ltr[9] = mul(e15141312, e1110l9);
  ltr[8] = mul(e15141312, e11109l8); ltr[7] = mul(e15141312111098, ltr[7]); ltr[6] = mul(e15141312111098, e7l6);
  ltr[5] = mul(e15141312111098, e76l5); ltr[4] = mul(e15141312111098, e765l4); ltr[3] = mul(e15141312111098, e7654l3);
  ltr[2] = mul(e15141312111098, e76543l2); ltr[1] = mul(e15141312111098, e765432l1); ltr[0] = mul(e15141312111098, e7654321l0);
  Ctxt r = ltr[0]; for (int i = 1; i < 16; ++i) add_inplace(r, ltr[i]); return r;
}

// ---- diagonal ctxt x ctxt matmul (bgv.py::mat_mul, OpenFHE-EvalRotate semantics) ----
int Backend::noise_budget(const Ctxt& c) const { return decr_->invariant_noise_budget(c); }

int Backend::replication_rounds(long rows, long colnum) const {
  // slot i + j (i < rows, j < colnum) must hold v[(i + j) mod colnum]; mat_mul builds >= 2 copies
  long copies = std::max<long>(2, (rows + 2 * colnum - 2) / colnum);
  if (copies * colnum > nslots_ / 2)
    throw std::runtime_error("MatMul needs " + std::to_string(copies) + " x " + std::to_string(colnum) +
                             " slots per rotation row but N/2 = " + std::to_string(nslots_ / 2));
  return (int)(copies - 1);
}

Ctxt Backend::mat_mul(const std::vector<Ctxt>& M, Ctxt v, long colnum, int recursive_time) const {
  // sequential rotate-by-1 -- identical algorithm to OpenFHE bgv.py::mat_mul, so cross-backend speedup
  // ratios are apples-to-apples. rotate_rows is cyclic within a row of N/2 slots: callers pass
  // recursive_time = replication_rounds(rows, colnum) so the replicated vector never wraps (the old
  // fixed recursive_time = 2, i.e. 3 copies, wrapped for MNIST's 2q = 3770 at N = 16384).
  Ctxt mv = eval_rotate(v, -colnum);        // OpenFHE EvalRotate(v, -colnum)
  Ctxt ev = v; add_inplace(ev, mv);
  for (int i = 0; i < recursive_time - 1; ++i) { mv = eval_rotate(mv, -colnum); add_inplace(ev, mv); }
  long rownum = (long)M.size();
  Ctxt r = mul(M[0], ev);
  for (long i = 1; i < rownum; ++i) { ev = eval_rotate(ev, 1); Ctxt t = mul(M[i], ev); add_inplace(r, t); }
  return r;
}

// ---- balanced product (operations.py::fhe_reduction) ----
Ctxt Backend::reduction(std::vector<Ctxt> cts) const {
  long n = (long)cts.size();
  if (n == 1) return cts[0];
  std::vector<long> ops; int nbits = 0; { long t = n - 1; while (t) { ++nbits; t >>= 1; } }
  for (int i = nbits - 1; i >= 0; --i) ops.push_back(1L << i);
  for (long op : ops)
    for (long i = 0; i < op; ++i) { if (i + op >= n) break; cts[i] = mul(cts[i], cts[i + op]); }
  return cts[0];
}

}  // namespace imm
