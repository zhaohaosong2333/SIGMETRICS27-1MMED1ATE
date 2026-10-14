// Core SEAL-backend gate + isolation of the VESTA/COPSE path (reduction product tree, mask add at
// depth, product of deep operands). 1MM already validates (golden MATCH); this pinpoints the rest.
#include "immed1ate/imm_core.h"
#include <iostream>
using namespace imm;

int main() {
  FheParams pp{0, 0, 0, 9};                    // depth 9 (VESTA/COPSE regime), auto ring
  Backend be(pp);
  std::cout << "N=" << be.ring() << " slots=" << be.nslots()
            << " bits=" << be.totbits() << " primes=" << be.nprimes() << " sec=" << be.security() << "\n";
  auto enc = [&](std::initializer_list<long> vals) {
    Row r(be.nslots(), 0); int i = 0; for (long x : vals) r[i++] = x; return be.encrypt_vector(r);
  };
  int fails = 0;
  auto chk = [&](const char* name, const std::vector<long>& got, std::initializer_list<long> exp) {
    int i = 0; bool ok = true; for (long e : exp) if (got[i++] != e) ok = false;
    std::cout << "  " << name << " = ["; for (size_t k = 0; k < exp.size(); ++k) std::cout << (k? " ":"") << got[k];
    std::cout << "]  " << (ok ? "OK" : "FAIL") << "\n"; if (!ok) ++fails;
  };

  // 0) roundtrip + rotation sign
  { auto ct = enc({1,2,3,4,5,6}); auto d = be.decrypt(be.eval_rotate(ct,2),6);
    chk("rotate2", d, {3,4,5,6,0,0}); }
  // 1) reduction logic (4-way product)
  { auto a=enc({2,3}), b=enc({3,2}), c=enc({1,4}), d=enc({5,1});
    chk("reduce4", be.decrypt(be.reduction({a,b,c,d}),2), {30,24}); }        // 2*3*1*5 ; 3*2*4*1
  // 2) fresh mask added onto a DEEP ciphertext (VESTA level+mask)
  { auto x=enc({7,7}); auto one=enc({1,1});
    for (int k=0;k<5;++k) x = be.reduction({x, one});                        // 5 mults deep, value stays 7
    auto m=enc({2,3}); be.add_inplace(x, m);
    chk("deep+mask", be.decrypt(x,2), {9,10}); }                            // 7+2 ; 7+3
  // 3) product of two DEEP operands (reduction over deepened level-results)
  { auto a=enc({2,2}), b=enc({3,3}), one=enc({1,1});
    for (int k=0;k<3;++k){ a=be.reduction({a,one}); b=be.reduction({b,one}); }
    chk("deep_reduce", be.decrypt(be.reduction({a,b}),1), {6}); }            // 2*3

  std::cout << (fails==0 ? "CORE OK" : "CORE FAIL") << " (" << fails << " failing)\n";
  return fails==0 ? 0 : 1;
}
