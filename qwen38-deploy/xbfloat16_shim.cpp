// xbfloat16 ABI shim for kunlun_ops 0.1.122 on older torch_xmlir
// Provides the two symbols that libapiinfer.so needs but the current
// libxpuapi.so does not export:
//   _ZN9xbfloat16C1Ef   (xbfloat16::xbfloat16(float)  C1 complete ctor)
//   _ZNK9xbfloat16cvfEv (xbfloat16::operator float() const)
//
// xbfloat16 is the standard 16-bit bfloat16 bit pattern (uint16_t).
// C1 is forwarded to libxpuapi's C2 ctor when it is visible, with a
// truncating fallback implementation otherwise.
#include <cstdint>
#include <cstring>
#include <dlfcn.h>

namespace {

typedef void (*c2_fn)(void*, float);

c2_fn resolve_c2() {
    // RTLD_DEFAULT lookup across already-loaded libs (and this shim itself
    // must NOT define C2, so there is no self-recursion).
    void* p = dlsym(RTLD_DEFAULT, "_ZN9xbfloat16C2Ef");
    return reinterpret_cast<c2_fn>(p);
}

}  // namespace

extern "C" {

// _ZN9xbfloat16C1Ef : void xbfloat16(void* this, float v)
__attribute__((visibility("default")))
void xbfloat16_c1(void* self, float v) __asm__("_ZN9xbfloat16C1Ef");

__attribute__((visibility("default")))
void xbfloat16_c1(void* self, float v) {
    static c2_fn c2 = resolve_c2();
    if (c2) {
        c2(self, v);
    } else {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        *reinterpret_cast<uint16_t*>(self) =
            static_cast<uint16_t>(bits >> 16);  // truncate toward zero
    }
}

// _ZNK9xbfloat16cvfEv : float xbfloat16(void* this) const
__attribute__((visibility("default")))
float xbfloat16_cvt(const void* self) __asm__("_ZNK9xbfloat16cvfEv");

__attribute__((visibility("default")))
float xbfloat16_cvt(const void* self) {
    uint16_t x;
    std::memcpy(&x, self, 2);
    uint32_t bits = static_cast<uint32_t>(x) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

}  // extern "C"
