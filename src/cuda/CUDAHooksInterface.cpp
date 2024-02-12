#include "cuda/CUDAHooksInterface.h"
#include "cuda/CUDAHooks.h"
#include "utils/CallOnce.h"

namespace c10::cuda {
// from pytorch:
// NB: We purposely leak the CUDA hooks object.  This is because under some
// situations, we may need to reference the CUDA hooks while running destructors
// of objects which were constructed *prior* to the first invocation of
// getCUDAHooks.  The example which precipitated this change was the fused
// kernel cache in the JIT.  The kernel cache is a global variable which caches
// both CPU and CUDA kernels; CUDA kernels must interact with CUDA hooks on
// destruction.  Because the kernel cache handles CPU kernels too, it can be
// constructed before we initialize CUDA; if it contains CUDA kernels at program
// destruction time, you will destruct the CUDA kernels after CUDA hooks has
// been unloaded.  In principle, we could have also fixed the kernel cache store
// CUDA kernels in a separate global variable, but this solution is much
// simpler.
//
// CUDAHooks doesn't actually contain any data, so leaking it is very benign;
// you're probably losing only a word (the vptr in the allocated object.)
static CUDAHooksInterface* cuda_hooks = nullptr;

const CUDAHooksInterface& getCUDAHooks() {
  static c10::OnceFlag once;
  c10::callOnce(once, []{
    // TODO: make this configurable instead of hardcoding
    cuda_hooks = new CUDAHooks();
  });
  return *cuda_hooks;
}
}
