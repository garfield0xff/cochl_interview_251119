// Dynamic loader for libcochl_api.so
// Handles dlopen, dlsym for all CochlApi C functions

#pragma once

#include <string>

namespace cochl {
namespace api {

class CochlApi {
 public:
  CochlApi();
  ~CochlApi();

  // Load library and all function symbols
  bool load(const std::string& library_path);

  // Check if library is loaded
  bool isLoaded() const { return lib_handle_ != nullptr; }

  // Function pointers (public for direct access)
  void* (*create)(const char*);
  int (*runInference)(void*, const float*, const long long*, size_t, float*);
  size_t (*getInputSize)(void*);
  size_t (*getOutputSize)(void*);
  void (*destroy)(void*);
  int (*loadImage)(const char*, float*, size_t);
  void* (*loadClassNames)(const char*);
  const char* (*getClassName)(void*, int);
  void (*destroyClassMap)(void*);

 private:
  void* lib_handle_;

  // Helper to load a single symbol
  template<typename FuncPtr>
  bool loadSymbol(FuncPtr& func_ptr, const char* symbol_name);
};

}  // namespace api
}  // namespace cochl
