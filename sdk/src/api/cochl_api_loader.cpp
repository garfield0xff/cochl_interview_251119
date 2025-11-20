#include "api/cochl_api_loader.h"
#include "error/sdk_error.h"

#include <dlfcn.h>
#include <glog/logging.h>

namespace cochl {
namespace api {

CochlApiLoader::CochlApiLoader()
    : lib_handle_(nullptr),
      create(nullptr),
      runInference(nullptr),
      getInputSize(nullptr),
      getOutputSize(nullptr),
      destroy(nullptr),
      loadImage(nullptr),
      loadClassNames(nullptr),
      getClassName(nullptr),
      destroyClassMap(nullptr) {}

CochlApiLoader::~CochlApiLoader() {
  if (lib_handle_) {
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
  }
}

template<typename FuncPtr>
bool CochlApiLoader::loadSymbol(FuncPtr& func_ptr, const char* symbol_name) {
  func_ptr = reinterpret_cast<FuncPtr>(dlsym(lib_handle_, symbol_name));
  if (!func_ptr) {
    error::printError(error::SdkError::LIBRARY_SYMBOL_NOT_FOUND, symbol_name);
    return false;
  }
  return true;
}

bool CochlApiLoader::load(const std::string& library_path) {
  if (lib_handle_) {
    error::printError(error::SdkError::LIBRARY_ALREADY_LOADED);
    return false;
  }

  // Load library with RTLD_LAZY | RTLD_LOCAL
  lib_handle_ = dlopen(library_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!lib_handle_) {
    error::printError(error::SdkError::LIBRARY_LOAD_FAILED, library_path);
    LOG(ERROR) << "dlerror: " << dlerror();
    return false;
  }

  // Clear any existing error
  dlerror();

  // Load all function symbols
  bool success = true;
  success &= loadSymbol(create, "CochlApi_Create");
  success &= loadSymbol(runInference, "CochlApi_RunInference");
  success &= loadSymbol(getInputSize, "CochlApi_GetInputSize");
  success &= loadSymbol(getOutputSize, "CochlApi_GetOutputSize");
  success &= loadSymbol(destroy, "CochlApi_Destroy");
  success &= loadSymbol(loadImage, "CochlApi_LoadImage");
  success &= loadSymbol(loadClassNames, "CochlApi_LoadClassNames");
  success &= loadSymbol(getClassName, "CochlApi_GetClassName");
  success &= loadSymbol(destroyClassMap, "CochlApi_DestroyClassMap");

  if (!success) {
    dlclose(lib_handle_);
    lib_handle_ = nullptr;
    return false;
  }

  LOG(INFO) << "[CochlApiLoader] Library loaded successfully: " << library_path;
  return true;
}

}  // namespace api
}  // namespace cochl
