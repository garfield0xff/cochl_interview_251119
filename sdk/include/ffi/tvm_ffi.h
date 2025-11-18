#ifndef TVM_FFI_H
#define TVM_FFI_H

#include "python_hook.h"
#include <string>
#include <map>
#include <vector>

namespace tvm_sdk {
namespace ffi {

/**
 * @brief TVM FFI (Foreign Function Interface)
 *
 * Provides high-level interface to call TVM functions from C++
 */
class TVMFFI {
public:
    /**
     * @brief Get TVM version
     * @return TVM version string
     */
    static std::string get_tvm_version();

    /**
     * @brief Get TVM target information
     * @param target_name Target name (e.g., "llvm", "cuda")
     * @return Target information as map
     */
    static std::map<std::string, std::string> get_tvm_target(const std::string& target_name = "llvm");

    /**
     * @brief Check available TVM modules
     * @return Module availability as map (module_name -> bool)
     */
    static std::map<std::string, bool> check_tvm_modules();

    /**
     * @brief Get MetaSchedule configuration
     * @return MetaSchedule configuration as map
     */
    static std::map<std::string, std::string> get_metaschedule_config();

    /**
     * @brief Check if tuning database exists
     * @param work_dir Directory to check
     * @return Database information as map
     */
    static std::map<std::string, std::string> check_tuning_database(const std::string& work_dir = "tuning_database");

    /**
     * @brief Compile with MetaSchedule tuning
     * @param relax_mod_ir Relax module IR string
     * @param target_name Target name (e.g., "llvm", "cuda")
     * @param use_auto_tuning Enable MetaSchedule auto-tuning
     * @param num_trials Maximum number of trials
     * @param max_workers Number of parallel workers (0 = auto)
     * @param work_dir Directory for tuning database
     * @param opt_level Optimization level (0-3)
     * @return Compilation results as map
     */
    static std::map<std::string, std::string> compile_with_metaschedule(
        const std::string& relax_mod_ir,
        const std::string& target_name = "llvm",
        bool use_auto_tuning = true,
        int num_trials = 64,
        int max_workers = 0,
        const std::string& work_dir = "tuning_database",
        int opt_level = 0
    );

    /**
     * @brief Tune with MetaSchedule only (no compilation)
     * @param relax_mod_ir Relax module IR string
     * @param target_name Target name
     * @param num_trials Maximum number of trials
     * @param max_workers Number of parallel workers (0 = auto)
     * @param work_dir Directory for tuning database
     * @return Tuning results as map
     */
    static std::map<std::string, std::string> tune_with_metaschedule(
        const std::string& relax_mod_ir,
        const std::string& target_name = "llvm",
        int num_trials = 64,
        int max_workers = 0,
        const std::string& work_dir = "tuning_database"
    );

    /**
     * @brief Apply tuning database and build
     * @param relax_mod_ir Relax module IR string
     * @param target_name Target name
     * @param work_dir Directory containing tuning database
     * @param opt_level Optimization level (0-3)
     * @return Build results as map
     */
    static std::map<std::string, std::string> apply_tuning_database(
        const std::string& relax_mod_ir,
        const std::string& target_name = "llvm",
        const std::string& work_dir = "tuning_database",
        int opt_level = 0
    );

private:
    static constexpr const char* MODULE_PATH = "tvm_ext";
};

} // namespace ffi
} // namespace tvm_sdk

#endif // TVM_FFI_H
