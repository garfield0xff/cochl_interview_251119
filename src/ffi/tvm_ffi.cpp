#include "ffi/tvm_ffi.h"
#include "python_hook.h"
#include <stdexcept>

namespace tvm_sdk {
namespace ffi {

std::string TVMFFI::get_tvm_version() {
    py::object result = PythonHook::call_function(MODULE_PATH, "get_tvm_version");
    return PythonHook::to_cpp<std::string>(result);
}

std::map<std::string, std::string> TVMFFI::get_tvm_target(const std::string& target_name) {
    py::object result = PythonHook::call_function(MODULE_PATH, "get_tvm_target", target_name);
    py::dict dict_result = py::cast<py::dict>(result);

    std::map<std::string, std::string> target_info;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);

        // Handle both string and list values
        if (key == "keys") {
            py::list keys_list = py::cast<py::list>(item.second);
            std::string keys_str = "[";
            for (size_t i = 0; i < py::len(keys_list); i++) {
                if (i > 0) keys_str += ", ";
                keys_str += py::cast<std::string>(keys_list[i]);
            }
            keys_str += "]";
            target_info[key] = keys_str;
        } else {
            target_info[key] = py::cast<std::string>(item.second);
        }
    }

    return target_info;
}

std::map<std::string, bool> TVMFFI::check_tvm_modules() {
    py::object result = PythonHook::call_function(MODULE_PATH, "check_tvm_modules");
    py::dict dict_result = py::cast<py::dict>(result);

    std::map<std::string, bool> modules;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);
        bool value = py::cast<bool>(item.second);
        modules[key] = value;
    }

    return modules;
}

std::map<std::string, std::string> TVMFFI::get_metaschedule_config() {
    py::object result = PythonHook::call_function(MODULE_PATH, "get_metaschedule_config");
    py::dict dict_result = py::cast<py::dict>(result);

    std::map<std::string, std::string> config;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);

        // Handle different types
        if (py::isinstance<py::str>(item.second)) {
            config[key] = py::cast<std::string>(item.second);
        } else if (py::isinstance<py::int_>(item.second)) {
            config[key] = std::to_string(py::cast<int>(item.second));
        } else if (py::isinstance<py::bool_>(item.second)) {
            config[key] = py::cast<bool>(item.second) ? "true" : "false";
        } else if (py::isinstance<py::list>(item.second)) {
            py::list list_val = py::cast<py::list>(item.second);
            std::string list_str = "[";
            for (size_t i = 0; i < py::len(list_val); i++) {
                if (i > 0) list_str += ", ";
                list_str += py::cast<std::string>(py::str(list_val[i]));
            }
            list_str += "]";
            config[key] = list_str;
        } else {
            config[key] = py::cast<std::string>(py::str(item.second));
        }
    }

    return config;
}

std::map<std::string, std::string> TVMFFI::check_tuning_database(const std::string& work_dir) {
    py::object result = PythonHook::call_function(MODULE_PATH, "check_tuning_database", work_dir);
    py::dict dict_result = py::cast<py::dict>(result);

    std::map<std::string, std::string> info;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);

        if (py::isinstance<py::str>(item.second)) {
            info[key] = py::cast<std::string>(item.second);
        } else if (py::isinstance<py::int_>(item.second)) {
            info[key] = std::to_string(py::cast<int>(item.second));
        } else if (py::isinstance<py::bool_>(item.second)) {
            info[key] = py::cast<bool>(item.second) ? "true" : "false";
        } else if (py::isinstance<py::list>(item.second)) {
            py::list list_val = py::cast<py::list>(item.second);
            std::string list_str = "[";
            for (size_t i = 0; i < py::len(list_val); i++) {
                if (i > 0) list_str += ", ";
                list_str += py::cast<std::string>(py::str(list_val[i]));
            }
            list_str += "]";
            info[key] = list_str;
        } else {
            info[key] = py::cast<std::string>(py::str(item.second));
        }
    }

    return info;
}

std::map<std::string, std::string> TVMFFI::compile_with_metaschedule(
    const std::string& relax_mod_ir,
    const std::string& target_name,
    bool use_auto_tuning,
    int num_trials,
    int max_workers,
    const std::string& work_dir,
    int opt_level
) {
    py::object max_workers_obj = max_workers > 0 ? py::cast(max_workers) : py::cast<py::none>(Py_None);

    py::object result = PythonHook::call_function(
        MODULE_PATH,
        "compile_with_metaschedule",
        relax_mod_ir,
        target_name,
        use_auto_tuning,
        num_trials,
        max_workers_obj,
        work_dir,
        opt_level
    );

    py::dict dict_result = py::cast<py::dict>(result);

    std::map<std::string, std::string> compile_info;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);

        if (py::isinstance<py::str>(item.second)) {
            compile_info[key] = py::cast<std::string>(item.second);
        } else if (py::isinstance<py::int_>(item.second)) {
            compile_info[key] = std::to_string(py::cast<int>(item.second));
        } else if (py::isinstance<py::bool_>(item.second)) {
            compile_info[key] = py::cast<bool>(item.second) ? "true" : "false";
        } else {
            compile_info[key] = py::cast<std::string>(py::str(item.second));
        }
    }

    return compile_info;
}

std::map<std::string, std::string> TVMFFI::tune_with_metaschedule(
    const std::string& relax_mod_ir,
    const std::string& target_name,
    int num_trials,
    int max_workers,
    const std::string& work_dir
) {
    py::object max_workers_obj = max_workers > 0 ? py::cast(max_workers) : py::cast<py::none>(Py_None);

    py::object result = PythonHook::call_function(
        MODULE_PATH,
        "tune_with_metaschedule",
        relax_mod_ir,
        target_name,
        num_trials,
        max_workers_obj,
        work_dir
    );

    py::dict dict_result = py::cast<py::dict>(result);

    std::map<std::string, std::string> tune_info;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);

        if (py::isinstance<py::str>(item.second)) {
            tune_info[key] = py::cast<std::string>(item.second);
        } else if (py::isinstance<py::int_>(item.second)) {
            tune_info[key] = std::to_string(py::cast<int>(item.second));
        } else if (py::isinstance<py::bool_>(item.second)) {
            tune_info[key] = py::cast<bool>(item.second) ? "true" : "false";
        } else {
            tune_info[key] = py::cast<std::string>(py::str(item.second));
        }
    }

    return tune_info;
}

std::map<std::string, std::string> TVMFFI::apply_tuning_database(
    const std::string& relax_mod_ir,
    const std::string& target_name,
    const std::string& work_dir,
    int opt_level
) {
    py::object result = PythonHook::call_function(
        MODULE_PATH,
        "apply_tuning_database",
        relax_mod_ir,
        target_name,
        work_dir,
        opt_level
    );

    py::dict dict_result = py::cast<py::dict>(result);

    std::map<std::string, std::string> build_info;
    for (auto item : dict_result) {
        std::string key = py::cast<std::string>(item.first);

        if (py::isinstance<py::str>(item.second)) {
            build_info[key] = py::cast<std::string>(item.second);
        } else if (py::isinstance<py::int_>(item.second)) {
            build_info[key] = std::to_string(py::cast<int>(item.second));
        } else if (py::isinstance<py::bool_>(item.second)) {
            build_info[key] = py::cast<bool>(item.second) ? "true" : "false";
        } else {
            build_info[key] = py::cast<std::string>(py::str(item.second));
        }
    }

    return build_info;
}

} // namespace ffi
} // namespace tvm_sdk
