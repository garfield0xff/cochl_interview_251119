#include "python_hook.h"
#include <pybind11/embed.h>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

namespace tvm_sdk {

bool PythonHook::is_initialized_ = false;
std::string PythonHook::python_path_ = "";

void PythonHook::initialize() {
    if (!is_initialized_) {
        if (!Py_IsInitialized()) {
            py::initialize_interpreter();
        }
        is_initialized_ = true;

        // Priority: Environment variable > Compile-time definition
        const char* env_path = std::getenv("TVM_SDK_PYTHON_PATH");
        if (env_path != nullptr) {
            python_path_ = env_path;
        }
#ifdef TVM_SDK_PYTHON_PATH
        else {
            python_path_ = TVM_SDK_PYTHON_PATH;
        }
#endif

        if (!python_path_.empty()) {
            add_python_path(python_path_);
        }
    }
}

void PythonHook::finalize() {
    if (is_initialized_) {
        if (Py_IsInitialized()) {
            py::finalize_interpreter();
        }
        is_initialized_ = false;
    }
}

bool PythonHook::is_initialized() {
    return is_initialized_;
}

void PythonHook::add_python_path(const std::string& path) {
    initialize();

    try {
        py::gil_scoped_acquire gil;

        py::module_ sys = py::module_::import("sys");
        py::list sys_path = sys.attr("path");

        // Check if path already exists
        bool path_exists = false;
        for (auto item : sys_path) {
            if (py::cast<std::string>(item) == path) {
                path_exists = true;
                break;
            }
        }

        if (!path_exists) {
            sys_path.insert(0, path);
        }

    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            std::string("Failed to add Python path: ") + e.what()
        );
    }
}

py::object PythonHook::import_module(const std::string& module_name) {
    initialize();

    try {
        py::gil_scoped_acquire gil;
        return py::module_::import(module_name.c_str());

    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            std::string("Failed to import module '") + module_name + "': " + e.what()
        );
    }
}

py::object PythonHook::get_module_attr(
    const std::string& module_name,
    const std::string& attr_name
) {
    initialize();

    try {
        py::gil_scoped_acquire gil;

        py::object module = import_module(module_name);
        return module.attr(attr_name.c_str());

    } catch (const py::error_already_set& e) {
        throw std::runtime_error(
            std::string("Failed to get attribute '") + attr_name + "' from module '" + module_name + "': " + e.what()
        );
    }
}

} // namespace tvm_sdk
