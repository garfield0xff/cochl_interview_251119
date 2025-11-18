#ifndef TVM_PYTHON_HOOK_H
#define TVM_PYTHON_HOOK_H

#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

namespace tvm_sdk {

/**
 * @brief Python Runtime and Object Management
 */
class PythonHook {
public:
    /**
     * @brief Initialize Python interpreter
     *
     * Uses TVM_SDK_PYTHON_PATH environment variable to locate Python scripts.
     */
    static void initialize();

    /**
     * @brief Finalize Python interpreter
     */
    static void finalize();

    /**
     * @brief Check if Python is initialized
     */
    static bool is_initialized();

    /**
     * @brief Add a path to python script to sys.path
     * @param path Path to add
     */
    static void add_python_path(const std::string& path);

    /**
     * @brief Import a Python module
     * @param module_name Module name (e.g., "numpy_samples")
     * @return Python module object
     */
    static py::object import_module(const std::string& module_name);

    /**
     * @brief Call a Python function with arguments
     * @param module_name Module name
     * @param function_name Function name
     * @param args Function arguments
     * @return Python object result
     */
    template<typename... Args>
    static py::object call_function(
        const std::string& module_name,
        const std::string& function_name,
        Args&&... args
    ) {
        initialize();
        py::gil_scoped_acquire gil;

        try {
            py::object module = import_module(module_name);
            py::object func = module.attr(function_name.c_str());
            return func(std::forward<Args>(args)...);
        } catch (const py::error_already_set& e) {
            throw std::runtime_error(
                std::string("Failed to call ") + module_name + "." + function_name + ": " + e.what()
            );
        }
    }

    /**
     * @brief Get attribute from Python module
     * @param module_name Module name
     * @param attr_name Attribute name
     * @return Python object
     */
    static py::object get_module_attr(
        const std::string& module_name,
        const std::string& attr_name
    );

    /**
     * @brief Convert Python object to C++ type
     * @tparam T Target C++ type
     * @param obj Python object
     * @return Converted C++ value
     */
    template<typename T>
    static T to_cpp(const py::object& obj) {
        return py::cast<T>(obj);
    }

private:
    static bool is_initialized_;
    static std::string python_path_;
};

} // namespace tvm_sdk

#endif // TVM_PYTHON_HOOK_H
