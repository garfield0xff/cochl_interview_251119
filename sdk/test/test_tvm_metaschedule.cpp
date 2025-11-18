#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "python_hook.h"
#include <string>
#include <map>
#include <iostream>
#include <iomanip>

using namespace tvm_sdk;
using ::testing::Not;
using ::testing::IsEmpty;
using ::testing::HasSubstr;

// Test fixture for TVM Schedule tests
class TVMScheduleTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize Python
        PythonHook::initialize();
    }

    void TearDown() override {
        // Cleanup if needed
    }

    void print_map(const std::map<std::string, std::string>& map_data, const std::string& prefix = "  ") {
        for (const auto& pair : map_data) {
            std::cout << prefix << std::setw(30) << std::left << pair.first << ": " << pair.second << "\n";
        }
    }

    void print_separator(const std::string& title = "") {
        std::cout << "\n" << std::string(70, '=') << "\n";
        if (!title.empty()) {
            std::cout << title << "\n";
            std::cout << std::string(70, '=') << "\n";
        }
    }
};

// Test: Create ResNet18 Relax IR
TEST_F(TVMScheduleTest, CreateResNet18RelaxIR) {
    print_separator("Test: Create ResNet18 Relax IR");

    try {
        py::object result = PythonHook::call_function(
            "tvm_ext",
            "create_resnet18_relax_ir",
            true,  // pretrained
            false  // keep_params
        );

        py::dict dict_result = py::cast<py::dict>(result);
        std::string status = py::cast<std::string>(dict_result["status"]);

        EXPECT_EQ(status, "success");

        if (status == "success") {
            std::map<std::string, std::string> ir_info;
            for (auto item : dict_result) {
                std::string key = py::cast<std::string>(item.first);
                if (key != "relax_mod") {  // Skip the large IR string
                    ir_info[key] = py::cast<std::string>(item.second);
                }
            }

            std::cout << "\nRelax IR Info:\n";
            print_map(ir_info);

            // Get IR string length
            std::string relax_mod = py::cast<std::string>(dict_result["relax_mod"]);
            std::cout << "  " << std::setw(30) << std::left << "relax_mod_length"
                      << ": " << relax_mod.length() << " bytes\n";

            EXPECT_THAT(relax_mod, Not(IsEmpty()));
            EXPECT_EQ(ir_info["input_shape"], "(1, 3, 224, 224)");
            EXPECT_EQ(ir_info["dtype"], "float32");
        } else {
            std::string error_msg = py::cast<std::string>(dict_result["error_message"]);
            FAIL() << "Failed to create Relax IR: " << error_msg;
        }

    } catch (const std::exception& e) {
        FAIL() << "Exception: " << e.what();
    }
}

// Test: Tune ResNet18 with MetaSchedule (Quick test with minimal trials)
TEST_F(TVMScheduleTest, TuneResNet18WithMetaSchedule) {
    print_separator("Test: Tune ResNet18 with MetaSchedule");

    // Path to dog.jpeg using project root
    std::string project_root = PROJECT_SOURCE_DIR;
    std::string image_path = project_root + "/test/dog.jpeg";

    std::cout << "\nTest Configuration:\n";
    std::cout << "  Image path: " << image_path << "\n";
    std::cout << "  Auto-tuning: enabled\n";
    std::cout << "  Num trials: 200 (minimal for testing)\n";
    std::cout << "  Opt level: 3\n";
    std::cout << "  Max workers: 10\n";
    std::cout << "  Work dir: tuning_database_test\n\n";

    std::cout << "Starting MetaSchedule tuning (this will take a few minutes)...\n";

    try {
        py::object result = PythonHook::call_function(
            "tvm_ext",
            "tune_resnet18_with_metaschedule",
            image_path,
            true,                      // use_auto_tuning
            200,                         // num_trials (minimal for quick test)
            3,                         // opt_level
            10,                         // max_workers
            "tuning_database_test"     // work_dir
        );

        py::dict dict_result = py::cast<py::dict>(result);
        std::string status = py::cast<std::string>(dict_result["status"]);

        EXPECT_EQ(status, "success");

        if (status == "success") {
            std::cout << "\n✓ Tuning completed successfully!\n";

            // Extract and print results
            std::map<std::string, std::string> tune_results;
            for (auto item : dict_result) {
                std::string key = py::cast<std::string>(item.first);
                if (key != "traceback") {
                    tune_results[key] = py::cast<std::string>(item.second);
                }
            }

            print_separator("Tuning Results");
            print_map(tune_results);

            // Validate key metrics
            EXPECT_THAT(tune_results["avg_inference_time_ms"], Not(IsEmpty()));
            EXPECT_THAT(tune_results["top1_class"], Not(IsEmpty()));

            double top1_prob = std::stod(tune_results["top1_probability"]);
            EXPECT_GT(top1_prob, 0.0);
            EXPECT_LE(top1_prob, 1.0);

            std::cout << "\n=== Inference Summary ===\n";
            std::cout << "  Top-1 Prediction: " << tune_results["top1_class"] << "\n";
            std::cout << "  Confidence: " << std::fixed << std::setprecision(2)
                      << (top1_prob * 100) << "%\n";
            std::cout << "  Avg Inference Time: " << tune_results["avg_inference_time_ms"] << " ms\n";
            std::cout << "  Iterations: " << tune_results["num_iterations"] << "\n";

        } else {
            std::string error_msg = py::cast<std::string>(dict_result["error_message"]);
            std::cout << "\nError Message: " << error_msg << "\n";

            if (dict_result.contains("traceback")) {
                std::string traceback = py::cast<std::string>(dict_result["traceback"]);
                std::cout << "\nTraceback:\n" << traceback << "\n";
            }

            FAIL() << "Tuning failed: " << error_msg;
        }

    } catch (const std::exception& e) {
        FAIL() << "Exception during tuning: " << e.what();
    }
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
