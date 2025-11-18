"""
TVM MetaSchedule Functions for C++ Integration
"""

import os
import tvm
from tvm import relax
import multiprocessing


def tune_with_metaschedule(
    relax_mod_ir,
    target_name="llvm",
    num_trials=64,
    max_workers=None,
    work_dir="tuning_database"
):
    """
    Tune TVM Relax module using MetaSchedule.

    Args:
        relax_mod_ir: Relax module IR string
        target_name: Target name (e.g., "llvm", "cuda")
        num_trials: Maximum number of trials for tuning
        max_workers: Number of parallel workers (default: CPU count)
        work_dir: Directory to store tuning database

    Returns:
        dict: Tuning results with status and work_dir
    """
    import tvm.meta_schedule as ms
    from tvm.meta_schedule.builder import LocalBuilder
    from tvm.ir.transform import PassContext

    try:
        # Parse IR string to module
        relax_mod = tvm.ir.load_json(relax_mod_ir)

        # Setup target
        num_cores = multiprocessing.cpu_count()
        if "llvm" in target_name:
            target = tvm.target.Target(f"llvm -num-cores {num_cores}")
        else:
            target = tvm.target.Target(target_name)

        # Apply zero pipeline
        with target:
            relax_mod = relax.get_pipeline("zero")(relax_mod)

        # Setup work directory
        os.makedirs(work_dir, exist_ok=True)

        # Setup max workers
        if max_workers is None:
            max_workers = num_cores

        # Create builder
        builder = LocalBuilder(max_workers=max_workers)

        # Run MetaSchedule tuning
        with target:
            ms.tune_tir(
                mod=relax_mod,
                target=target,
                work_dir=work_dir,
                max_trials_global=num_trials,
                builder=builder,
                num_tuning_cores=max_workers,
            )

        return {
            "status": "success",
            "work_dir": work_dir,
            "num_trials": num_trials,
            "max_workers": max_workers,
            "target": str(target)
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def apply_tuning_database(
    relax_mod_ir,
    target_name="llvm",
    work_dir="tuning_database",
    opt_level=0
):
    """
    Apply tuning database to Relax module and build.

    Args:
        relax_mod_ir: Relax module IR string
        target_name: Target name (e.g., "llvm", "cuda")
        work_dir: Directory containing tuning database
        opt_level: Optimization level (0-3)

    Returns:
        dict: Build results with compiled module path
    """
    from tvm.ir.transform import PassContext

    try:
        # Parse IR string to module
        relax_mod = tvm.ir.load_json(relax_mod_ir)

        # Setup target
        num_cores = multiprocessing.cpu_count()
        if "llvm" in target_name:
            target = tvm.target.Target(f"llvm -num-cores {num_cores}")
        else:
            target = tvm.target.Target(target_name)

        # Apply tuning database
        with target, PassContext(opt_level=opt_level):
            application_pass = relax.transform.MetaScheduleApplyDatabase(work_dir)
            relax_mod = application_pass(relax_mod)

        # Build the module
        ex = relax.build(relax_mod, target)

        # Save the built module
        lib_path = os.path.join(work_dir, "compiled_lib.so")
        ex.export_library(lib_path)

        return {
            "status": "success",
            "lib_path": lib_path,
            "work_dir": work_dir,
            "opt_level": opt_level,
            "target": str(target)
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def compile_with_metaschedule(
    relax_mod_ir,
    target_name="llvm",
    use_auto_tuning=True,
    num_trials=64,
    max_workers=None,
    work_dir="tuning_database",
    opt_level=0
):
    """
    Complete compilation pipeline with optional MetaSchedule tuning.

    Args:
        relax_mod_ir: Relax module IR string
        target_name: Target name (e.g., "llvm", "cuda")
        use_auto_tuning: Whether to use MetaSchedule tuning
        num_trials: Maximum number of trials for tuning
        max_workers: Number of parallel workers
        work_dir: Directory to store tuning database
        opt_level: Optimization level (0-3)

    Returns:
        dict: Compilation results
    """
    try:
        # Parse IR string to module
        relax_mod = tvm.ir.load_json(relax_mod_ir)

        # Setup target
        num_cores = multiprocessing.cpu_count()
        if "llvm" in target_name:
            target = tvm.target.Target(f"llvm -num-cores {num_cores}")
        else:
            target = tvm.target.Target(target_name)

        # Apply zero pipeline
        with target:
            relax_mod = relax.get_pipeline("zero")(relax_mod)

        # MetaSchedule tuning (if enabled)
        if use_auto_tuning:
            import tvm.meta_schedule as ms
            from tvm.meta_schedule.builder import LocalBuilder
            from tvm.ir.transform import PassContext

            os.makedirs(work_dir, exist_ok=True)

            if max_workers is None:
                max_workers = num_cores

            builder = LocalBuilder(max_workers=max_workers)

            with target:
                ms.tune_tir(
                    mod=relax_mod,
                    target=target,
                    work_dir=work_dir,
                    max_trials_global=num_trials,
                    builder=builder,
                    num_tuning_cores=max_workers,
                )

            # Apply database
            with target, PassContext(opt_level=opt_level):
                application_pass = relax.transform.MetaScheduleApplyDatabase(work_dir)
                relax_mod = application_pass(relax_mod)

        # Build
        ex = relax.build(relax_mod, target)

        # Save
        lib_path = os.path.join(work_dir, "compiled_lib.so")
        ex.export_library(lib_path)

        return {
            "status": "success",
            "lib_path": lib_path,
            "work_dir": work_dir,
            "auto_tuning": use_auto_tuning,
            "num_trials": num_trials if use_auto_tuning else 0,
            "opt_level": opt_level,
            "target": str(target)
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def create_simple_relax_ir():
    """
    Create a simple Relax IR for testing MetaSchedule.
    Creates a simple matrix multiplication module.

    Returns:
        str: Relax module IR as JSON string
    """
    import tvm
    from tvm.script import relax as R
    from tvm.script import tir as T

    @tvm.script.ir_module
    class SimpleMatmul:
        @T.prim_func
        def matmul(
            A: T.Buffer((128, 128), "float32"),
            B: T.Buffer((128, 128), "float32"),
            C: T.Buffer((128, 128), "float32"),
        ):
            for i, j, k in T.grid(128, 128, 128):
                with T.block("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

        @R.function
        def main(
            x: R.Tensor((128, 128), dtype="float32"),
            y: R.Tensor((128, 128), dtype="float32"),
        ) -> R.Tensor((128, 128), dtype="float32"):
            cls = SimpleMatmul
            with R.dataflow():
                lv0 = R.call_tir(cls.matmul, (x, y), out_sinfo=R.Tensor((128, 128), dtype="float32"))
                R.output(lv0)
            return lv0

    # Convert to JSON string
    return tvm.ir.save_json(SimpleMatmul)


def get_metaschedule_config():
    """
    Get MetaSchedule configuration information.

    Returns:
        dict: Configuration information
    """
    num_cores = multiprocessing.cpu_count()

    return {
        "available_cores": num_cores,
        "default_work_dir": "tuning_database",
        "supported_targets": ["llvm", "cuda", "metal", "opencl"],
        "metaschedule_available": True
    }


def check_tuning_database(work_dir="tuning_database"):
    """
    Check if tuning database exists and get info.

    Args:
        work_dir: Directory to check

    Returns:
        dict: Database information
    """
    if not os.path.exists(work_dir):
        return {
            "exists": False,
            "path": work_dir
        }

    # Check for database files
    db_files = []
    if os.path.isdir(work_dir):
        for file in os.listdir(work_dir):
            if file.endswith(".json") or file.endswith(".db") or file.endswith(".so"):
                db_files.append(file)

    return {
        "exists": True,
        "path": work_dir,
        "files": db_files,
        "file_count": len(db_files)
    }


if __name__ == "__main__":
    print("=== TVM MetaSchedule Functions Test ===\n")

    # Test get_metaschedule_config
    config = get_metaschedule_config()
    print(f"MetaSchedule Config: {config}\n")

    # Test check_tuning_database
    db_info = check_tuning_database()
    print(f"Tuning Database Info: {db_info}\n")
