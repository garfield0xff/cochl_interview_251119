"""
TVM MetaSchedule ResNet18 Model Builder
"""

import os
import torch
import torch.fx as fx
import tvm
from tvm import relax
import tvm.meta_schedule as ms
from tvm.meta_schedule.builder import LocalBuilder
from tvm.ir.transform import PassContext
import multiprocessing
import numpy as np
from PIL import Image


def load_resnet18_pytorch(pretrained=True):
    """
    Load ResNet18 PyTorch model

    Args:
        pretrained (bool): Use pretrained weights

    Returns:
        torch.nn.Module: ResNet18 model
    """
    from torchvision.models import resnet18, ResNet18_Weights

    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    model.eval()
    return model


def preprocess_image_for_resnet(image_path):
    """
    Preprocess image for ResNet18

    Args:
        image_path (str): Path to image

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    return input_batch


def build_resnet18_with_metaschedule(
    output_path,
    image_path=None,
    use_auto_tuning=True,
    num_trials=2000,
    num_workers=10,
    opt_level=3,
    work_dir="tuning_database",
    use_existing_tuning=False
):
    """
    Build and tune ResNet18 using TVM MetaSchedule, exporting to .so library

    Args:
        output_path (str): Path to save the compiled .so library
        image_path (str): Path to test image (for validation)
        use_auto_tuning (bool): Enable MetaSchedule auto-tuning
        num_trials (int): Number of tuning trials (default: 2000)
        num_workers (int): Number of parallel workers (default: 10)
        opt_level (int): Optimization level (0-3)
        work_dir (str): Directory for tuning database
        use_existing_tuning (bool): Use existing tuning database without re-tuning

    Returns:
        dict: Compilation and validation results
    """
    try:
        print("=" * 80)
        print("TVM MetaSchedule ResNet18 Model Builder")
        print("=" * 80)
        print(f"Output path: {output_path}")
        print(f"Use existing tuning: {use_existing_tuning}")
        print(f"Auto-tuning: {use_auto_tuning}")
        if use_auto_tuning or use_existing_tuning:
            print(f"Work dir: {work_dir}")
            if use_existing_tuning:
                print(f"  Using existing tuning database from: {work_dir}")
            else:
                print(f"Trials: {num_trials}")
                print(f"Workers: {num_workers}")
        print(f"Optimization level: {opt_level}")
        print("=" * 80)

        # Step 1: Load PyTorch model and convert to Relax IR
        print("\n[1/5] Loading ResNet18 model...")
        pytorch_model = load_resnet18_pytorch(pretrained=True)

        with torch.no_grad():
            traced_model = fx.symbolic_trace(pytorch_model)

        print("[1/5] Converting to TVM Relax IR...")
        from tvm.relax.frontend.torch import from_fx
        input_info = [((1, 3, 224, 224), "float32")]

        with torch.no_grad():
            relax_mod = from_fx(
                traced_model,
                input_info,
                keep_params_as_input=False  # Embed parameters in the module
            )

        print("[1/5] Relax IR conversion completed")

        # Step 2: Setup target
        print("\n[2/5] Setting up target configuration...")
        num_cores = multiprocessing.cpu_count()
        # Disable LLVM debug info to avoid verification errors
        target = tvm.target.Target(f"llvm -num-cores {num_cores}")
        print(f"[2/5] Target: {target}, CPU cores: {num_cores}")

        # Step 3: Apply optimization pipeline
        print("\n[3/5] Applying optimization pipeline...")
        with target:
            relax_mod = relax.get_pipeline("zero")(relax_mod)
        print("[3/5] Optimization pipeline applied")

        # Step 4: MetaSchedule tuning (if enabled)
        if use_existing_tuning:
            # Use existing tuning database
            print(f"\n[4/5] Loading existing tuning database from {work_dir}...")

            if not os.path.exists(work_dir):
                raise ValueError(f"Tuning database not found at: {work_dir}")

            # Check if database has content
            db_files = os.listdir(work_dir)
            if not db_files:
                raise ValueError(f"Tuning database is empty at: {work_dir}")

            print(f"[4/5] Found tuning database with {len(db_files)} files")

            # Apply existing tuning database to the module
            print("[4/5] Applying existing tuning database to module...")
            with target, PassContext(opt_level=opt_level):
                application_pass = relax.transform.MetaScheduleApplyDatabase(work_dir)
                relax_mod = application_pass(relax_mod)

            print("[4/5] Existing tuning database applied successfully")

        elif use_auto_tuning:
            print(f"\n[4/5] Starting MetaSchedule tuning with {num_trials} trials...")
            os.makedirs(work_dir, exist_ok=True)

            builder = LocalBuilder(max_workers=num_workers)

            with target, PassContext(opt_level=opt_level):
                # Tune the TIR primitives in the Relax module
                ms.tune_tir(
                    mod=relax_mod,
                    target=target,
                    work_dir=work_dir,
                    max_trials_per_task=200,        # Max 200 trials per task
                    num_trials_per_iter=64,         # 64 trials per iteration
                    max_trials_global=num_trials,   # Total global trials
                    builder=builder,
                    num_tuning_cores=num_workers,
                    post_optimization=True,         # Enable post-processing optimization
                )

            print(f"[4/5] Tuning completed. Database saved to: {work_dir}")

            # Apply tuning database to the module
            print("[4/5] Applying tuning database to module...")
            with target, PassContext(opt_level=opt_level):
                application_pass = relax.transform.MetaScheduleApplyDatabase(work_dir)
                relax_mod = application_pass(relax_mod)

            print("[4/5] Tuning database applied successfully")
        else:
            print("\n[4/5] Skipping auto-tuning (disabled)")

        # Step 5: Build and export to shared library
        print(f"\n[5/5] Building model and exporting to {output_path}...")

        # Build the module
        with target, tvm.transform.PassContext(opt_level=opt_level):
            ex = relax.build(relax_mod, target)

        # Export to shared library (.so file)
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Export the module
        ex.export_library(output_path)

        print(f"[5/5] Model successfully exported to: {output_path}")
        print(f"[5/5] Library size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

        result = {
            'status': 'success',
            'output_path': output_path,
            'library_size_mb': f"{os.path.getsize(output_path) / (1024*1024):.2f}",
            'tuning_enabled': str(use_auto_tuning),
            'num_trials': str(num_trials),
            'num_workers': str(num_workers),
            'opt_level': str(opt_level),
            'work_dir': work_dir if use_auto_tuning else 'N/A',
            'target': str(target)
        }

        # Step 6: Optional validation with test image
        if image_path and os.path.exists(image_path):
            print("\n[Validation] Running inference validation with test image...")
            try:
                # Load the compiled module
                loaded_module = tvm.runtime.Module.LoadFromFile(output_path)
                device = tvm.cpu()
                vm = relax.VirtualMachine(loaded_module, device)

                # Preprocess image
                image_tensor = preprocess_image_for_resnet(image_path)
                img_np = image_tensor.numpy()
                img_tvm = tvm.runtime.NDArray.array(img_np, device)

                # Run inference
                output = vm["main"](img_tvm)
                output_np = output.numpy()

                # Get top-5 predictions
                exp_output = np.exp(output_np - np.max(output_np))
                probabilities = exp_output / np.sum(exp_output)
                top_indices = np.argsort(probabilities[0])[::-1][:5]

                # Load ImageNet labels
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.IMAGENET1K_V1
                categories = weights.meta["categories"]

                print("\n[Validation] Top-5 predictions:")
                for i, idx in enumerate(top_indices):
                    print(f"  {i+1}. {categories[idx]}: {probabilities[0][idx]*100:.2f}%")

                result['validation'] = 'success'
                result['top1_class'] = categories[top_indices[0]]
                result['top1_probability'] = f"{probabilities[0][top_indices[0]]*100:.2f}%"

            except Exception as e:
                print(f"[Validation] Warning: Validation failed: {e}")
                result['validation'] = f'failed: {str(e)}'

        print("\n" + "=" * 80)
        print("Build completed successfully!")
        print("=" * 80)

        return result

    except Exception as e:
        import traceback
        error_msg = {
            'status': 'error',
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }
        print("\n" + "=" * 80)
        print("Build failed!")
        print("=" * 80)
        print(f"Error: {e}")
        print(traceback.format_exc())
        return error_msg


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='TVM MetaSchedule ResNet18 Builder - Build .so library for C++ runtime'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for compiled .so library'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to test image for validation (optional)'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Enable MetaSchedule auto-tuning (performs new tuning)'
    )
    parser.add_argument(
        '--use-existing-tuning',
        action='store_true',
        help='Use existing tuning database without re-tuning'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=2000,
        help='Number of tuning trials (default: 2000)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=10,
        help='Number of parallel workers (default: 10)'
    )
    parser.add_argument(
        '--opt-level',
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help='Optimization level (default: 3)'
    )
    parser.add_argument(
        '--work-dir',
        type=str,
        default='tuning_database',
        help='Tuning database directory (default: tuning_database)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.tune and args.use_existing_tuning:
        parser.error("Cannot use both --tune and --use-existing-tuning at the same time")

    # Handle directory output path
    output_path = args.output
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "resnet18_tvm.so")

    result = build_resnet18_with_metaschedule(
        output_path=output_path,
        image_path=args.image,
        use_auto_tuning=args.tune,
        num_trials=args.trials,
        num_workers=args.workers,
        opt_level=args.opt_level,
        work_dir=args.work_dir,
        use_existing_tuning=args.use_existing_tuning
    )

    print("\n" + "=" * 80)
    print("Build Summary")
    print("=" * 80)
    print(json.dumps(result, indent=2))

    if result['status'] != 'success':
        exit(1)
