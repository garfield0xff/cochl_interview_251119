"""
TVM MetaSchedule Integration for ResNet18

This module provides TVM compilation and auto-tuning functionality
for ResNet18 model using MetaSchedule.
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


def create_resnet18_relax_ir(pretrained=True, keep_params=False):
    """
    Convert ResNet18 PyTorch model to TVM Relax IR

    Args:
        pretrained (bool): Use pretrained weights
        keep_params (bool): Keep parameters as input (False embeds them)

    Returns:
        dict: Contains 'relax_mod' (IR module as string) and metadata
    """
    try:
        # Load PyTorch model
        pytorch_model = load_resnet18_pytorch(pretrained=pretrained)

        # Trace model using torch.fx
        with torch.no_grad():
            traced_model = fx.symbolic_trace(pytorch_model)

        # Convert to TVM Relax IR
        from tvm.relax.frontend.torch import from_fx
        input_info = [((1, 3, 224, 224), "float32")]

        with torch.no_grad():
            relax_mod = from_fx(
                traced_model,
                input_info,
                keep_params_as_input=keep_params
            )

        # Convert IR to string for C++ consumption
        relax_mod_str = str(relax_mod)

        return {
            'status': 'success',
            'relax_mod': relax_mod_str,
            'input_shape': '(1, 3, 224, 224)',
            'dtype': 'float32',
            'pretrained': str(pretrained)
        }

    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e)
        }


def preprocess_image_for_resnet(image_path):
    """
    Preprocess image for ResNet18

    Args:
        image_path (str): Path to image

    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    from torchvision import transforms
    from PIL import Image

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


def tune_resnet18_with_metaschedule(
    image_path,
    use_auto_tuning=True,
    num_trials=64,
    opt_level=3,
    max_workers=None,
    work_dir="tuning_database"
):
    """
    Compile and tune ResNet18 using TVM MetaSchedule

    Args:
        image_path (str): Path to test image
        use_auto_tuning (bool): Enable MetaSchedule auto-tuning
        num_trials (int): Number of tuning trials
        opt_level (int): Optimization level (0-3)
        max_workers (int): Number of parallel workers (None = auto)
        work_dir (str): Directory for tuning database

    Returns:
        dict: Compilation and inference results
    """
    try:
        # Load model and convert to Relax IR
        pytorch_model = load_resnet18_pytorch(pretrained=True)

        with torch.no_grad():
            traced_model = fx.symbolic_trace(pytorch_model)

        from tvm.relax.frontend.torch import from_fx
        input_info = [((1, 3, 224, 224), "float32")]

        with torch.no_grad():
            relax_mod = from_fx(
                traced_model,
                input_info,
                keep_params_as_input=False
            )

        # Setup target
        num_cores = multiprocessing.cpu_count()
        target = tvm.target.Target(f"llvm -num-cores {num_cores}")

        # Apply pipeline
        with target:
            relax_mod = relax.get_pipeline("zero")(relax_mod)

        # MetaSchedule tuning
        if use_auto_tuning:
            os.makedirs(work_dir, exist_ok=True)

            if max_workers is None:
                max_workers = num_cores

            builder = LocalBuilder(max_workers=max_workers)

            with target, PassContext(opt_level=opt_level):
                ms.tune_tir(
                    mod=relax_mod,
                    target=target,
                    work_dir=work_dir,
                    max_trials_per_task=200,        # 각 task당 최대 200 trials
                    num_trials_per_iter=64,         # 반복당 64 trials -> batch size
                    max_trials_global=num_trials,
                    builder=builder,
                    num_tuning_cores=max_workers,
                    post_optimization=True,         # 후처리 최적화 활성화
                )

            # Apply tuning database
            with target, PassContext(opt_level=opt_level):
                application_pass = relax.transform.MetaScheduleApplyDatabase(work_dir)
                relax_mod = application_pass(relax_mod)

        # Build
        ex = relax.build(relax_mod, target)

        # Create VM
        device = tvm.cpu()
        vm = relax.VirtualMachine(ex, device)

        # Load and preprocess image
        import time
        import numpy as np

        image_tensor = preprocess_image_for_resnet(image_path)
        img_np = image_tensor.numpy()
        img_tvm = tvm.runtime.tensor(img_np)

        # Warmup
        for _ in range(5):
            _ = vm["main"](img_tvm)

        # Benchmark
        num_iterations = 10
        inference_times = []

        for _ in range(num_iterations):
            start_time = time.time()
            output = vm["main"](img_tvm)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)

        # Statistics
        avg_time = float(np.mean(inference_times))
        std_time = float(np.std(inference_times))
        min_time = float(np.min(inference_times))
        max_time = float(np.max(inference_times))

        # Get predictions
        output_np = output.numpy()
        exp_output = np.exp(output_np - np.max(output_np))
        probabilities = exp_output / np.sum(exp_output)

        top_indices = np.argsort(probabilities[0])[::-1][:5]

        # Load ImageNet labels
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        categories = weights.meta["categories"]

        top5_predictions = []
        for idx in top_indices:
            top5_predictions.append({
                'class_id': int(idx),
                'class_name': categories[idx],
                'probability': float(probabilities[0][idx])
            })

        return {
            'status': 'success',
            'tuning_enabled': str(use_auto_tuning),
            'num_trials': str(num_trials),
            'opt_level': str(opt_level),
            'work_dir': work_dir,
            'avg_inference_time_ms': str(avg_time),
            'std_inference_time_ms': str(std_time),
            'min_inference_time_ms': str(min_time),
            'max_inference_time_ms': str(max_time),
            'top1_class': categories[top_indices[0]],
            'top1_probability': str(float(probabilities[0][top_indices[0]])),
            'num_iterations': str(num_iterations)
        }

    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }
