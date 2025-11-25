"""
TFLite to TVM Runtime Exporter

변환 흐름:
    TFLite (.tflite) -> ONNX (.onnx) -> TVM Relax IR -> Shared Library (.so)

필요 패키지:
    pip install tf2onnx onnx tensorflow
"""

import os
import tempfile
import subprocess
import numpy as np
import tvm
from tvm import relax


def convert_tflite_to_onnx(tflite_path: str, onnx_path: str, opset: int = 14):
    """TFLite -> ONNX 변환

    Parameters
    ----------
    tflite_path : str -> 입력 TFLite 모델 경로
    onnx_path : str -> 출력 ONNX 모델 경로
    opset : int -> ONNX opset 버전
    """
    cmd = [
        "python3", "-m", "tf2onnx.convert",
        "--tflite", tflite_path,
        "--output", onnx_path,
        "--opset", str(opset)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"TFLite -> ONNX 변환 실패:\n{result.stderr}")

    print(f"TFLite -> ONNX 변환 완료: {onnx_path}")


def load_onnx_to_relax(onnx_path: str, opset: int = 14):
    """ONNX -> TVM Relax 변환

    Parameters
    ----------
    onnx_path : str -> ONNX 모델 경로
    opset : int -> ONNX opset 버전

    Returns
    -------
    mod : IRModule -> Relax 모듈
    """
    import onnx
    from tvm.relax.frontend.onnx import from_onnx

    onnx_model = onnx.load(onnx_path)
    print(f"ONNX 모델 로드됨: {onnx_path}")

    # ONNX -> Relax 변환
    # keep_params_in_input=False: 파라미터를 모델 내에 상수로 바인딩
    # 이렇게 하면 main 함수가 입력 텐서만 받게 됨
    mod = from_onnx(onnx_model, opset=opset, keep_params_in_input=False)
    print("ONNX -> Relax 변환 완료 (params embedded in model)")

    return mod


def apply_relax_transforms(mod, target):
    """Relax 모듈에 최적화 변환 적용

    Parameters
    ----------
    mod : IRModule -> Relax 모듈
    target : Target -> TVM 타겟

    Returns
    -------
    mod : IRModule -> 변환된 Relax 모듈
    """
    # BatchNorm 등을 추론에 적합한 형태로 변환
    mod = relax.transform.DecomposeOpsForInference()(mod)
    print("  - DecomposeOpsForInference 완료")

    # Relax 연산자를 TensorIR로 변환
    mod = relax.transform.LegalizeOps()(mod)
    print("  - LegalizeOps 완료")

    # 최적화 파이프라인 적용
    with target:
        mod = relax.get_pipeline("zero")(mod)
    print("  - Optimization pipeline 완료")

    return mod


def compile_relax_model(mod, target: str = "llvm", opt_level: int = 3):
    """Relax 모듈 컴파일

    Parameters
    ----------
    mod : IRModule -> Relax 모듈
    target : str -> 타겟 (e.g., "llvm", "cuda")
    opt_level : int -> 최적화 레벨 (0-3)

    Returns
    -------
    ex : Executable -> 컴파일된 실행 파일
    """
    target = tvm.target.Target(target)

    print(f"컴파일 시작 (target: {target}, opt_level: {opt_level})")

    # 변환 적용
    mod = apply_relax_transforms(mod, target)

    # 컴파일
    with target, tvm.transform.PassContext(opt_level=opt_level):
        ex = relax.build(mod, target)

    print("Relax 컴파일 완료")
    return ex


def create_virtual_machine(ex, target: str = "llvm"):
    """VirtualMachine 생성

    Parameters
    ----------
    ex : Executable -> 컴파일된 실행 파일
    target : str -> 타겟

    Returns
    -------
    vm : VirtualMachine -> 실행 VM
    device : Device -> 디바이스
    """
    if "cuda" in target:
        device = tvm.cuda()
    else:
        device = tvm.cpu()

    vm = relax.VirtualMachine(ex, device)
    return vm, device


def run_inference(vm, input_data: np.ndarray, device):
    """TVM Relax 추론 실행

    Parameters
    ----------
    vm : VirtualMachine -> 실행 VM
    input_data : np.ndarray -> 입력 데이터
    device : Device -> 디바이스

    Returns
    -------
    output : np.ndarray -> 추론 결과
    """
    # 입력 텐서 생성
    input_tvm = tvm.runtime.empty(input_data.shape, input_data.dtype, device)
    input_tvm.copyfrom(input_data)

    # 추론 실행 (파라미터가 모델 내에 바인딩되어 있으므로 입력만 전달)
    output = vm["main"](input_tvm)

    return output.numpy()


def export_to_so(ex, output_path: str):
    """컴파일된 모듈을 .so 파일로 내보내기

    Parameters
    ----------
    ex : Executable -> 컴파일된 실행 파일
    output_path : str -> 출력 .so 파일 경로
    """
    ex.export_library(output_path)
    print(f"라이브러리 저장됨: {output_path}")


def main():
    # 모델 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, "..", "..", ".."))
    tflite_path = os.path.join(project_root, "models", "resnet50.tflite")

    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite model not found: {tflite_path}")

    print("=" * 50)
    print("TFLite to TVM Runtime Exporter")
    print("=" * 50)
    print(f"Input: {tflite_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")

        # Step 1: TFLite -> ONNX
        print("\n" + "=" * 50)
        print("[1/4] TFLite -> ONNX")
        print("=" * 50)
        convert_tflite_to_onnx(tflite_path, onnx_path)

        # Step 2: ONNX -> Relax
        print("\n" + "=" * 50)
        print("[2/4] ONNX -> TVM Relax")
        print("=" * 50)
        mod = load_onnx_to_relax(onnx_path)

        # Step 3: Compile
        print("\n" + "=" * 50)
        print("[3/4] Compile Relax Model")
        print("=" * 50)
        ex = compile_relax_model(mod, target="llvm", opt_level=3)

        # VirtualMachine 생성 및 테스트
        vm, device = create_virtual_machine(ex, target="llvm")

        # Inference 테스트
        print("\n" + "=" * 50)
        print("Inference Test")
        print("=" * 50)
        input_shape = (1, 224, 224, 3)  # NHWC for TFLite
        input_data = np.random.randn(*input_shape).astype(np.float32)
        print(f"input shape: {input_data.shape}")

        tvm_output = run_inference(vm, input_data, device)
        print(f"TVM output shape: {tvm_output.shape}")
        print(f"TVM output sample: {tvm_output.flatten()[:5]}")

        # Step 4: Export .so
        print("\n" + "=" * 50)
        print("[4/4] Export to .so")
        print("=" * 50)
        output_dir = os.path.join(project_root, "models")
        so_path = os.path.join(output_dir, "resnet50_tvm.so")
        export_to_so(ex, so_path)

    print("\n" + "=" * 50)
    print("Success!")
    print("=" * 50)


if __name__ == "__main__":
    main()
