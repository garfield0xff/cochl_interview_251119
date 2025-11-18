"""
TVM Version Information
"""

import tvm


def get_tvm_version():
    """
    Get TVM version string.

    Returns:
        str: TVM version
    """
    return tvm.__version__


if __name__ == "__main__":
    version = get_tvm_version()
    print(f"version: {version}")
