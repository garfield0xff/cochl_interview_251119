"""
TVM Extensions for TVM SDK

This module provides TVM integration utilities including version info,
configuration, and MetaSchedule functionality.
"""

# Import from version module
from .version import get_tvm_version

# Import from config module
from .config import (
    get_tvm_target,
    check_tvm_modules,
    create_simple_ir,
    get_tvm_build_config
)

# Import from metaschedule module
from .metaschedule import (
    tune_with_metaschedule,
    apply_tuning_database,
    compile_with_metaschedule,
    create_simple_relax_ir,
    get_metaschedule_config,
    check_tuning_database
)

# Import from resnet_schedule module
from .resnet_schedule import (
    create_resnet18_relax_ir,
    tune_resnet18_with_metaschedule
)

# Expose all functions
__all__ = [
    # Version
    'get_tvm_version',
    # Config
    'get_tvm_target',
    'check_tvm_modules',
    'create_simple_ir',
    'get_tvm_build_config',
    # MetaSchedule
    'tune_with_metaschedule',
    'apply_tuning_database',
    'compile_with_metaschedule',
    'create_simple_relax_ir',
    'get_metaschedule_config',
    'check_tuning_database',
    # ResNet Schedule
    'create_resnet18_relax_ir',
    'tune_resnet18_with_metaschedule'
]
