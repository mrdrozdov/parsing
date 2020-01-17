# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "0.0.0"

from .utils import configure_logger

configure_logger()

# Work around to update TensorFlow's absl.logging threshold which alters the
# default Python logging output behavior when present.
# see: https://github.com/abseil/abseil-py/issues/99
# and: https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-500369493
try:
    import absl.logging
except ImportError:
    pass
else:
    absl.logging.set_verbosity("info")
    absl.logging.set_stderrthreshold("info")
    absl.logging._warn_preinit_stderr = False

from .configuration_diora import DioraConfig

from .configuration_utils import PretrainedConfig

from .modeling_diora import DioraModel

from .modeling_utils import PretrainedModel

from .tokenization_diora import DioraTokenizer

from .data import (
    FixedLengthBatchSampler,
    SimpleDataset,
)
