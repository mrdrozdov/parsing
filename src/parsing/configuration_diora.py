from .configuration_utils import PretrainedConfig


DIORA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class DioraConfig(PretrainedConfig):
    pretrained_config_archive_map = DIORA_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "diora"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
