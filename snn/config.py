"""配置加载与消息队列选项解析工具。"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """加载项目配置文件，发生错误时返回空字典。"""

    if not _CONFIG_PATH.exists():
        return {}

    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
    except Exception as exc:  # pragma: no cover - 极端情况下的容错
        logger.warning("读取配置文件失败：%s", exc)
        return {}

    if not isinstance(data, dict):
        logger.warning("配置文件格式无效：%r", data)
        return {}
    return data


def get_logging_level(default_level: int = logging.INFO) -> int:
    """根据配置返回日志等级，未配置时使用默认值。"""

    config = load_config()
    logging_cfg = config.get("logging")
    if isinstance(logging_cfg, dict):
        level_name = logging_cfg.get("level")
        if isinstance(level_name, str):
            level_value = getattr(logging, level_name.upper(), None)
            if isinstance(level_value, int):
                return level_value
    return default_level


def get_message_queue_config() -> Dict[str, Any]:
    """返回消息队列配置块，未设置时返回空字典。"""

    config = load_config()
    mq_cfg = config.get("message_queue")
    if isinstance(mq_cfg, dict):
        return mq_cfg
    return {}


__all__ = ["load_config", "get_logging_level", "get_message_queue_config"]
