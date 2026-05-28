"""全局配置管理"""
import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Settings:
    """测试生成器全局配置"""
    error_thresholds: Dict[str, tuple] = field(default_factory=lambda: {
        "float16": (1e-3, 1e-2),
        "bfloat16": (5e-3, 5e-2),
        "float32": (1e-5, 1e-4),
        "float64": (1e-10, 1e-9),
    })

    op_specific_thresholds: Dict[str, tuple] = field(default_factory=lambda: {
        "aten::conv2d": (1e-1, 1e-1),
        "aten::convolution": (1e-1, 1e-1),
        "aten::mm": (1e-3, 1e-1),
        "aten::matmul": (1e-3, 1e-1),
        "aten::bmm": (1e-3, 1e-1),
        "aten::native_batch_norm": (1e-3, 1e-1),
        "aten::batch_norm": (1e-3, 1e-1),
        "aten::sum": (1e-3, 1e-1),
        "aten::mean": (1e-3, 1e-1),
        "aten::softmax": (1e-3, 1e-1),
        "aten::log_softmax": (1e-3, 1e-1),
        "aten::linear": (1e-3, 1e-1),
        "aten::embedding": (1e-3, 1e-1),
        "aten::embedding_bag": (1e-3, 1e-1),
        "aten::cross_entropy_loss": (1e-3, 1e-1),
        "aten::nll_loss": (1e-3, 1e-1),
    })

    # 性能测试
    perf_warmup_iters: int = 0
    perf_cpu_iters: int = 1
    perf_target_iters: int = 3
    perf_benchmark_iters: int = 3  # 保留兼容，语义等同 target_iters

    # 随机种子
    random_seed: int = 42

    # 白名单与分类表路径
    whitelist_path: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), "op_whitelist.yaml"
    ))
    classification_path: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(__file__), "op_classification.yaml"
    ))

    # 报告输出
    default_output_format: str = "html"
    default_output_path: str = "op_testgen_report.html"


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """获取全局配置单例"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
