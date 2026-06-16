"""
龙珠 (LoongPearl) — 基于汉字嵌入的确定性知识内核
=================================================
以94117个汉字锚点字场为底座，构建可微分能量景观，
实现知识播种、增量学习、频率感知、自演化对话。

包结构:
    loongpearl.core          — 字场、能量景观（基础层）
    loongpearl.learning      — 知识播种、增量学习、婴儿课程
    loongpearl.voice         — 婴儿嗓音合成
    loongpearl.web           — 联网检索
    loongpearl.interaction   — 对话引擎、原生回答
    loongpearl.utils         — 可视化、监控、工具

用法:
    from loongpearl import HanziAnchorField, EnergyLandscape, LoongPearl
"""

from pathlib import Path

# ── 数据根路径 ──
# 所有数据文件统一通过此路径解析，避免硬编码
_DATA_ROOT = Path(__file__).parent.parent / "data"

def data_path(relative: str) -> Path:
    """返回数据文件的绝对路径"""
    return _DATA_ROOT / relative

def model_path(name: str) -> Path:
    """返回模型文件路径 (data/models/<name>)"""
    return _DATA_ROOT / "models" / name

def dict_path(name: str) -> Path:
    """返回字典文件路径 (data/dicts/<name>)"""
    return _DATA_ROOT / "dicts" / name

def runtime_path(name: str) -> Path:
    """返回运行时文件路径 (data/runtime/<name>)"""
    return _DATA_ROOT / "runtime" / name

def wordlist_path(name: str) -> Path:
    """返回字表路径 (data/wordlists/<name>)"""
    return _DATA_ROOT / "wordlists" / name

# ── 延迟导入核心类 ──
def __getattr__(name):
    """延迟导入，避免循环依赖"""
    _exports = {
        # core
        "HanziAnchorField": "loongpearl.core.zichang",
        "EnergyLandscape": "loongpearl.core.energy_landscape",
        "FreqEnergyLandscape": "loongpearl.core.freq_landscape",
        # learning
        "LoongPearlLearner": "loongpearl.learning.learner",
        "LoongPearlSeeder": "loongpearl.learning.seeder",
        # interaction
        "LoongPearl": "loongpearl.interaction.engine",
    }
    if name in _exports:
        import importlib
        mod = importlib.import_module(_exports[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'loongpearl' has no attribute '{name}'")
