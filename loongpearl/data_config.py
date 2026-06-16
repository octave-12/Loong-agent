"""龙珠数据路径配置 — 所有模块通过此模块解析数据文件路径"""
from pathlib import Path

# 项目根路径 = Loong-pearl/
_PROJECT_ROOT = Path(__file__).parent.parent

# 数据根路径
DATA_ROOT = _PROJECT_ROOT / "data"

def resolve(*parts: str) -> Path:
    """解析数据文件路径: resolve('models', 'zichang_94117_1024d.pt')"""
    return DATA_ROOT.joinpath(*parts)

# 常用路径快捷方式
MODEL_DIR = DATA_ROOT / "models"
DICT_DIR = DATA_ROOT / "dicts"
WORDLIST_DIR = DATA_ROOT / "wordlists"
RUNTIME_DIR = DATA_ROOT / "runtime"
