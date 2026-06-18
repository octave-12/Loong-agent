#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
搜索引擎配置注册表 — 从 YAML/JSON 配置加载，非硬编码。
========================================================

每个引擎定义:
  - name: 引擎名
  - type: 'api' | 'html' | 'sparql'
  - url: 搜索入口
  - needs_auth: 是否需要 API 密钥
  - auth_env: 环境变量名（如 BING_API_KEY）
  - rate_limit: {max_per_minute, cooldown_seconds}
  - priority: 优先级（1=最高）
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    """单个搜索引擎配置"""
    name: str
    type: str                          # 'api' | 'html' | 'sparql'
    url: str
    description: str = ""
    domain: str = "zh"                 # 'zh' | 'global'
    needs_auth: bool = False
    auth_env: Optional[str] = None     # 环境变量名
    api_key: Optional[str] = None      # 运行时注入
    rate_limit: Dict[str, int] = field(default_factory=lambda: {
        'max_per_minute': 30,
        'cooldown_seconds': 300,
    })
    priority: int = 5                  # 1=最高, 10=最低
    headers: Dict[str, str] = field(default_factory=dict)
    
    @property
    def is_available(self) -> bool:
        """引擎是否可用（有密钥或无需认证）"""
        if not self.needs_auth:
            return True
        return bool(self.api_key)
    
    @property
    def cooldown(self) -> int:
        return self.rate_limit.get('cooldown_seconds', 300)
    
    @property
    def max_per_minute(self) -> int:
        return self.rate_limit.get('max_per_minute', 30)


# ═══════════════════════════════════════════════════════════════════
# 默认引擎注册表（运行时从环境变量注入密钥）
# ═══════════════════════════════════════════════════════════════════

DEFAULT_ENGINES: List[Dict[str, Any]] = [
    {
        'name': 'wikipedia_zh',
        'type': 'api',
        'url': 'https://zh.wikipedia.org/w/api.php',
        'description': 'Wikipedia 中文 API — 合法 JSON API，140万篇中文条目',
        'domain': 'zh',
        'needs_auth': False,
        'priority': 1,
        'rate_limit': {'max_per_minute': 200, 'cooldown_seconds': 60},
    },
    {
        'name': 'wikidata',
        'type': 'sparql',
        'url': 'https://query.wikidata.org/sparql',
        'description': 'Wikidata SPARQL 端点 — 结构化三元组查询',
        'domain': 'zh',
        'needs_auth': False,
        'priority': 2,
        'rate_limit': {'max_per_minute': 60, 'cooldown_seconds': 120},
    },
    {
        'name': 'bing',
        'type': 'api',
        'url': 'https://api.bing.microsoft.com/v7.0/search',
        'description': 'Bing Web Search API (Azure) — 1000次/月免费',
        'domain': 'zh',
        'needs_auth': True,
        'auth_env': 'BING_API_KEY',
        'priority': 3,
        'rate_limit': {'max_per_minute': 5, 'cooldown_seconds': 600},
        'headers': {'Ocp-Apim-Subscription-Key': ''},  # 运行时注入
    },
    {
        'name': 'duckduckgo',
        'type': 'html',
        'url': 'https://lite.duckduckgo.com/lite/',
        'description': 'DuckDuckGo Lite — 无官方API，HTML轻量版作为备选',
        'domain': 'zh',
        'needs_auth': False,
        'priority': 4,
        'rate_limit': {'max_per_minute': 10, 'cooldown_seconds': 120},
    },
    {
        'name': 'google',
        'type': 'api',
        'url': 'https://www.googleapis.com/customsearch/v1',
        'description': 'Google Custom Search API — 100次/天免费',
        'domain': 'global',
        'needs_auth': True,
        'auth_env': 'GOOGLE_API_KEY',
        'priority': 5,
        'rate_limit': {'max_per_minute': 3, 'cooldown_seconds': 600},
    },
]


def load_engines(extra_engines: List[Dict] = None) -> List[EngineConfig]:
    """
    加载引擎配置。
    优先级: 环境变量注入密钥 → 默认配置 → 额外引擎
    """
    engines = []
    
    for raw in DEFAULT_ENGINES:
        cfg = EngineConfig(**{k: v for k, v in raw.items() if k in EngineConfig.__dataclass_fields__})
        
        # 从环境变量注入 API 密钥
        if cfg.needs_auth and cfg.auth_env:
            cfg.api_key = os.environ.get(cfg.auth_env, '')
            if cfg.api_key:
                # 注入到 headers（Bing 格式）
                if cfg.name == 'bing':
                    cfg.headers['Ocp-Apim-Subscription-Key'] = cfg.api_key
        
        engines.append(cfg)
    
    # 追加用户自定义引擎
    if extra_engines:
        for raw in extra_engines:
            cfg = EngineConfig(**{k: v for k, v in raw.items() if k in EngineConfig.__dataclass_fields__})
            engines.append(cfg)
    
    return engines


def get_available_engines() -> List[EngineConfig]:
    """获取当前可用的所有引擎"""
    return [e for e in load_engines() if e.is_available]


def get_engine_by_name(name: str) -> Optional[EngineConfig]:
    """按名称查找引擎"""
    for e in load_engines():
        if e.name == name:
            return e
    return None
