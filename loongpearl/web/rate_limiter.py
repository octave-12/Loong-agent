#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
请求速率控制器 — 熔断、抖动、User-Agent 轮转。
==================================================

设计目标:
  1. 熔断: 单引擎连续失败 N 次 → 冷却 M 秒，不阻塞其他引擎
  2. 抖动: 请求间隔 random.uniform(min, max) 而非固定延迟
  3. UA 轮转: 多个 User-Agent 随机选取，降低指纹识别
  4. 全局限速: 单分钟内请求上限，溢出 → 排队等待

所有阈值数据驱动（运行时可调），零硬编码魔法数字。
"""

import time
import random
import threading
from typing import Dict, Optional
from collections import deque
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════
# User-Agent 池
# ═══════════════════════════════════════════════════════════════════

UA_POOL = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
    '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
    
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) '
    'Gecko/20100101 Firefox/126.0',
    
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 '
    '(KHTML, like Gecko) Version/17.5 Safari/605.1.15',
]


def random_ua() -> str:
    """随机 User-Agent"""
    return random.choice(UA_POOL)


def random_delay(jitter_min: float = 0.3, jitter_max: float = 1.2) -> float:
    """随机请求抖动"""
    return random.uniform(jitter_min, jitter_max)


# ═══════════════════════════════════════════════════════════════════
# 熔断器
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CircuitBreaker:
    """单引擎熔断器"""
    engine_name: str
    max_failures: int = 3
    cooldown_seconds: int = 300
    half_open_timeout: int = 30
    
    _failure_count: int = 0
    _last_failure_time: float = 0.0
    _open_until: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    @property
    def is_open(self) -> bool:
        """熔断是否开启（拒绝请求）"""
        with self._lock:
            if self._open_until > 0 and time.time() < self._open_until:
                return True
            # 半开状态：超时后允许一次试探
            if self._open_until > 0 and time.time() >= self._open_until:
                self._open_until = 0
                self._failure_count = 0
            return False
    
    @property
    def remaining_cooldown(self) -> float:
        """剩余冷却时间（秒）"""
        return max(0, self._open_until - time.time())
    
    def record_success(self):
        """记录成功 → 重置失败计数"""
        with self._lock:
            self._failure_count = 0
            self._open_until = 0
    
    def record_failure(self):
        """记录失败 → 可能触发熔断"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.max_failures:
                self._open_until = time.time() + self.cooldown_seconds
    
    def can_request(self) -> bool:
        """当前是否可以发起请求"""
        return not self.is_open


# ═══════════════════════════════════════════════════════════════════
# 限速器
# ═══════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    全局限速：分钟级请求上限 + 滑动窗口。
    多引擎共享一个 RateLimiter 实例。
    """
    
    def __init__(self, max_per_minute: int = 30):
        self._max_per_minute = max_per_minute
        self._window: deque = deque()
        self._lock = threading.Lock()
    
    def _clean_window(self):
        """清理过期的时间戳"""
        now = time.time()
        while self._window and now - self._window[0] > 60:
            self._window.popleft()
    
    def acquire(self) -> bool:
        """
        尝试获取请求许可。
        Returns: True=可以发请求, False=需等待
        """
        with self._lock:
            self._clean_window()
            if len(self._window) < self._max_per_minute:
                self._window.append(time.time())
                return True
            return False
    
    def wait_if_needed(self, timeout: float = 5.0) -> bool:
        """如果需要等待，阻塞直到许可可用或超时"""
        start = time.time()
        while not self.acquire():
            if time.time() - start > timeout:
                return False
            time.sleep(0.1)
        return True
    
    @property
    def current_rate(self) -> int:
        """当前分钟内的请求数"""
        with self._lock:
            self._clean_window()
            return len(self._window)


# ═══════════════════════════════════════════════════════════════════
# 引擎请求管理器（熔断 + 限速合体）
# ═══════════════════════════════════════════════════════════════════

class EngineRequestManager:
    """
    管理多个搜索引擎的请求节律。
    每个引擎独立的熔断器 + 共享的全局速率限制。
    """
    
    def __init__(self, global_max_per_minute: int = 30):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._global_limiter = RateLimiter(global_max_per_minute)
        self._engine_limits: Dict[str, RateLimiter] = {}
    
    def register_engine(self, name: str, max_failures: int = 3,
                        cooldown_seconds: int = 300,
                        max_per_minute: int = 10):
        """注册一个引擎"""
        self._breakers[name] = CircuitBreaker(
            engine_name=name,
            max_failures=max_failures,
            cooldown_seconds=cooldown_seconds,
        )
        self._engine_limits[name] = RateLimiter(max_per_minute)
    
    def can_request(self, engine_name: str) -> bool:
        """检查引擎是否可用（熔断未开启 + 未达限速）"""
        breaker = self._breakers.get(engine_name)
        if breaker is None:
            return True  # 未注册引擎默认放行
        if breaker.is_open:
            return False
        
        limiter = self._engine_limits.get(engine_name)
        if limiter is None:
            return True
        return limiter.acquire() and self._global_limiter.acquire()
    
    def record_success(self, engine_name: str):
        breaker = self._breakers.get(engine_name)
        if breaker:
            breaker.record_success()
    
    def record_failure(self, engine_name: str):
        breaker = self._breakers.get(engine_name)
        if breaker:
            breaker.record_failure()
    
    def engine_status(self, engine_name: str) -> str:
        """返回引擎状态字符串"""
        breaker = self._breakers.get(engine_name)
        if breaker is None:
            return 'unregistered'
        if breaker.is_open:
            return f'circuit_open ({breaker.remaining_cooldown:.0f}s remaining)'
        limiter = self._engine_limits.get(engine_name)
        if limiter:
            return f'ok (rate: {limiter.current_rate}/min)'
        return 'ok'
    
    def status_summary(self) -> Dict[str, str]:
        """所有引擎状态摘要"""
        return {name: self.engine_status(name) for name in self._breakers}
