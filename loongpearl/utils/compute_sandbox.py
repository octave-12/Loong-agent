#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠计算沙盒 (compute_sandbox.py)
=================================
安全的 Python 数学计算执行器，为龙珠提供数值计算能力。

龙珠的能量景观擅长语义推理，但不擅长精确数值计算。
计算沙盒作为辅助模块，接管数学类问题，通过 AST 白名单审计确保安全执行。

安全策略:
  1. AST 白名单语法检查 —— 只允许安全的数学运算节点
  2. 禁止导入模块、文件 IO、网络访问、系统调用
  3. 超时控制 —— 防止无限循环
  4. 受限的全局命名空间 —— 仅暴露 math 模块和安全内置函数

支持的计算类型:
  - 基本算术: + - * / // % **
  - 数学函数: sin, cos, sqrt, log, log10, exp, pow, pi, e 等
  - 数值运算: abs, sum, min, max, round, int, float
  - 比较运算: == != < > <= >=
  - 列表/元组字面量: [1, 2, 3], (1, 2)

作者: Hermes + 李泽坤
"""

import ast
import math
import re
import signal
import sys
import traceback
from typing import Dict, Optional, Tuple


# ============================================================================
# 允许的 AST 节点类型（白名单）
# ============================================================================

# 表达式节点
_ALLOWED_NODES = {
    # 字面量
    ast.Constant,       # 数字、字符串、None、True、False
    ast.List,           # [1, 2, 3]
    ast.Tuple,          # (1, 2)
    ast.Dict,           # {"a": 1}
    ast.Set,            # {1, 2, 3}
    ast.ListComp,       # [x for x in ...]
    ast.comprehension,  # 推导式内部

    # 变量与赋值
    ast.Name,           # 变量名
    ast.Starred,        # *args
    ast.Load,           # 读取上下文
    ast.Store,          # 存储上下文

    # 表达式节点
    ast.Expr,           # 独立表达式语句
    ast.Expression,     # eval 模式的顶层节点
    ast.Attribute,      # math.pi, math.e 等模块属性访问

    # 二元运算
    ast.BinOp,          # a + b
    ast.Add,            # +
    ast.Sub,            # -
    ast.Mult,           # *
    ast.Div,            # /
    ast.FloorDiv,       # //
    ast.Mod,            # %
    ast.Pow,            # **

    # 一元运算
    ast.UnaryOp,        # -a, +a, ~a
    ast.UAdd,           # 一元 +
    ast.USub,           # 一元 -
    ast.Invert,         # ~

    # 比较运算
    ast.Compare,        # a < b
    ast.Eq,             # ==
    ast.NotEq,          # !=
    ast.Lt,             # <
    ast.Gt,             # >
    ast.LtE,            # <=
    ast.GtE,            # >=
    ast.Is,             # is
    ast.IsNot,          # is not
    ast.In,             # in
    ast.NotIn,          # not in

    # 布尔运算
    ast.BoolOp,         # and / or
    ast.And,            # and
    ast.Or,             # or
    ast.Not,            # not

    # 条件与循环
    ast.IfExp,          # x if cond else y
    ast.For,            # for 循环
    ast.If,             # if 语句

    # 下标与切片
    ast.Subscript,      # a[i]
    ast.Slice,          # a[i:j]

    # 函数调用
    ast.Call,           # f(x)
    ast.keyword,        # f(x=1) 中的关键字参数
    ast.arguments,      # 函数参数

    # 赋值
    ast.Assign,         # x = 1
    ast.AugAssign,      # x += 1

    # 模块体
    ast.Module,         # exec 模式的顶层节点

    # 格式化字符串（用于构建输出）
    ast.JoinedStr,      # f"..."
    ast.FormattedValue, # f"{x}"
}

# 允许的数学函数和白名单内置函数
_ALLOWED_FUNCTIONS = {
    # math 模块函数
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2',
    'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
    'sqrt', 'exp', 'log', 'log2', 'log10', 'pow',
    'ceil', 'floor', 'trunc', 'fabs', 'factorial',
    'degrees', 'radians', 'hypot',
    'gcd', 'lcm', 'perm', 'comb',
    'isclose', 'isfinite', 'isinf', 'isnan',
    'erf', 'erfc', 'gamma', 'lgamma',
    'copysign', 'fmod', 'frexp', 'ldexp', 'modf',
    'nextafter', 'ulp', 'remainder',
    'prod', 'dist',

    # math 常量
    'pi', 'e', 'tau', 'inf', 'nan',

    # 安全内置函数
    'abs', 'sum', 'min', 'max', 'round',
    'int', 'float', 'str', 'bool', 'complex',
    'len', 'range', 'enumerate', 'zip', 'reversed', 'sorted',
    'list', 'tuple', 'dict', 'set', 'frozenset',
    'map', 'filter', 'divmod', 'pow', 'hex', 'oct', 'bin',
    'chr', 'ord', 'repr', 'format', 'hash',
    'all', 'any', 'isinstance', 'issubclass',
    'type', 'id', 'callable', 'dir', 'vars',
    'iter', 'next', 'slice',
    'print',
}

# 允许在 Attribute 访问中出现的模块名
_ALLOWED_MODULES = {'math'}

# 数学问题检测关键词/模式
_MATH_KEYWORDS = [
    r'\d+\s*[\+\-\*/%\^]\s*\d+',         # 数字 运算符 数字
    r'[\+\-\*/%\^\(\)]',                  # 运算符
    r'\b(sin|cos|tan|sqrt|log|exp|abs|pow)\s*\(',  # 数学函数调用
    r'\b(sum|min|max|avg|average|mean)\b',          # 聚合函数
    r'\b(计算|算|等于|乘|除|加|减|平方|开方|立方|根号|乘以|除以|加上|减去)\b',  # 中文数学词
    r'\b(\d+)\s*(的|个)?\s*(倍|次方|次幂|平方|立方|根号)\b', # 中文数学描述
    r'[=＝]\s*\d',                        # = 号接数字（计算结果）
    r'\b(pi|π|e)\b',                      # 数学常量
    r'[\(]\d+[\+\-\*/]\d+[\)]',           # 括号内算式
    r'\d+!',                              # 阶乘
    r'\b\d+[eE][+-]?\d+\b',              # 科学计数法
]

# 编译正则
_MATH_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _MATH_KEYWORDS]


# ============================================================================
# AST 安全审计器
# ============================================================================

class ASTSecurityError(Exception):
    """AST 安全审计失败"""
    def __init__(self, message: str, node=None):
        self.node = node
        detail = f" (节点: {ast.dump(node) if node else 'N/A'})"
        super().__init__(message + detail)


class ASTSecurityAuditor(ast.NodeVisitor):
    """
    AST 安全审计器 —— 遍历 AST 树，检查所有节点是否在白名单内。
    
    对函数调用和属性访问进行额外检查，防止绕过白名单的攻击。
    """
    
    def __init__(self):
        self.violations = []
    
    def generic_visit(self, node):
        """检查节点类型是否在白名单中"""
        node_type = type(node)
        if node_type not in _ALLOWED_NODES:
            self.violations.append(
                ASTSecurityError(f"禁止的操作类型: {node_type.__name__}", node)
            )
        super().generic_visit(node)
    
    def visit_Call(self, node):
        """检查函数调用"""
        # 直接名称调用: sin(x), abs(x) → 检查函数名
        if isinstance(node.func, ast.Name):
            if node.func.id not in _ALLOWED_FUNCTIONS:
                self.violations.append(
                    ASTSecurityError(f"禁止的函数调用: {node.func.id}()", node)
                )
        # 属性调用: math.sin(x) → 检查 math 模块
        elif isinstance(node.func, ast.Attribute):
            if (isinstance(node.func.value, ast.Name) and
                node.func.value.id in _ALLOWED_MODULES and
                node.func.attr in _ALLOWED_FUNCTIONS):
                pass  # math.sin() 等是允许的
            else:
                self.violations.append(
                    ASTSecurityError(
                        f"禁止的属性调用: {ast.unparse(node.func)}()", node
                    )
                )
        # 复杂调用: getattr(...)(), 或 (lambda...)()
        else:
            self.violations.append(
                ASTSecurityError(f"禁止的动态调用: {ast.unparse(node.func)}()", node)
            )
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """检查属性访问 —— 只允许 math.xxx"""
        if isinstance(node.value, ast.Name):
            if node.value.id not in _ALLOWED_MODULES:
                self.violations.append(
                    ASTSecurityError(
                        f"禁止的属性访问: {node.value.id}.{node.attr}", node
                    )
                )
        else:
            # 禁止链式属性访问: a.b.c, obj.__class__ 等
            self.violations.append(
                ASTSecurityError(
                    f"禁止的复杂属性访问: {ast.unparse(node)}", node
                )
            )
        self.generic_visit(node)
    
    def visit_Import(self, node):
        """禁止所有 import 语句"""
        self.violations.append(
            ASTSecurityError(f"禁止导入模块: {' '.join(a.name for a in node.names)}", node)
        )
    
    def visit_ImportFrom(self, node):
        self.violations.append(
            ASTSecurityError(f"禁止导入模块: from {node.module}", node)
        )
    
    def visit_Global(self, node):
        self.violations.append(ASTSecurityError("禁止使用 global", node))
    
    def visit_Nonlocal(self, node):
        self.violations.append(ASTSecurityError("禁止使用 nonlocal", node))


# ============================================================================
# 计算沙盒
# ============================================================================

class ComputeSandbox:
    """
    安全 Python 计算沙盒。
    
    用法:
        sandbox = ComputeSandbox()
        result = sandbox.calculate("123 * 456 + sqrt(144)")
        print(result)  # "56088 + 12 = 56100" 或类似格式
        
        # 判断是否为数学问题
        if sandbox.is_math_question("123 * 456 等于多少"):
            ...
    """
    
    def __init__(self, timeout: int = 10, max_output_length: int = 4096):
        """
        Args:
            timeout: 执行超时（秒）
            max_output_length: 最大输出长度（字符数）
        """
        self.timeout = timeout
        self.max_output_length = max_output_length
        
        # __builtins__ 可能是 dict（脚本模式）或 module（模块模式）
        import builtins as _builtins_module
        
        # 构建安全内置函数字典
        self._safe_builtins = {}
        for name in _ALLOWED_FUNCTIONS:
            # math 模块属性和函数
            if hasattr(math, name):
                self._safe_builtins[name] = getattr(math, name)
            # Python 内置函数
            elif hasattr(_builtins_module, name):
                self._safe_builtins[name] = getattr(_builtins_module, name)
            # 安全辅助
            elif name == 'print':
                self._safe_builtins[name] = print
            elif name == 'pi':
                self._safe_builtins[name] = math.pi
            elif name == 'e':
                self._safe_builtins[name] = math.e
            elif name == 'tau':
                self._safe_builtins[name] = math.tau
            elif name == 'inf':
                self._safe_builtins[name] = math.inf
            elif name == 'nan':
                self._safe_builtins[name] = math.nan
    
    # ── 公开方法 ──────────────────────────────────────────────
    
    def is_math_question(self, query: str) -> bool:
        """
        判断输入是否为数学计算类问题。
        
        基于正则模式匹配，识别包含数字、运算符、数学函数的查询。
        
        Args:
            query: 用户输入文本
        
        Returns:
            是否为数学计算类问题
        """
        # 纯计算表达式：以数字或运算符开头
        stripped = query.strip()
        
        # 快速检查：如果由数字、运算符、空格、括号、中文数学词组成 → 数学
        math_chars = set('0123456789+-*/%^.()=＝,， 　\t\n')
        cn_math_words = {'乘', '除', '加', '减', '乘以', '除以', '加上', '减去',
                         '平方', '立方', '根号', '开方', '等于', '多少', '几', '计算', '算'}
        
        if all(c in math_chars or '\u4e00' <= c <= '\u9fff' for c in stripped if c not in ' \t\n\u3000'):
            has_digit = any(c.isdigit() for c in stripped)
            has_op = any(c in '+-*/%^' for c in stripped) or any(w in stripped for w in cn_math_words)
            if has_digit and has_op:
                return True
        
        # 匹配数学模式
        match_count = 0
        for pattern in _MATH_PATTERNS:
            if pattern.search(query):
                match_count += 1
        
        # 需要至少匹配 2 个模式（或匹配到强模式）
        # 强模式: 数字+运算符+数字, 数学函数调用
        strong_patterns = _MATH_PATTERNS[:2]  # 前两个是强模式
        has_strong = any(p.search(query) for p in strong_patterns)
        
        return match_count >= 1 or has_strong
    
    def calculate(self, expression: str) -> str:
        """
        安全计算数学表达式，返回结果字符串。
        
        会自动尝试两种解析模式:
          1. 提取算式部分（如 "123 * 456 = ?" → "123 * 456"）
          2. 直接作为 Python 表达式计算
        
        Args:
            expression: 数学表达式字符串
        
        Returns:
            计算结果字符串
        """
        # 提取纯数学表达式
        cleaned = self._extract_expression(expression)
        
        result = self.execute(cleaned)
        
        if result['status'] == 'success':
            return str(result['stdout']).strip()
        elif result['status'] == 'timeout':
            return f"计算超时（>{self.timeout}秒）"
        elif result['status'] == 'security_error':
            return f"该计算涉及不安全操作，已拒绝执行"
        else:
            return f"计算错误: {result.get('stderr', '未知错误')}"
    
    def execute(self, code: str) -> Dict:
        """
        在受限环境中安全执行 Python 代码。
        
        执行流程:
          1. AST 解析 + 安全审计
          2. 构建受限全局命名空间
          3. exec() 执行（带超时信号）
        
        Args:
            code: 要执行的 Python 代码
        
        Returns:
            {
                'status': 'success' | 'timeout' | 'security_error' | 'error',
                'stdout': str,     # 标准输出
                'stderr': str,     # 错误输出
            }
        """
        # ── 步骤 1: AST 安全审计 ──
        try:
            tree = ast.parse(code, mode='exec')
        except SyntaxError as e:
            return {
                'status': 'error',
                'stdout': '',
                'stderr': f'语法错误: {e.msg} (第{e.lineno}行, 第{e.offset}列)',
            }
        
        auditor = ASTSecurityAuditor()
        auditor.visit(tree)
        
        if auditor.violations:
            return {
                'status': 'security_error',
                'stdout': '',
                'stderr': '\n'.join(str(v) for v in auditor.violations),
            }
        
        # ── 步骤 2: 构建受限环境 ──
        safe_globals = {
            '__builtins__': self._safe_builtins,
            'math': math,
        }
        safe_locals = {}
        
        # ── 步骤 3: 捕获输出并执行 ──
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        # 超时信号
        def timeout_handler(signum, frame):
            raise TimeoutError(f"执行超时（>{self.timeout}秒）")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)
        
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(tree, '<sandbox>', 'exec'), safe_globals, safe_locals)
            
            stdout = stdout_buf.getvalue()
            stderr = stderr_buf.getvalue()
            
            if len(stdout) > self.max_output_length:
                stdout = stdout[:self.max_output_length] + "\n... (输出已截断)"
            
            # 如果表达式没有输出（如赋值的 intermediate 结果），
            # 尝试获取最后一个表达式的值
            if not stdout.strip():
                # 检查是否是单个表达式
                try:
                    expr_tree = ast.parse(code.strip(), mode='eval')
                    # 重新审计 eval 模式
                    auditor2 = ASTSecurityAuditor()
                    auditor2.visit(expr_tree)
                    if not auditor2.violations:
                        result = eval(compile(expr_tree, '<sandbox>', 'eval'), safe_globals, safe_locals)
                        stdout = str(result)
                except Exception:
                    pass
            
            return {
                'status': 'success',
                'stdout': stdout.strip() or '（无输出）',
                'stderr': stderr.strip(),
            }
            
        except TimeoutError as e:
            return {
                'status': 'timeout',
                'stdout': stdout_buf.getvalue()[:500],
                'stderr': str(e),
            }
        except Exception as e:
            return {
                'status': 'error',
                'stdout': stdout_buf.getvalue()[:500],
                'stderr': f'{type(e).__name__}: {e}',
            }
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    # ── 内部方法 ──────────────────────────────────────────────
    
    def _extract_expression(self, text: str) -> str:
        """
        从自然语言文本中提取纯数学表达式。
        
        例:
          "123 * 456 等于多少" → "123 * 456"
          "计算 sqrt(144) + 10" → "sqrt(144) + 10"
        """
        # 去掉常见的前缀后缀
        text = text.strip()
        
        # 移除中文疑问后缀
        for suffix in ['等于多少', '是多少', '=?', '=？', '多少', '几', '怎么算', '怎么计算']:
            if text.endswith(suffix):
                text = text[:-len(suffix)].strip()
        
        # 移除常见前缀
        for prefix in ['计算', '算一下', '求', '请计算', '帮我算']:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # 如果以 = 开头，去掉
        text = text.lstrip('=＝')
        
        # 移除末尾的 ?！
        text = text.rstrip('?？!！,，。.')
        
        # 中文运算符 → 符号
        replacements = {
            '乘': '*', '乘以': '*', '✕': '*', '×': '*',
            '除': '/', '除以': '/', '÷': '/',
            '加': '+', '加上': '+', '＋': '+',
            '减': '-', '减去': '-', '－': '-',
            '的平方': '**2', '平方': '**2',
            '的立方': '**3', '立方': '**3',
            '开方': 'sqrt', '开根号': 'sqrt',
            '％': '%',
        }
        for cn, op in replacements.items():
            text = text.replace(cn, op)
        
        # 处理 "根号 X" → "sqrt(X)"
        text = re.sub(r'\b根号\s*(\d+(?:\.\d+)?)', r'sqrt(\1)', text)
        # 处理 "根号(...)""  → "sqrt(...)"
        text = text.replace('根号(', 'sqrt(')
        
        return text.strip() or '0'


# ============================================================================
# 便捷函数
# ============================================================================

# 全局单例
_sandbox_instance = None

def get_sandbox(timeout: int = 10) -> ComputeSandbox:
    """获取计算沙盒单例"""
    global _sandbox_instance
    if _sandbox_instance is None or _sandbox_instance.timeout != timeout:
        _sandbox_instance = ComputeSandbox(timeout=timeout)
    return _sandbox_instance


# ============================================================================
# 自测
# ============================================================================

if __name__ == '__main__':
    sandbox = ComputeSandbox()
    
    tests = [
        ("基本算术", "123 * 456"),
        ("数学函数", "sqrt(144) + sin(0)"),
        ("幂运算", "2 ** 10"),
        ("复杂表达式", "(3.14 * 10 ** 2) / 2"),
        ("中文输入", "123 乘 456 等于多少"),
        ("中文输入2", "根号 144 加 10"),
        ("sum函数", "sum([1, 2, 3, 4, 5])"),
        ("列表推导", "[x**2 for x in range(1, 6)]"),
    ]
    
    print("=" * 60)
    print("🐉 龙珠计算沙盒 — 自测")
    print("=" * 60)
    
    for name, expr in tests:
        is_math = sandbox.is_math_question(expr)
        result = sandbox.calculate(expr)
        print(f"\n[{name}] {expr}")
        print(f"  是数学题: {is_math}")
        print(f"  结果: {result}")
    
    # 安全测试
    print("\n" + "=" * 60)
    print("🛡️ 安全性测试")
    print("=" * 60)
    
    dangerous = [
        "open('/etc/passwd')",
        "import os",
        "os.system('rm -rf /')",
        "__import__('os')",
        "().__class__.__bases__[0]",
    ]
    
    for code in dangerous:
        result = sandbox.execute(code)
        status_icon = "✅ 已拦截" if result['status'] == 'security_error' else "❌ 未拦截!"
        print(f"\n  {status_icon}: {code}")
        if result['status'] == 'security_error':
            print(f"    理由: {result['stderr'][:80]}...")
