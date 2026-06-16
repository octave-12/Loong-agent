#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
龙珠计算沙盒测试 (test_compute.py)
===================================
测试 ComputeSandbox 的安全性、计算能力，以及
与 LoongPearl 主流程的集成。

测试场景:
  1. 基本算术: 123 * 456
  2. 数学函数: sqrt(144) + sin(0)
  3. 中文输入: "123 乘 456 等于多少"
  4. 安全性——非法代码: open('/etc/passwd'), import os
  5. 安全性——超时: while True: pass
  6. 龙珠集成: LoongPearl 查询数学题（不加载模型）
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loongpearl.utils.compute_sandbox import ComputeSandbox, ASTSecurityAuditor, ASTSecurityError


class TestComputeSandbox(unittest.TestCase):
    """计算沙盒单元测试"""

    def setUp(self):
        self.sandbox = ComputeSandbox(timeout=3)

    # ── 基本计算 ──────────────────────────────────────

    def test_basic_arithmetic(self):
        """基本算术: 123 * 456"""
        result = self.sandbox.calculate("123 * 456")
        self.assertIn("56088", result)

    def test_division(self):
        """除法: 100 / 3"""
        result = self.sandbox.calculate("100 / 3")
        self.assertIn("33.33", result)

    def test_power(self):
        """幂运算: 2 ** 10"""
        result = self.sandbox.calculate("2 ** 10")
        self.assertIn("1024", result)

    def test_modulo(self):
        """取模: 17 % 5"""
        result = self.sandbox.calculate("17 % 5")
        self.assertIn("2", result)

    # ── 数学函数 ──────────────────────────────────────

    def test_sqrt(self):
        """sqrt(144)"""
        result = self.sandbox.calculate("sqrt(144)")
        self.assertIn("12", result)

    def test_sin_zero(self):
        """sin(0)"""
        result = self.sandbox.calculate("sin(0)")
        self.assertIn("0", result)

    def test_math_combination(self):
        """sqrt(144) + sin(0) + cos(0)"""
        result = self.sandbox.calculate("sqrt(144) + sin(0) + cos(0)")
        self.assertIn("13", result)

    def test_log(self):
        """log(e) — 自然对数"""
        result = self.sandbox.calculate("log(math.e)")
        self.assertIn("1", result)

    def test_pi(self):
        """math.pi 常量"""
        result = self.sandbox.calculate("math.pi")
        self.assertIn("3.14", result)

    # ── 内置函数 ──────────────────────────────────────

    def test_sum(self):
        """sum([1, 2, 3, 4, 5])"""
        result = self.sandbox.calculate("sum([1, 2, 3, 4, 5])")
        self.assertIn("15", result)

    def test_abs(self):
        """abs(-42)"""
        result = self.sandbox.calculate("abs(-42)")
        self.assertIn("42", result)

    def test_min_max(self):
        """min + max"""
        result = self.sandbox.calculate("min(3, 1, 4, 1, 5) + max(1, 2, 3)")
        self.assertIn("4", result)

    def test_round(self):
        """round"""
        result = self.sandbox.calculate("round(3.14159, 2)")
        self.assertIn("3.14", result)

    # ── 中文输入 ──────────────────────────────────────

    def test_chinese_multiply(self):
        """中文: 123 乘 456 等于多少"""
        self.assertTrue(self.sandbox.is_math_question("123 乘 456 等于多少"))
        result = self.sandbox.calculate("123 乘 456 等于多少")
        self.assertIn("56088", result)

    def test_chinese_sqrt(self):
        """中文: 根号 144 加 10"""
        self.assertTrue(self.sandbox.is_math_question("根号 144 加 10"))
        result = self.sandbox.calculate("根号 144 加 10")
        self.assertIn("22", result)

    # ── 非数学问题不应误判 ──────────────────────────────

    def test_not_math_text(self):
        """普通文本不应识别为数学题"""
        self.assertFalse(self.sandbox.is_math_question("人工智能是什么"))
        self.assertFalse(self.sandbox.is_math_question("你好"))
        self.assertFalse(self.sandbox.is_math_question("今天天气怎么样"))

    # ── 安全性 —— 拒绝不安全代码 ────────────────────────

    def test_block_open(self):
        """禁止: open('/etc/passwd')"""
        result = self.sandbox.execute("open('/etc/passwd')")
        self.assertEqual(result['status'], 'security_error')

    def test_block_import(self):
        """禁止: import os"""
        result = self.sandbox.execute("import os")
        self.assertEqual(result['status'], 'security_error')

    def test_block_import_from(self):
        """禁止: from os import system"""
        result = self.sandbox.execute("from os import system")
        self.assertEqual(result['status'], 'security_error')

    def test_block_dunder_import(self):
        """禁止: __import__('os')"""
        result = self.sandbox.execute("__import__('os')")
        self.assertEqual(result['status'], 'security_error')

    def test_block_sandbox_escape(self):
        """禁止: ().__class__.__bases__ 沙盒逃逸"""
        result = self.sandbox.execute("().__class__.__bases__[0]")
        self.assertEqual(result['status'], 'security_error')

    def test_block_attribute_chain(self):
        """禁止: 链式属性访问逃逸"""
        result = self.sandbox.execute("[].__class__.__base__.__subclasses__()")
        self.assertEqual(result['status'], 'security_error')

    # ── 超时测试 ────────────────────────────────────────

    def test_timeout_infinite_loop(self):
        """超时: while True: pass"""
        result = self.sandbox.execute("while True: pass")
        self.assertEqual(result['status'], 'security_error',
                         msg="无限循环应该在 AST 阶段就被拦截")

    # ── 错误处理 ────────────────────────────────────────

    def test_syntax_error(self):
        """语法错误应该有明确提示"""
        result = self.sandbox.execute("123 +* 456")
        self.assertEqual(result['status'], 'error')
        self.assertIn("语法错误", result['stderr'])

    def test_runtime_error(self):
        """运行时错误（如除以零）"""
        result = self.sandbox.execute("1 / 0")
        self.assertIn(result['status'], ['error', 'success'])
        if result['status'] == 'error':
            self.assertIn("ZeroDivisionError", result['stderr'])


class TestLoongPearlIntegration(unittest.TestCase):
    """
    龙珠集成测试 —— 测试 LoongPearl 是否能正确识别并调度数学问题。
    注意: 不加载模型，只测试识别逻辑。
    """

    def test_sandbox_importable_from_engine(self):
        """引擎可以导入 ComputeSandbox"""
        from loongpearl.interaction.engine import LoongPearl
        # 创建一个未初始化的龙珠实例
        lp = LoongPearl()
        self.assertIsNotNone(lp.sandbox)
        self.assertIsInstance(lp.sandbox, ComputeSandbox)

    def test_math_detection_via_sandbox(self):
        """通过沙盒检测数学问题"""
        sandbox = ComputeSandbox()
        # 数学题
        self.assertTrue(sandbox.is_math_question("123 * 456"))
        self.assertTrue(sandbox.is_math_question("sqrt(144)"))
        # 非数学题
        self.assertFalse(sandbox.is_math_question("量子计算是什么"))
        self.assertFalse(sandbox.is_math_question("你好世界"))

    def test_query_result_format(self):
        """计算沙盒返回格式兼容 QueryResult"""
        from loongpearl.interaction.engine import LoongPearl
        lp = LoongPearl()
        
        # 模拟 query 方法中的计算沙盒分支
        sandbox = lp.sandbox
        question = "123 * 456"
        
        if sandbox.is_math_question(question):
            answer = sandbox.calculate(question)
            # 验证返回格式
            self.assertIsInstance(answer, str)
            self.assertIn("56088", answer)


if __name__ == '__main__':
    print("=" * 60)
    print("🐉 龙珠计算沙盒 — 测试套件")
    print("=" * 60)
    
    # 只运行核心测试，跳过集成测试（不需要加载模型）
    suite = unittest.TestLoader().loadTestsFromTestCase(TestComputeSandbox)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ 全部测试通过！")
    else:
        print(f"❌ {len(result.failures)} 失败, {len(result.errors)} 错误")
    print("=" * 60)
