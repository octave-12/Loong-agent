#!/bin/bash
# 龙珠 LoongPearl - 模型下载
# .pt files too large for GitHub (>100MB), download separately
set -e
D="$(cd "$(dirname "$0")" && pwd)"
echo "Download from: https://huggingface.co/octave-12/loong-pearl"
echo "  huggingface-cli download octave-12/loong-pearl zichang_94117_1024d.pt energy_landscape_1024d.pt --local-dir $D"
echo "Or build yourself: python zichang.py && python energy_landscape.py"
