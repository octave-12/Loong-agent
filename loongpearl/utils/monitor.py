#!/usr/bin/env python3
"""龙珠训练实时监控 —— 每5秒刷新最新日志"""
import sys, os, time

LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "train_v4.log")

last_size = 0
while True:
    try:
        if os.path.exists(LOG):
            size = os.path.getsize(LOG)
            if size > last_size:
                with open(LOG, 'r') as f:
                    if last_size > 0:
                        f.seek(last_size)
                    new = f.read()
                    if new.strip():
                        print(new, end='', flush=True)
                last_size = size
        time.sleep(2)
    except KeyboardInterrupt:
        print("\n👋 监控停止")
        break
    except Exception as e:
        print(f"Error: {e}", flush=True)
        time.sleep(5)
