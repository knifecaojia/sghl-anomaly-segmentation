#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Supervisor: 等 pipeline-6 (focal) 进程退出后自动启动 pipeline-7 (A5 扫描).

用途: 串行化两条本机 GPU 队列, 避免看门狗 gap 竞争。输出进 logs/_chained_p7.log。
"""
import os
import subprocess
import sys
import time

import psutil

HERE = os.path.dirname(os.path.abspath(__file__))
MARKER = 'run_linec_pipeline6.py'


def marker_alive():
    me = os.getpid()
    for p in psutil.process_iter(['pid', 'cmdline']):
        try:
            if p.pid == me:
                continue
            cl = ' '.join(p.info['cmdline'] or [])
            if MARKER in cl:
                return p.pid
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def log(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)


def main():
    pid = marker_alive()
    log(f'supervisor start: pipeline-6 watcher pid={pid}; waiting for exit.')
    while True:
        pid = marker_alive()
        if pid is None:
            log('pipeline-6 gone. 60s grace, then start pipeline-7.')
            time.sleep(60)
            if marker_alive() is not None:
                log('pipeline-6 reappeared (restart?); keep waiting.')
                continue
            break
        time.sleep(120)
    log('launching pipeline-7 (A5 sensitivity OFAT).')
    rc = subprocess.run(
        [sys.executable, os.path.join(HERE, 'run_linec_pipeline7.py')]).returncode
    log(f'pipeline-7 exit code {rc}. supervisor done.')
    return rc


if __name__ == '__main__':
    sys.exit(main())
