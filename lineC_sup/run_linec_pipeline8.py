#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line C 第八阶段 (REVIEW_V2 2026-09-02): M1b focal alpha 扫描 + M6 base_pw 补种子.

作业 (串行, 与既有 pipeline 同配置):
  1. {mvtec,realiad}->visa focal_a025 seed1   (M1b: alpha=0.25, 正类 1/3x)
  2. {mvtec,realiad}->visa focal_a05  seed1   (M1b: alpha=0.5,  正类 1x)
  3. {mvtec,visa}->realiad base_pw seed2      (M6: 补齐单种子对照)
  4. {mvtec,visa}->realiad base_pw seed3      (M6)

预期: 1-2 仍塌缩 (P-F1<trivial 或 P-AUROC in 0.5±0.02) —— "VisA 3/3 塌缩"
不是 alpha 选档产物; 3-4 与 seed1 同型 (P-AUROC 高 / P-F1 低: ranking intact,
firing absent). 若 1-2 反而救活, M1 处置退回表述修正并改写 §4.3 论断.

状态写 pipeline_status8.txt; 日志 logs/_{train,eval}_linec_{tag}.log.
"""
import os
import re
import subprocess
import sys
import time

LINE_C_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(LINE_C_DIR, 'train.py')
EVAL_SCRIPT = os.path.join(LINE_C_DIR, 'eval.py')
LOGS = os.path.join(LINE_C_DIR, 'logs')
STATUS = os.path.join(LINE_C_DIR, 'pipeline_status8.txt')
PYTHON = r'C:\Users\max-11\miniconda3\python.exe'

QUEUE = [
    dict(train_sets=['mvtec', 'realiad'], test_set='visa',
         arm='focal_a025', seed=1),
    dict(train_sets=['mvtec', 'realiad'], test_set='visa',
         arm='focal_a05', seed=1),
    dict(train_sets=['mvtec', 'visa'], test_set='realiad',
         arm='base_pw', seed=2),
    dict(train_sets=['mvtec', 'visa'], test_set='realiad',
         arm='base_pw', seed=3),
]
EPOCHS, BS, LR = 5, 12, 1e-4
POOL = {('mvtec', 'realiad'): 92681, ('mvtec', 'visa'): 14746}
EVAL_MIN = {'visa': 45, 'realiad': 45}
SEC_PER_STEP = 0.096

os.makedirs(LOGS, exist_ok=True)


def est_arm_minutes(q):
    n = POOL[tuple(sorted(q['train_sets']))]
    return (n // BS) * SEC_PER_STEP * EPOCHS / 60 + EVAL_MIN[q['test_set']]


def set_status(msg):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(STATUS, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def run_stage(name, cmd, log_path):
    set_status(f'{name}: start')
    with open(log_path, 'w', encoding='utf-8', errors='replace') as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                cwd=LINE_C_DIR)
        rc = proc.wait()
    set_status(f'{name}: exit code {rc}, log: {log_path}')
    return rc


def log_has_error(log_path):
    try:
        with open(log_path, 'r', errors='ignore') as f:
            content = f.read()
    except Exception:
        return True
    return 'Traceback' in content or 'CUDA out of memory' in content


def main():
    total = sum(est_arm_minutes(q) for q in QUEUE)
    set_status(f'=== Line C pipeline-8 start: {len(QUEUE)} jobs '
               f'(R2 M1b alpha-scan + M6 base_pw seeds), ETA {total/60:.1f}h ===')
    t_start = time.time()
    for i, q in enumerate(QUEUE):
        seed = q['seed']
        tag = f'{"+".join(q["train_sets"])}_to_{q["test_set"]}_{q["arm"]}_seed{seed}'
        train_log = os.path.join(LOGS, f'_train_linec_{tag}.log')
        eval_log = os.path.join(LOGS, f'_eval_linec_{tag}.log')
        remain = sum(est_arm_minutes(x) for x in QUEUE[i:])
        eta = time.strftime('%H:%M', time.localtime(time.time() + remain * 60))
        set_status(f'JOB[{i+1}/{len(QUEUE)}] {tag}: est {est_arm_minutes(q):.0f}min, '
                   f'queue ETA ~{eta}')

        rc = run_stage(f'TRAIN[{tag}]',
                       [PYTHON, TRAIN_SCRIPT, '--train_sets', *q['train_sets'],
                        '--test_set', q['test_set'], '--arm', q['arm'],
                        '--epochs', str(EPOCHS), '--bs', str(BS),
                        '--lr', str(LR), '--seed', str(seed)], train_log)
        if rc != 0 or log_has_error(train_log):
            set_status(f'ABORT: training failed for {tag}, skip to next job.')
            continue
        if not re.search(r'FINISHED', open(train_log, errors='ignore').read()):
            set_status(f'ABORT: training log missing FINISHED for {tag}.')
            continue

        rc = run_stage(f'EVAL[{tag}]',
                       [PYTHON, EVAL_SCRIPT, '--train_sets', *q['train_sets'],
                        '--test_set', q['test_set'], '--arm', q['arm'],
                        '--seed', str(seed)], eval_log)
        if rc != 0 or log_has_error(eval_log):
            set_status(f'ABORT: eval failed for {tag}.')
            continue
        set_status(f'DONE[{tag}]: train + eval complete.')

    set_status(f'=== Line C pipeline-8 ALL DONE '
               f'(total {(time.time()-t_start)/3600:.1f}h) ===')
    return 0


if __name__ == '__main__':
    sys.exit(main())
