#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line C 第七阶段看门狗: A5 超参敏感性扫描 (REVIEW_V1 A5).

OFAT 自基线 (m=0.3, r=5), 方向 {mvtec,realiad}->visa (hic 零塌缩且方差最紧的
方向, 对超参变化最敏感可测), seed 1, 4 作业:
  1. hic_m01  (margin 0.3 -> 0.1)
  2. hic_m05  (margin 0.3 -> 0.5)
  3. hic_r3   (halo radius 5 -> 3)
  4. hic_r8   (halo radius 5 -> 8)

变体在 losses.py HIC_VARIANTS 定义; 其余训练/评估配置与 hic 完全一致.
状态写 pipeline_status7.txt; 由 chained_p7.py supervisor 在 pipeline-6
(focal) 结束后自动启动.
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
STATUS = os.path.join(LINE_C_DIR, 'pipeline_status7.txt')
PYTHON = r'C:\Users\max-11\miniconda3\python.exe'

QUEUE = [dict(train_sets=['mvtec', 'realiad'], test_set='visa',
              arm=arm, seed=1)
         for arm in ('hic_m01', 'hic_m05', 'hic_r3', 'hic_r8')]
EPOCHS, BS, LR = 5, 12, 1e-4
POOL = {('mvtec', 'realiad'): 92681}
EVAL_MIN = {'visa': 18}
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


def gpu_busy():
    """psutil 检测任何在跑的 lineC 训练/评估或其他训练管线 (不依赖 NVML)."""
    try:
        import psutil
        me = os.getpid()
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if p.pid == me:
                    continue
                cl = ' '.join(p.info['cmdline'] or [])
                if ('lineC_sup' in cl and ('train.py' in cl or 'eval.py' in cl)):
                    return True, f'pid={p.pid} {cl[:100]}'
                if ('train_test.py' in cl and '--train' in cl) or \
                   ('run_lambda_ablation' in cl) or ('run_halo_pipeline' in cl):
                    return True, f'pid={p.pid} {cl[:100]}'
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False, ''
    except Exception as e:
        return True, f'psutil error: {e}'


def wait_gpu():
    set_status('WAIT_GPU: waiting for GPU ...')
    deadline = time.time() + 24 * 3600
    while time.time() < deadline:
        busy, info = gpu_busy()
        if not busy:
            time.sleep(60)
            busy2, _ = gpu_busy()
            if not busy2:
                set_status('WAIT_GPU: GPU free. Proceeding.')
                return True
        else:
            set_status(f'WAIT_GPU: busy ({info.splitlines()[0] if info else "?"}), sleep 120s')
        time.sleep(120)
    set_status('ERROR: WAIT_GPU timeout (24h). Abort.')
    return False


def run_stage(name, cmd, log_path):
    set_status(f'{name}: start: {" ".join(cmd)}')
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
    set_status(f'=== Line C pipeline-7 start: {len(QUEUE)} jobs '
               f'(A5 sensitivity OFAT), ETA {total/60:.1f}h ===')
    if not wait_gpu():
        return 1
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

    set_status(f'=== Line C pipeline-7 ALL DONE '
               f'(total {(time.time()-t_start)/3600:.1f}h) ===')
    return 0


if __name__ == '__main__':
    sys.exit(main())
