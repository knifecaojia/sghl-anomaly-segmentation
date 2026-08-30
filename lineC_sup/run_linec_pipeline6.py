#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line C 第六阶段看门狗: focal 对照臂 (REVIEW_V1 A2, 论文 Sec 2.2 断言的实验支撑).

作业 (串行, train 5ep + 全量评估, 与既有 pipeline 同配置仅改 arm):
  1-3. {visa,realiad}->mvtec focal seed{1,2,3}
  4-6. {mvtec,realiad}->visa focal seed{1,2,3}

focal = Dice + Binary Focal(γ=2, α=0.75), 其余与 base 完全一致 (losses.py).
裁决预期 (论文 Sec 2.2): 幅度类重加权 (无/线性/focal 非线性聚焦) 均不能
对抗塌缩 —— 若 3/3 种子仍塌缩则断言坐实; 若部分救活则断言需改写为
"能发射但无边界几何" (P-F1/AUPRO 仍应显著低于 hic).

状态写 pipeline_status6.txt; 日志 logs/_{train,eval}_linec_{tag}.log;
看门狗输出 logs/_linec_pipeline6.log.
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
STATUS = os.path.join(LINE_C_DIR, 'pipeline_status6.txt')
PYTHON = r'C:\Users\max-11\miniconda3\python.exe'

QUEUE = []
for seed in (1, 2, 3):
    QUEUE.append(dict(train_sets=['visa', 'realiad'], test_set='mvtec',
                      arm='focal', seed=seed))
for seed in (1, 2, 3):
    QUEUE.append(dict(train_sets=['mvtec', 'realiad'], test_set='visa',
                      arm='focal', seed=seed))
EPOCHS, BS, LR = 5, 12, 1e-4
# 修正池规模 (与 pipeline4/5 一致: 源域 test 异常入池)
POOL = {('realiad', 'visa'): 97653, ('mvtec', 'realiad'): 92681}
EVAL_MIN = {'mvtec': 15, 'visa': 18}
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
    """psutil 检测其他训练进程 (不依赖 NVML); 查询失败保守视为忙碌."""
    try:
        import psutil
        me = os.getpid()
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if p.pid == me:
                    continue
                if 'python' not in (p.info['name'] or '').lower():
                    continue
                cl = ' '.join(p.info['cmdline'] or [])
                if 'lineC_sup' in cl:
                    continue
                if ('train_test.py' in cl and '--train' in cl) or \
                   ('run_lambda_ablation' in cl) or ('run_halo_pipeline' in cl):
                    return True, f'pid={p.pid} {cl[:120]}'
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False, ''
    except Exception as e:
        return True, f'psutil error: {e}'


def wait_gpu():
    set_status('WAIT_GPU: waiting for GPU ...')
    deadline = time.time() + 12 * 3600
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
    set_status('ERROR: WAIT_GPU timeout (12h). Abort.')
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
    set_status(f'=== Line C pipeline-6 start: {len(QUEUE)} jobs '
               f'(focal arm A2), ETA {total/60:.1f}h ===')
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

    set_status(f'=== Line C pipeline-6 ALL DONE '
               f'(total {(time.time()-t_start)/3600:.1f}h) ===')
    return 0


if __name__ == '__main__':
    sys.exit(main())
