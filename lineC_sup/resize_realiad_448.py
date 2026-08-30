#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Real-IAD 预缩放暂存: json 引用的全部图像 1024->448 (BILINEAR),
mask ->448 (NEAREST). 远程适配器加载时 resize(448,448) 恒等 => 与本机像素一致.
输出: realiad_448_staging/{cat}/{rel_path} (图像 jpg q95, mask png).
"""
import io
import json
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

SRC = r'D:\datasets\Real-IAD\extracted'
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'realiad_448_staging')
JSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'realiad_jsons')
SIZE = (448, 448)


def collect_jobs():
    jobs = []
    for jf in sorted(glob.glob(os.path.join(JSON_DIR, '*.json'))):
        cat = os.path.splitext(os.path.basename(jf))[0]
        with io.open(jf, encoding='utf-8-sig') as f:
            meta = json.load(f)
        for t in meta['train'] + meta['test']:
            ip = t.get('image_path')
            if ip:
                jobs.append((cat, ip, False))
            mp = t.get('mask_path')
            if mp:
                jobs.append((cat, mp, True))
    return jobs


def process(job):
    cat, rel, is_mask = job
    src = os.path.join(SRC, cat, rel.replace('/', os.sep))
    dst = os.path.join(DST, cat, rel.replace('/', os.sep))
    if os.path.exists(dst):
        return 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        if is_mask:
            m = Image.open(src).convert('L')
            if m.size != SIZE:
                m = m.resize(SIZE, Image.NEAREST)
            m.save(dst)
        else:
            im = Image.open(src).convert('RGB')
            if im.size != SIZE:
                im = im.resize(SIZE, Image.BILINEAR)
            im.save(dst, quality=95)
        return 1
    except Exception as e:
        return f'FAIL {src}: {e}'


def main():
    jobs = collect_jobs()
    print(f'total jobs: {len(jobs)}', flush=True)
    done = 0
    fails = []
    with ProcessPoolExecutor(max_workers=10) as ex:
        for r in ex.map(process, jobs, chunksize=64):
            if r == 1:
                done += 1
            else:
                fails.append(r)
            if done % 20000 == 0:
                print(f'progress {done}/{len(jobs)} fails={len(fails)}', flush=True)
    print(f'DONE {done}/{len(jobs)} fails={len(fails)}', flush=True)
    for x in fails[:20]:
        print(x, flush=True)


if __name__ == '__main__':
    main()
