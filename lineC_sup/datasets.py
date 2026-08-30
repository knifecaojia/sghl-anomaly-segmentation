#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Line C 跨数据集通用缺陷分割 —— 统一数据集适配器.

三个数据集统一为样本列表接口, 每条样本:
    dict(image_path, mask_path(None=正常), label(0/1), dataset_id, category)

- MVTecAdapter / VisAAdapter: 标准 MVTec 风格目录
    root/{category}/train/good/*.png|JPG                 (正常训练图)
    root/{category}/test/{defect}/*                      (测试图, good=正常)
    root/{category}/ground_truth/{defect}/{stem}_mask.png (像素级 GT)
  VisA (mvtec_style) 的缺陷类目录统一为 'anomaly'.

- RealIADAdapter: 官方 JSON 元数据 (本目录 realiad_jsons/{category}.json)
    train split = 全部 OK 正常图 (官方 train 无异常);
    训练用异常 = test split 中带 mask_path 的条目 (Real-IAD 仅测试集有像素标注);
    eval 用完整 test split (正常+异常), 与 EVAL_PROTOCOL 一致.

图像加载/变换在 LineCDataset (torch Dataset) 内完成:
    resize 448 -> center crop 392, ImageNet 归一化; mask 同几何变换, 二值化 (>0).
"""
import glob
import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP')

MVTEC_ROOT = r'D:\datasets\MVTec_AD\extracted'
VISA_ROOT = r'D:\datasets\VisA\mvtec_style'
REALIAD_ROOT = r'D:\datasets\Real-IAD\extracted'
REALIAD_JSON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'realiad_jsons')

IMAGE_SIZE = 448
CROP_SIZE = 392
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _list_images(d):
    # Windows glob 大小写不敏感, *.png 与 *.PNG 会重复命中同一文件 -> 用 set 去重
    files = set()
    for ext in IMG_EXTS:
        files.update(glob.glob(os.path.join(d, f'*{ext}')))
    return sorted(files)


class MVTecStyleAdapter:
    """MVTec / VisA(mvtec_style) 通用适配器."""

    def __init__(self, root, dataset_id):
        self.root = root
        self.dataset_id = dataset_id

    def categories(self):
        return sorted(d for d in os.listdir(self.root)
                      if os.path.isdir(os.path.join(self.root, d)))

    def train_samples(self):
        """训练池 = train/good 正常图 + test split 中带掩码的异常图.

        [2026-08-15 BUGFIX] 此前只返回 train/good 正常图, 漏掉了 MVTec/VisA 唯一
        有像素标注的异常来源 (test split), 导致 {MVTec+VisA}->Real-IAD 方向训练池
        零异常样本. 设计文档: "训练数据 = N-1 个数据集的全部真实异常标注 + 正常图".
        LODO 无泄漏: 评测数据集整体不在训练池内.
        """
        samples = []
        for cat in self.categories():
            for p in _list_images(os.path.join(self.root, cat, 'train', 'good')):
                samples.append(dict(image_path=p, mask_path=None, label=0,
                                    dataset_id=self.dataset_id, category=cat))
        for s in self.test_samples():
            if s['label'] == 1 and s['mask_path']:
                samples.append(s)
        return samples

    def test_samples(self):
        """完整官方测试集: test/{cls}/*, ground_truth/{cls}/{stem}_mask.*"""
        samples = []
        for cat in self.categories():
            test_dir = os.path.join(self.root, cat, 'test')
            gt_dir = os.path.join(self.root, cat, 'ground_truth')
            for cls in sorted(os.listdir(test_dir)):
                for p in _list_images(os.path.join(test_dir, cls)):
                    stem = os.path.splitext(os.path.basename(p))[0]
                    mask = None
                    if cls != 'good':
                        cand = os.path.join(gt_dir, cls, stem + '_mask.png')
                        if not os.path.exists(cand):  # VisA 偶有 .JPG 对应 mask
                            g = glob.glob(os.path.join(gt_dir, cls, stem + '_mask.*'))
                            cand = g[0] if g else None
                        mask = cand
                    samples.append(dict(image_path=p, mask_path=mask,
                                        label=0 if cls == 'good' else 1,
                                        dataset_id=self.dataset_id, category=cat,
                                        defect=cls))
        return samples


class RealIADAdapter:
    """Real-IAD 适配器 (官方 JSON 元数据驱动)."""

    dataset_id = 'realiad'

    def __init__(self, root=REALIAD_ROOT, json_dir=REALIAD_JSON_DIR):
        self.root = root
        self.json_dir = json_dir

    def categories(self):
        return sorted(os.path.splitext(f)[0] for f in os.listdir(self.json_dir)
                      if f.endswith('.json'))

    def _load(self, cat):
        with open(os.path.join(self.json_dir, cat + '.json'), 'r', encoding='utf-8-sig') as f:
            return json.load(f)

    def _abs(self, cat, rel):
        return os.path.join(self.root, cat, rel.replace('/', os.sep))

    def train_samples(self):
        """正常图取官方 train split (全 OK); 异常图取 test split 中带掩码的条目
        (Real-IAD 仅测试集有像素级标注; LODO 下评测集为另一数据集, 无泄漏)."""
        samples = []
        for cat in self.categories():
            meta = self._load(cat)
            for t in meta['train']:
                samples.append(dict(image_path=self._abs(cat, t['image_path']),
                                    mask_path=None, label=0,
                                    dataset_id=self.dataset_id, category=cat))
            for t in meta['test']:
                if t['anomaly_class'] != 'OK' and t.get('mask_path'):
                    samples.append(dict(image_path=self._abs(cat, t['image_path']),
                                        mask_path=self._abs(cat, t['mask_path']),
                                        label=1, dataset_id=self.dataset_id,
                                        category=cat, defect=t['anomaly_class']))
        return samples

    def test_samples(self):
        """完整官方 test split."""
        samples = []
        for cat in self.categories():
            meta = self._load(cat)
            for t in meta['test']:
                mask = t.get('mask_path')
                samples.append(dict(image_path=self._abs(cat, t['image_path']),
                                    mask_path=self._abs(cat, mask) if mask else None,
                                    label=0 if t['anomaly_class'] == 'OK' else 1,
                                    dataset_id=self.dataset_id, category=cat,
                                    defect=t['anomaly_class']))
        return samples


def get_adapter(dataset_id):
    if dataset_id == 'mvtec':
        return MVTecStyleAdapter(MVTEC_ROOT, 'mvtec')
    if dataset_id == 'visa':
        return MVTecStyleAdapter(VISA_ROOT, 'visa')
    if dataset_id == 'realiad':
        return RealIADAdapter()
    raise ValueError(f'unknown dataset_id: {dataset_id}')


class LineCDataset(Dataset):
    """统一 (image, mask, label, dataset_id, category) 输出的 torch Dataset.

    返回:
        image: (3,392,392) float, ImageNet 归一化
        mask:  (1,392,392) float 0/1 (正常图为全零)
        label: int 0/1
        dataset_id: str, category: str
    """

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        img = Image.open(path).convert('RGB')
        img = TF.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
        img = TF.center_crop(img, [CROP_SIZE, CROP_SIZE])
        return TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)

    def _load_mask(self, path):
        if path is None:
            return torch.zeros(1, CROP_SIZE, CROP_SIZE)
        m = Image.open(path).convert('L')
        m = TF.resize(m, [IMAGE_SIZE, IMAGE_SIZE], interpolation=TF.InterpolationMode.NEAREST)
        m = TF.center_crop(m, [CROP_SIZE, CROP_SIZE])
        t = TF.to_tensor(m)
        return (t > 0).float()

    def __getitem__(self, idx):
        s = self.samples[idx]
        return dict(image=self._load_image(s['image_path']),
                    mask=self._load_mask(s['mask_path']),
                    label=s['label'], dataset_id=s['dataset_id'],
                    category=s['category'], image_path=s['image_path'])


def collate_linec(batch):
    return dict(image=torch.stack([b['image'] for b in batch]),
                mask=torch.stack([b['mask'] for b in batch]),
                label=torch.tensor([b['label'] for b in batch]),
                dataset_id=[b['dataset_id'] for b in batch],
                category=[b['category'] for b in batch],
                image_path=[b['image_path'] for b in batch])


def build_train_pool(dataset_ids):
    """pooled 训练集: 各数据集全部真实异常标注 + 正常图."""
    pool = []
    for did in dataset_ids:
        pool.extend(get_adapter(did).train_samples())
    return pool


def build_test_set(dataset_id):
    return get_adapter(dataset_id).test_samples()


def dataset_balanced_weights(samples):
    """数据集级平衡采样权重: 每个样本权重 = 1/其数据集样本数 (各数据集等概率)."""
    counts = {}
    for s in samples:
        counts[s['dataset_id']] = counts.get(s['dataset_id'], 0) + 1
    return torch.DoubleTensor([1.0 / counts[s['dataset_id']] for s in samples])
