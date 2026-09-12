# -*- coding: utf-8 -*-
"""
model_manager.py
분류 모델(RandomForest)의 학습/저장/불러오기/예측을 담당합니다.

이 프로그램의 핵심 기능(학습/자동판정)의 정확도를 높이기 위해:
- 학습 시 이미지마다 데이터 증강(좌우/상하 반전, 회전, 밝기/대비, 노이즈, 블러)을
  적용하여 실제 현장 조건 변화에 강인하도록 학습 데이터를 확장합니다.
- 이상치(조명 편차 등)에 덜 민감한 RobustScaler를 사용합니다.
- 특징이 매우 많고(1800차원+) 학습 데이터는 상대적으로 적은 상황에서 과적합을
  줄이기 위해, 중요도가 높은 특징만 선택해서 최종 학습에 사용합니다
  (SelectFromModel 기반 특징 차원 축소).
- 판정 유형이 1개만 있어도 학습이 가능합니다.

[대용량/매일 학습 시나리오를 위한 속도 최적화]
- 이 프로그램은 매번 "누적된 전체 데이터"로 재학습합니다 (RandomForest는 일부
  데이터만 추가로 학습하는 게 불가능하기 때문). 데이터가 하루 수백~수천 장씩
  쌓이면 매번 전체를 다시 "특징 추출(+증강)"하는 비용이 가장 큰 병목이 됩니다.
  이를 해결하기 위해 이미지별로 추출한 특징 벡터를 캐시(model/feature_cache.pkl)에
  저장해두고, 다음 학습부터는 "이미 처리한 적 없는 새 이미지"만 실제로 특징을
  추출합니다. (정확도에는 영향이 없습니다 - 여전히 전체 데이터로 학습합니다)
- 또한 한 유형(class)의 이미지 수가 지나치게 많아지면(MAX_SAMPLES_PER_CLASS 초과)
  분류기 학습(fit) 자체의 시간이 계속 늘어나므로, 유형별로 최대 개수를 두고
  그 이상은 무작위로 표본을 추려 사용합니다. (오래된 이미지의 특징도 캐시에
  남아있으므로 다음에 다시 뽑히면 재사용됩니다)
"""

import os
import sys
import json
import pickle
import random
import numpy as np
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

from feature_extractor import (
    extract_features, extract_features_from_array, augment_variants,
    _load_image_any, FEATURE_VERSION,
)


def _resolve_app_data_dir():
    """실행 위치(CWD)에 의존하지 않는, 항상 동일한 저장 폴더를 반환합니다.

    이전에는 MODEL_DIR = "model" (상대경로)를 사용했는데, 이 경우 exe를 어떤
    폴더/바로가기에서 실행하느냐에 따라 실제 저장되는 위치가 매번 달라질 수
    있어 "학습한 모델을 다음 실행 때 기억하지 못하는" 문제의 원인이 됩니다.
    Windows에서는 %APPDATA%, 그 외 OS에서는 사용자 홈 디렉토리 하위의 고정
    경로를 사용해 항상 같은 곳에 저장/로드되도록 합니다.
    """
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.path.join(os.path.expanduser("~"), ".config")
    path = os.path.join(base, "DefectInspector")
    os.makedirs(path, exist_ok=True)
    return path


MODEL_DIR = os.path.join(_resolve_app_data_dir(), "model")
MODEL_FILE = os.path.join(MODEL_DIR, "classifier.pkl")
META_FILE = os.path.join(MODEL_DIR, "meta.json")
CACHE_FILE = os.path.join(MODEL_DIR, "feature_cache.pkl")

# 유형(클래스)당 학습에 사용할 최대 이미지 수. 이보다 많으면 무작위로 이만큼만
# 뽑아서 학습합니다 (재학습 소요 시간이 무한정 늘어나는 것을 방지).
# 필요시 늘리거나 줄일 수 있습니다.
MAX_SAMPLES_PER_CLASS = 2000


class ModelManager:
    def __init__(self):
        self.clf = None
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.trained_at = None
        self.train_count = 0
        self.cv_accuracy = None
        self.feature_dim_before = None
        self.feature_dim_after = None
        self.last_cache_hits = 0
        self.last_cache_misses = 0
        self.last_capped_classes = []  # 표본 수 제한이 적용된 유형 목록
        self.feature_cache = {}
        os.makedirs(MODEL_DIR, exist_ok=True)
        self.load()
        self._load_feature_cache()

    # ---------------- 특징 캐시 ----------------
    def _load_feature_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "rb") as f:
                    self.feature_cache = pickle.load(f)
            except Exception:
                self.feature_cache = {}

    def _save_feature_cache(self):
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(self.feature_cache, f)
        except Exception as e:
            print("특징 캐시 저장 실패:", e)

    def _prune_feature_cache(self, valid_paths):
        """더 이상 학습에 쓰이지 않는(목록에서 사라진) 이미지의 캐시를 정리해 캐시 파일이 무한정 커지지 않게 합니다."""
        stale = [p for p in self.feature_cache if p not in valid_paths]
        for p in stale:
            del self.feature_cache[p]

    def clear_feature_cache(self):
        self.feature_cache = {}
        self._save_feature_cache()

    # ---------------- 학습 ----------------
    def train(self, image_paths, labels, progress_cb=None, augment=True,
              max_per_class=MAX_SAMPLES_PER_CLASS):
        """
        image_paths: 이미지 경로 리스트
        labels: 동일 길이의 문자열 라벨(작업자 판정값) 리스트
        progress_cb: (idx, total) 진행률 콜백 (선택, 원본 이미지 기준 진행률)
        augment: True면 이미지마다 반전/회전/밝기/대비/노이즈/블러 변형을 추가로 학습에 사용
        max_per_class: 유형별 최대 학습 이미지 수 (None이면 제한 없음)
        """
        if len(image_paths) < 1:
            raise ValueError("학습을 위해서는 최소 1장 이상의 라벨링된 이미지가 필요합니다.")

        # ---- 유형별 표본 수 제한 (장기적으로 재학습 시간이 계속 늘어나는 것 방지) ----
        self.last_capped_classes = []
        if max_per_class:
            by_label = {}
            for p, l in zip(image_paths, labels):
                by_label.setdefault(l, []).append(p)
            new_paths, new_labels = [], []
            rng = random.Random(42)
            for l, plist in by_label.items():
                if len(plist) > max_per_class:
                    plist = rng.sample(plist, max_per_class)
                    self.last_capped_classes.append(l)
                new_paths.extend(plist)
                new_labels.extend([l] * len(plist))
            image_paths, labels = new_paths, new_labels

        feats = []
        expanded_labels = []
        total = len(image_paths)
        cache_hits = 0
        cache_misses = 0

        for i, (p, label) in enumerate(zip(image_paths, labels)):
            try:
                stat = os.stat(p)
            except OSError:
                if progress_cb:
                    progress_cb(i + 1, total)
                continue

            cache_entry = self.feature_cache.get(p)
            if (cache_entry is not None
                    and cache_entry.get("mtime") == stat.st_mtime
                    and cache_entry.get("size") == stat.st_size
                    and cache_entry.get("augment") == augment
                    and cache_entry.get("feature_version") == FEATURE_VERSION):
                variant_feats = cache_entry["features"]
                cache_hits += 1
            else:
                try:
                    img = _load_image_any(p)
                except Exception:
                    if progress_cb:
                        progress_cb(i + 1, total)
                    continue
                variants = augment_variants(img) if augment else [img]
                variant_feats = [extract_features_from_array(v) for v in variants]
                self.feature_cache[p] = {
                    "mtime": stat.st_mtime, "size": stat.st_size,
                    "augment": augment, "features": variant_feats,
                    "feature_version": FEATURE_VERSION,
                }
                cache_misses += 1

            for vf in variant_feats:
                feats.append(vf)
                expanded_labels.append(label)

            if progress_cb:
                progress_cb(i + 1, total)

        self.last_cache_hits = cache_hits
        self.last_cache_misses = cache_misses

        # 이번 학습에 쓰이지 않는 이미지의 캐시는 정리하고 저장
        self._prune_feature_cache(valid_paths=set(image_paths))
        self._save_feature_cache()

        if not feats:
            raise ValueError("학습 가능한 이미지를 하나도 읽지 못했습니다.")

        X = np.vstack(feats)

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(expanded_labels)

        self.scaler = RobustScaler()
        Xs = self.scaler.fit_transform(X)

        num_classes = len(self.label_encoder.classes_)
        n_samples = Xs.shape[0]
        self.feature_dim_before = Xs.shape[1]

        # ---- 특징 차원 축소 (과적합 방지) ----
        if num_classes >= 2 and self.feature_dim_before > 30:
            max_features = max(30, min(self.feature_dim_before, n_samples * 8))
            try:
                prelim = RandomForestClassifier(
                    n_estimators=200, random_state=42, n_jobs=-1,
                    class_weight="balanced",
                )
                prelim.fit(Xs, y)
                self.feature_selector = SelectFromModel(
                    prelim, max_features=max_features, threshold=-np.inf, prefit=True
                )
                Xsel = self.feature_selector.transform(Xs)
            except Exception:
                self.feature_selector = None
                Xsel = Xs
        else:
            self.feature_selector = None
            Xsel = Xs

        self.feature_dim_after = Xsel.shape[1]

        self.clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced" if num_classes > 1 else None,
            random_state=42,
            n_jobs=-1,
        )
        self.clf.fit(Xsel, y)

        # 교차검증 정확도 (클래스가 1개뿐이거나 데이터가 너무 적으면 생략)
        self.cv_accuracy = None
        if num_classes >= 2:
            try:
                min_class_count = min(np.bincount(y))
                cv_folds = min(5, int(min_class_count))
                if cv_folds >= 2:
                    scores = cross_val_score(self.clf, Xsel, y, cv=cv_folds)
                    self.cv_accuracy = float(np.mean(scores))
            except Exception:
                self.cv_accuracy = None

        self.trained_at = datetime.now().isoformat(timespec="seconds")
        self.train_count = len(image_paths)
        self.save()
        return self.cv_accuracy

    # ---------------- 예측 ----------------
    def predict(self, image_path):
        """단일 이미지에 대해 (예측라벨, 신뢰도[0~1]) 반환. 모델 미학습시 (None, 0.0)"""
        if self.clf is None:
            return None, 0.0
        feat = extract_features(image_path).reshape(1, -1)
        feat_s = self.scaler.transform(feat)
        if self.feature_selector is not None:
            feat_s = self.feature_selector.transform(feat_s)
        proba = self.clf.predict_proba(feat_s)[0]
        idx = int(np.argmax(proba))
        label = str(self.label_encoder.inverse_transform([idx])[0])
        confidence = float(proba[idx])
        return label, confidence

    def is_trained(self):
        return self.clf is not None

    # ---------------- 저장/불러오기 ----------------
    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(MODEL_FILE, "wb") as f:
            pickle.dump({
                "clf": self.clf,
                "scaler": self.scaler,
                "label_encoder": self.label_encoder,
                "feature_selector": self.feature_selector,
            }, f)
        with open(META_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "trained_at": self.trained_at,
                "train_count": self.train_count,
                "cv_accuracy": self.cv_accuracy,
                "classes": list(self.label_encoder.classes_) if self.label_encoder else [],
                "feature_dim_before": self.feature_dim_before,
                "feature_dim_after": self.feature_dim_after,
            }, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, "rb") as f:
                    d = pickle.load(f)
                self.clf = d.get("clf")
                self.scaler = d.get("scaler")
                self.label_encoder = d.get("label_encoder")
                self.feature_selector = d.get("feature_selector")
                if os.path.exists(META_FILE):
                    with open(META_FILE, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    self.trained_at = meta.get("trained_at")
                    self.train_count = meta.get("train_count", 0)
                    self.cv_accuracy = meta.get("cv_accuracy")
                    self.feature_dim_before = meta.get("feature_dim_before")
                    self.feature_dim_after = meta.get("feature_dim_after")
            except Exception:
                # 손상된 모델 파일은 무시
                self.clf = None
                self.scaler = None
                self.label_encoder = None
                self.feature_selector = None
