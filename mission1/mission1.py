# main_enhanced.py
"""
增强版pipeline：使用EnhancedPreprocessor和集成模型
运行方式：python main_enhanced.py [config_file]
默认使用 config_enhanced.yaml
"""

import os
import sys
import yaml
import joblib
import csv
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from enhanced_preprocessor import EnhancedPreprocessor, imread_unicode
from enhanced_model import make_model, make_hybrid_ensemble
from two_stage_refiner import TwoStageRefiner

# ---- 默认配置 ----
DEFAULT_CONFIG = {
    "dataset_root": "./dataset",
    "train_subdir": "train",
    "test_subdir": "test",
    "image_side": 512,
    "use_sift": True,
    "dense": True,
    "step_dense": 12,
    "k": 1200,
    "sample_descs": 700000,
    "random_state": 99,
    "model_name": "voting",
    "artifacts_dir": "artifacts",
    "results_dir": "results",
    "do_validation": True,
    "validation_size": 0.2,
    "validation_only": False,
    "save_validation_artifact": True,
    "template_csv": None,
    "output_predictions": "predictions.csv",
    "validation_output": "validation_results.csv"
}

def load_config(path="config_enhanced.yaml"):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        merged = DEFAULT_CONFIG.copy()
        merged.update(cfg)
        return merged
    return DEFAULT_CONFIG

def get_template_header(path):
    if path and os.path.isfile(path):
        with open(path, newline='', encoding='utf-8') as f:
            r = csv.reader(f)
            try:
                return next(r)
            except StopIteration:
                return None
    return None

def write_predictions_csv(out_path, paths, preds_idx, classes, template_header=None):
    if template_header and len(template_header) >= 2:
        header = [template_header[0], template_header[1]]
    else:
        header = ["ID", "Category"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for fp, p in zip(paths, preds_idx):
            fname = os.path.basename(fp)
            label = classes[int(p)]
            writer.writerow([fname, label])

def print_confusion_analysis(y_true, y_pred, classes):
    """打印混淆矩阵分析，找出最容易混淆的类对"""
    cm = confusion_matrix(y_true, y_pred)
    n = len(classes)

    print("\n=== 混淆分析 ===")
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((classes[i], classes[j], cm[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    print("Top混淆对 (真实类 -> 预测类: 数量):")
    for true_cls, pred_cls, count in pairs[:10]:
        print(f"  {true_cls} -> {pred_cls}: {count}")

    # 各类别F1
    print("\n各类别性能:")
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(n))
    for i, cls in enumerate(classes):
        print(f"  {cls}: P={p[i]:.3f}, R={r[i]:.3f}, F1={f1[i]:.3f} (n={support[i]})")

def run():
    # 加载配置
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config_enhanced.yaml"
    print(f"Loading config from: {config_file}")
    cfg = load_config(config_file)

    dataset_root = cfg["dataset_root"]
    train_dir = os.path.join(dataset_root, cfg["train_subdir"])
    test_dir = os.path.join(dataset_root, cfg["test_subdir"])
    os.makedirs(cfg["artifacts_dir"], exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    # 构建训练集列表
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    if not classes:
        raise RuntimeError("No classes found under train dir: " + train_dir)
    print(f"Found {len(classes)} classes: {classes}")

    mapping = {c: i for i, c in enumerate(classes)}
    img_list = []
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        for fn in sorted(os.listdir(cls_dir)):
            fp = os.path.join(cls_dir, fn)
            if os.path.isfile(fp):
                img_list.append((fp, mapping[cls]))

    print(f"Total training images: {len(img_list)}")

    # 使用增强版预处理器
    print("\n=== 特征提取 ===")
    pre = EnhancedPreprocessor(cfg)
    print("Fitting preprocessor and building vocabulary...")
    img_list_filtered = pre.fit(img_list)
    X_full, y_full, paths_full = pre.transform_train(img_list_filtered)
    print(f"Feature matrix shape: {X_full.shape}")

    # 本地验证
    refiner = None  # 初始化
    if cfg.get("do_validation", True):
        print("\n=== 验证阶段 ===")
        X_tr, X_val, y_tr, y_val, items_tr, items_val = train_test_split(
            X_full, y_full, img_list_filtered,
            test_size=cfg.get("validation_size", 0.2),
            stratify=y_full,
            random_state=cfg.get("random_state", 0)
        )

        normalizer = Normalizer(norm='l2')
        X_tr_n = normalizer.fit_transform(X_tr)
        X_val_n = normalizer.transform(X_val)

        # 创建并训练模型
        model = make_model(cfg)
        print(f"Training model: {cfg.get('model_name', 'rf')}...")
        model.fit(X_tr_n, y_tr)

        preds_val = model.predict(X_val_n)
        acc = accuracy_score(y_val, preds_val)
        macf1 = f1_score(y_val, preds_val, average='macro')

        print(f"\n[主模型] Validation Accuracy: {acc:.4f}")
        print(f"[主模型] Validation Macro-F1: {macf1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, preds_val, target_names=classes))

        # 混淆分析
        print_confusion_analysis(y_val, preds_val, classes)

        # 保存验证CSV
        val_out = os.path.join(cfg["results_dir"], cfg["validation_output"])
        write_predictions_csv(val_out, [p for p, _ in items_val], preds_val, classes,
                             template_header=get_template_header(cfg.get("template_csv")))
        print(f"\nSaved validation CSV to: {val_out}")

        if cfg.get("save_validation_artifact", True):
            joblib.dump(model, os.path.join(cfg["artifacts_dir"], "model_validation.joblib"))
            joblib.dump(normalizer, os.path.join(cfg["artifacts_dir"], "normalizer_validation.joblib"))

        # Two-stage refinement
        two_cfg = cfg.get("two_stage", None)
        refiner = None
        if two_cfg is not None:
            print("\n=== Two-Stage Refinement ===")
            refiner = TwoStageRefiner(two_cfg)
            pair_meta = refiner.fit(X_tr_n, y_tr, X_val_n, y_val, preds_val, classes)
            print("Pair meta:", pair_meta)

            refined_val_preds = refiner.refine_predictions(X_val_n, preds_val)
            acc_ref = accuracy_score(y_val, refined_val_preds)
            macf1_ref = f1_score(y_val, refined_val_preds, average='macro')

            print(f"\n[Refined] Validation Accuracy: {acc_ref:.4f}")
            print(f"[Refined] Validation Macro-F1: {macf1_ref:.4f}")
            print("\nRefined Classification Report:")
            print(classification_report(y_val, refined_val_preds, target_names=classes))

            # 保存refined结果
            val_ref_out = os.path.join(cfg["results_dir"], "validation_refined_" + cfg["validation_output"])
            write_predictions_csv(val_ref_out, [p for p, _ in items_val], refined_val_preds, classes,
                                 template_header=get_template_header(cfg.get("template_csv")))
            print(f"Saved refined validation CSV to: {val_ref_out}")

            # 保存refiner
            two_dir = os.path.join(cfg["artifacts_dir"], "two_stage")
            refiner.save(two_dir)

        if cfg.get("validation_only", False):
            print("\nValidation only mode. Exiting.")
            return

    # 全量训练
    print("\n=== 全量训练 ===")
    normalizer_full = Normalizer(norm='l2')
    X_full_n = normalizer_full.fit_transform(X_full)

    model_full = make_model(cfg)
    print(f"Training final model on full data ({len(X_full_n)} samples)...")
    model_full.fit(X_full_n, y_full)

    # 保存模型
    joblib.dump(pre.kmeans, os.path.join(cfg["artifacts_dir"], "kmeans.joblib"))
    joblib.dump(model_full, os.path.join(cfg["artifacts_dir"], "model_full.joblib"))
    joblib.dump(normalizer_full, os.path.join(cfg["artifacts_dir"], "normalizer_full.joblib"))
    joblib.dump(classes, os.path.join(cfg["artifacts_dir"], "classes.joblib"))
    pre.save(cfg["artifacts_dir"])
    print(f"Saved artifacts to: {cfg['artifacts_dir']}")

    # 如果有two-stage，在全量数据上重新训练
    refiner_full = None
    if cfg.get("do_validation", True) and cfg.get("two_stage") is not None:
        try:
            if refiner is not None:
                print("\nRetraining two-stage refiner on full data...")
                # 用全量训练数据重新训练pair models
                # 这里简化处理：使用验证阶段发现的pairs，但用全量数据训练
                full_preds = model_full.predict(X_full_n)
                refiner_full = TwoStageRefiner(cfg.get("two_stage", {}))
                # 使用之前发现的pairs
                if hasattr(refiner, 'pair_meta') and len(refiner.pair_meta) > 0:
                    manual_pairs = [[meta["a_name"], meta["b_name"]] for meta in refiner.pair_meta.values()]
                    refiner_full.fit(X_full_n, y_full, X_full_n, y_full, full_preds, classes, manual_pairs=manual_pairs)
                    two_dir = os.path.join(cfg["artifacts_dir"], "two_stage")
                    refiner_full.save(two_dir)
        except NameError:
            pass  # refiner was not defined

    # 测试集预测
    print("\n=== 测试集预测 ===")
    test_files = [os.path.join(test_dir, f) for f in sorted(os.listdir(test_dir))
                  if os.path.isfile(os.path.join(test_dir, f))]
    if not test_files:
        raise RuntimeError("No test images found in: " + test_dir)

    print(f"Encoding {len(test_files)} test images...")
    X_test, _, test_paths = pre.transform_test(test_files)
    X_test_n = normalizer_full.transform(X_test)

    preds_test = model_full.predict(X_test_n)

    # 应用two-stage refinement到测试集
    if cfg.get("two_stage") is not None and 'refiner_full' in locals():
        print("Applying two-stage refinement to test predictions...")
        preds_test = refiner_full.refine_predictions(X_test_n, preds_test)

    final_out = os.path.join(cfg["results_dir"], cfg["output_predictions"])
    write_predictions_csv(final_out, test_paths, preds_test, classes,
                         template_header=get_template_header(cfg.get("template_csv")))
    print(f"\nSaved final predictions to: {final_out}")
    print("\n=== 完成 ===")

if __name__ == "__main__":
    run()

