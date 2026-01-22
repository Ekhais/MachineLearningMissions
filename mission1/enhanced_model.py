# enhanced_model.py
# 增强版模型：支持多种分类器和集成方法

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

def make_base_models(config: dict):
    """创建基础模型字典"""
    models = {}

    # Random Forest
    rf_params = config.get("rf_params", {})
    models["rf"] = RandomForestClassifier(
        n_estimators=rf_params.get("n_estimators", 300),
        max_depth=rf_params.get("max_depth", None),
        min_samples_split=rf_params.get("min_samples_split", 2),
        min_samples_leaf=rf_params.get("min_samples_leaf", 1),
        max_features=rf_params.get("max_features", "sqrt"),
        class_weight=rf_params.get("class_weight", "balanced"),
        random_state=rf_params.get("random_state", 0),
        n_jobs=-1
    )

    # Extra Trees (更随机的决策树集成)
    et_params = config.get("et_params", {})
    models["et"] = ExtraTreesClassifier(
        n_estimators=et_params.get("n_estimators", 300),
        max_depth=et_params.get("max_depth", None),
        min_samples_split=et_params.get("min_samples_split", 2),
        max_features=et_params.get("max_features", "sqrt"),
        class_weight=et_params.get("class_weight", "balanced"),
        random_state=et_params.get("random_state", 0),
        n_jobs=-1
    )

    # SVM with RBF kernel
    svm_params = config.get("svm_params", {})
    models["svm"] = SVC(
        kernel=svm_params.get("kernel", "rbf"),
        C=svm_params.get("C", 80.0),
        gamma=svm_params.get("gamma", "scale"),
        class_weight=svm_params.get("class_weight", "balanced"),
        probability=True,  # 启用概率估计以便集成
        random_state=0
    )

    # KNN
    knn_params = config.get("knn_params", {})
    models["knn"] = KNeighborsClassifier(
        n_neighbors=knn_params.get("n_neighbors", 5),
        weights=knn_params.get("weights", "distance"),
        metric=knn_params.get("metric", "euclidean"),
        n_jobs=-1
    )

    # Gradient Boosting
    gb_params = config.get("gb_params", {})
    models["gb"] = GradientBoostingClassifier(
        n_estimators=gb_params.get("n_estimators", 100),
        learning_rate=gb_params.get("learning_rate", 0.1),
        max_depth=gb_params.get("max_depth", 5),
        random_state=0
    )

    # AdaBoost
    ada_params = config.get("ada_params", {})
    models["ada"] = AdaBoostClassifier(
        n_estimators=ada_params.get("n_estimators", 100),
        learning_rate=ada_params.get("learning_rate", 1.0),
        random_state=0
    )

    # Logistic Regression
    lr_params = config.get("lr_params", {})
    models["logreg"] = LogisticRegression(
        C=lr_params.get("C", 1.0),
        max_iter=lr_params.get("max_iter", 1000),
        class_weight=lr_params.get("class_weight", "balanced"),
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1
    )

    return models

def make_model(config: dict):
    """根据配置返回单个模型或集成模型"""
    name = config.get("model_name", "rf").lower()

    # 集成模型
    if name == "voting" or name == "ensemble":
        return make_voting_ensemble(config)
    elif name == "stacking":
        return make_stacking_ensemble(config)

    # 单模型
    base_models = make_base_models(config)

    if name == "rf":
        return base_models["rf"]
    elif name == "et":
        return base_models["et"]
    elif name == "svm":
        svm_params = config.get("svm_params", {})
        return SVC(
            kernel=svm_params.get("kernel", "rbf"),
            C=svm_params.get("C", 80.0),
            gamma=svm_params.get("gamma", "scale"),
            class_weight=svm_params.get("class_weight", "balanced"),
            probability=svm_params.get("probability", False),
            random_state=0
        )
    elif name == "knn":
        return base_models["knn"]
    elif name == "gb":
        return base_models["gb"]
    elif name == "ada":
        return base_models["ada"]
    elif name == "logreg":
        return base_models["logreg"]
    else:
        # 默认返回随机森林
        return base_models["rf"]

def make_voting_ensemble(config: dict):
    """创建投票集成分类器"""
    ensemble_cfg = config.get("ensemble_params", {})
    voting_type = ensemble_cfg.get("voting", "soft")
    selected_models = ensemble_cfg.get("models", ["rf", "svm", "et"])

    base_models = make_base_models(config)

    estimators = []
    for name in selected_models:
        if name in base_models:
            estimators.append((name, base_models[name]))

    if len(estimators) == 0:
        # 默认组合
        estimators = [
            ("rf", base_models["rf"]),
            ("svm", base_models["svm"]),
            ("et", base_models["et"])
        ]

    weights = ensemble_cfg.get("weights", None)

    return VotingClassifier(
        estimators=estimators,
        voting=voting_type,
        weights=weights,
        n_jobs=-1
    )

def make_stacking_ensemble(config: dict):
    """创建Stacking集成分类器（使用sklearn的StackingClassifier）"""
    from sklearn.ensemble import StackingClassifier

    ensemble_cfg = config.get("ensemble_params", {})
    selected_models = ensemble_cfg.get("models", ["rf", "svm", "et"])

    base_models = make_base_models(config)

    estimators = []
    for name in selected_models:
        if name in base_models:
            estimators.append((name, base_models[name]))

    if len(estimators) == 0:
        estimators = [
            ("rf", base_models["rf"]),
            ("svm", base_models["svm"]),
            ("et", base_models["et"])
        ]

    # 元分类器（最终分类器）
    meta_clf = ensemble_cfg.get("meta_classifier", "logreg")
    if meta_clf == "logreg":
        final_estimator = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    elif meta_clf == "rf":
        final_estimator = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    else:
        final_estimator = LogisticRegression(max_iter=1000, n_jobs=-1)

    return StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        stack_method="auto",
        n_jobs=-1
    )

class HybridEnsemble:
    """
    混合集成分类器：结合Voting和Two-Stage Refinement思想
    - 对主分类器的高置信度预测保留
    - 对低置信度预测使用辅助分类器投票
    """
    def __init__(self, primary_clf, secondary_clfs, confidence_threshold=0.7):
        self.primary_clf = primary_clf
        self.secondary_clfs = secondary_clfs  # list of classifiers
        self.confidence_threshold = confidence_threshold

    def fit(self, X, y):
        print("Training primary classifier...")
        self.primary_clf.fit(X, y)
        print("Training secondary classifiers...")
        for clf in self.secondary_clfs:
            clf.fit(X, y)
        return self

    def predict(self, X):
        # 获取主分类器预测
        if hasattr(self.primary_clf, 'predict_proba'):
            proba = self.primary_clf.predict_proba(X)
            confidence = np.max(proba, axis=1)
            primary_preds = np.argmax(proba, axis=1)
        else:
            primary_preds = self.primary_clf.predict(X)
            confidence = np.ones(len(X))  # 无概率时全部高置信

        # 对低置信度样本进行投票
        low_conf_mask = confidence < self.confidence_threshold
        final_preds = primary_preds.copy()

        if np.any(low_conf_mask):
            X_low = X[low_conf_mask]

            # 收集所有分类器（包括主分类器）的投票
            all_preds = [primary_preds[low_conf_mask]]
            for clf in self.secondary_clfs:
                all_preds.append(clf.predict(X_low))

            # 多数投票
            all_preds = np.array(all_preds)  # (n_classifiers, n_low_conf)
            for i, idx in enumerate(np.where(low_conf_mask)[0]):
                votes = all_preds[:, i]
                unique, counts = np.unique(votes, return_counts=True)
                final_preds[idx] = unique[np.argmax(counts)]

        return final_preds

    def predict_proba(self, X):
        # 简单地返回主分类器的概率
        if hasattr(self.primary_clf, 'predict_proba'):
            return self.primary_clf.predict_proba(X)
        else:
            raise AttributeError("Primary classifier does not support predict_proba")

def make_hybrid_ensemble(config: dict):
    """创建混合集成分类器"""
    base_models = make_base_models(config)

    ensemble_cfg = config.get("ensemble_params", {})
    primary = ensemble_cfg.get("primary", "svm")
    secondary_names = ensemble_cfg.get("secondary", ["rf", "et", "knn"])
    threshold = ensemble_cfg.get("confidence_threshold", 0.7)

    primary_clf = base_models.get(primary, base_models["svm"])
    # 确保主分类器支持概率
    if primary == "svm":
        svm_params = config.get("svm_params", {})
        primary_clf = SVC(
            kernel=svm_params.get("kernel", "rbf"),
            C=svm_params.get("C", 80.0),
            gamma=svm_params.get("gamma", "scale"),
            class_weight=svm_params.get("class_weight", "balanced"),
            probability=True,
            random_state=0
        )

    secondary_clfs = [base_models[name] for name in secondary_names if name in base_models]

    return HybridEnsemble(primary_clf, secondary_clfs, threshold)

