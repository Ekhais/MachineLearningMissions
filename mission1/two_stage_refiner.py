# two_stage_refiner.py
# TwoStageRefiner: 二阶段微调（主分类器 + 对指定/自动发现的易混类对训练二分类器）
# 依赖：numpy, sklearn, joblib, tqdm (tqdm 可选)
import os
import joblib
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class TwoStageRefiner:
    """
    TwoStageRefiner(config)
    config keys (示例见下方 yaml):
      mode: "auto" or "manual"          # auto: 自动选 top_k 混淆对；manual: 用 manual_pairs 列表
      top_k: 1                         # auto 模式选多少对
      manual_pairs: [["A","B"], ["C","D"]]   # manual 模式手工指定（用类名）
      binary_model: "rf" | "svm" | "logreg"
      binary_params: dict              # 传给 sklearn 模型的参数（例如 rf n_estimators 等）
      balance_method: "oversample" | "none"
      min_samples_for_pair: 20         # 若某对训练样本过少则跳过
      prob_threshold: 0.5              # 若 binary 模型支持 predict_proba，使用此阈值决定切换
      verbose: True
    """
    def __init__(self, config: dict):
        self.cfg = config.copy()
        self.mode = config.get("mode", "auto")
        self.top_k = int(config.get("top_k", 1))
        self.manual_pairs = config.get("manual_pairs", [])
        self.binary_model_name = config.get("binary_model", "rf")
        self.binary_params = config.get("binary_params", {})
        self.balance_method = config.get("balance_method", "oversample")
        self.min_samples_for_pair = int(config.get("min_samples_for_pair", 20))
        self.prob_threshold = float(config.get("prob_threshold", 0.5))
        self.verbose = bool(config.get("verbose", True))

        self.pair_models = {}   # key: (label_idx_a,label_idx_b) sorted tuple -> sklearn model
        self.pair_meta = {}     # info about pair -> { 'a_name','b_name','train_counts', 'val_metrics' }
        self.classes = None     # list of class names
        self.class_to_idx = None

    # -----------------------
    # internal: build classifier by name
    # -----------------------
    def _make_binary_clf(self):
        name = self.binary_model_name.lower()
        if name == "rf":
            params = {"n_estimators": 200, "random_state": 0, "n_jobs": -1}
            params.update(self.binary_params or {})
            return RandomForestClassifier(**params)
        elif name == "svm":
            params = {"kernel": "rbf", "probability": True}
            params.update(self.binary_params or {})
            return SVC(**params)
        elif name == "logreg":
            params = {"max_iter": 10000}
            params.update(self.binary_params or {})
            return LogisticRegression(**params)
        else:
            raise ValueError("Unknown binary_model: " + str(self.binary_model_name))

    # -----------------------
    # internal: oversample minority class by simple duplication
    # X: ndarray (n,d), y: ndarray (n,)
    # -----------------------
    def _oversample_balance(self, X, y):
        # only two classes expected in y
        uniq, counts = np.unique(y, return_counts=True)
        if len(uniq) <= 1:
            return X, y
        max_count = counts.max()
        Xs = []
        ys = []
        for cls in uniq:
            idx = np.where(y == cls)[0]
            Xi = X[idx]
            yi = y[idx]
            if len(idx) == 0:
                continue
            # how many to add
            need = max_count - len(idx)
            if need <= 0:
                Xs.append(Xi); ys.append(yi)
            else:
                # sample with replacement
                choice = np.random.choice(len(idx), need, replace=True)
                X_dup = Xi[choice]
                Xs.append(np.vstack([Xi, X_dup]))
                ys.append(np.hstack([yi, yi[choice]]))
        Xb = np.vstack(Xs)
        yb = np.hstack(ys)
        # shuffle
        perm = np.random.RandomState(0).permutation(len(yb))
        return Xb[perm], yb[perm]

    # -----------------------
    # detect confused pairs from validation confusion matrix
    # returns list of pairs (i,j) where i != j and confusion count high
    # -----------------------
    def _detect_top_confused_pairs(self, y_true, y_pred, classes, top_k=1):
        cm = confusion_matrix(y_true, y_pred)
        n = cm.shape[0]
        # we consider directed confusion counts cm[i,j] for i->j (true i predicted j)
        # convert to symmetric pair confusion score: cm[i,j] + cm[j,i]
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                score = cm[i,j] + cm[j,i]
                pairs.append(((i,j), int(score)))
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        top = [p for p,s in pairs[:top_k] if p is not None and s>0]
        if self.verbose:
            print("Top confused pairs (idx,score):", [(p, cnt) for p,cnt in pairs[:top_k]])
        return top

    # -----------------------
    # main fit interface
    # X_train_full, y_train_full: arrays (train full set used to train binary models)
    # X_val, y_val: arrays used to compute confusion and for optional val metrics (can be small)
    # main_val_preds: predictions of main model on validation set (used to compute confusion)
    # classes: list of class names (len = n_classes)
    # optionally manual_pairs list of [name1,name2] or can be auto-detected
    # -----------------------
    def fit(self, X_train_full, y_train_full, X_val, y_val, main_val_preds, classes, manual_pairs=None):
        """
        Train binary refiners based on confusion observed in validation.
        Parameters:
          - X_train_full (ndarray): features for training set (n_train, d)
          - y_train_full (ndarray): labels for training set (n_train,)
          - X_val, y_val (ndarray): validation features and true labels
          - main_val_preds (ndarray): predictions of main classifier on validation set
          - classes (list): class names ordered by index
          - manual_pairs (list of 2-tuples or name pairs): optional override of which pairs to train
        """
        self.classes = list(classes)
        self.class_to_idx = {c:i for i,c in enumerate(self.classes)}
        # determine pairs to train
        pairs_to_train = []
        if self.mode == "manual" or manual_pairs is not None:
            mp = manual_pairs if manual_pairs is not None else self.manual_pairs
            for p in mp:
                if isinstance(p[0], str):
                    a_name, b_name = p[0], p[1]
                    if a_name not in self.class_to_idx or b_name not in self.class_to_idx:
                        if self.verbose:
                            print("Warning: manual pair names not found in classes:", p)
                        continue
                    ia = self.class_to_idx[a_name]; ib = self.class_to_idx[b_name]
                else:
                    ia, ib = int(p[0]), int(p[1])
                if ia == ib: continue
                pairs_to_train.append(tuple(sorted((ia,ib))))
        else:
            auto_pairs = self._detect_top_confused_pairs(y_val, main_val_preds, self.classes, top_k=self.top_k)
            pairs_to_train = [tuple(sorted(p)) for p in auto_pairs]

        # unique and filter pairs with enough training samples
        pairs_to_train = sorted(set(pairs_to_train))
        if self.verbose:
            print("Pairs candidate to train (by idx):", pairs_to_train)

        for (ia, ib) in pairs_to_train:
            # build training subset where true label in {ia,ib}
            mask = np.logical_or(y_train_full == ia, y_train_full == ib)
            X_pair = X_train_full[mask]
            y_pair = y_train_full[mask]
            # require enough samples
            if len(y_pair) < self.min_samples_for_pair:
                if self.verbose:
                    print(f"Skipping pair {(ia,ib)}: only {len(y_pair)} samples (<{self.min_samples_for_pair})")
                continue
            # relabel to original indices (still use ia/ib labels)
            # optionally balance by oversampling minority
            if self.balance_method == "oversample":
                Xb, yb = self._oversample_balance(X_pair, y_pair)
            else:
                Xb, yb = X_pair, y_pair
            # train-test split for binary validation
            Xtr, Xte, ytr, yte = train_test_split(Xb, yb, stratify=yb, test_size=0.2, random_state=0)
            clf = self._make_binary_clf()
            # train
            clf.fit(Xtr, ytr)
            # val metrics
            ypred = clf.predict(Xte)
            f1 = f1_score(yte, ypred, average='macro')
            acc = accuracy_score(yte, ypred)
            rep = classification_report(yte, ypred, target_names=[self.classes[ia], self.classes[ib]], zero_division=0)
            # record model and metadata
            self.pair_models[(ia,ib)] = clf
            self.pair_meta[(ia,ib)] = {
                "a_idx": ia, "b_idx": ib,
                "a_name": self.classes[ia], "b_name": self.classes[ib],
                "train_count": int(len(y_pair)),
                "val_f1": float(f1), "val_acc": float(acc), "val_report": rep
            }
            if self.verbose:
                print(f"Trained binary model for pair ({self.classes[ia]}, {self.classes[ib]}) "
                      f"train_samples={len(y_pair)}, val_acc={acc:.3f}, val_f1={f1:.3f}")
        return self.pair_meta

    # -----------------------
    # predict/refine:
    # - main_preds: ndarray of predictions from main classifier (n_samples,)
    # - X: ndarray features
    # returns refined_preds (ndarray)
    # behavior: if main predicts class c and (c,other) pair exists, call binary classifier to possibly switch
    # If clf supports predict_proba, use threshold prob_threshold on probability of predicted label to decide override.
    # -----------------------
    def refine_predictions(self, X, main_preds):
        if self.classes is None:
            raise RuntimeError("TwoStageRefiner not initialized (no classes). Call fit(...) or load(...).")
        final_preds = main_preds.copy()
        n = len(main_preds)
        for i in range(n):
            mp = int(main_preds[i])
            # find pairs that include mp (we stored keys as sorted tuples)
            # there can be multiple pairs; we will check each where mp is a member
            related = [pair for pair in self.pair_models.keys() if mp in pair]
            if not related:
                continue
            # iterate through related pairs; if any classifier says the other class, override
            xi = X[i:i+1]
            overridden = False
            for pair in related:
                clf = self.pair_models[pair]
                ia, ib = pair
                # determine label mapping: clf predicts either ia or ib
                try:
                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(xi)[0]
                        # find index of class with max prob according to clf.classes_
                        # map clf.classes_ values to ints
                        class_indices = clf.classes_.astype(int)
                        # get predicted class by prob
                        pred_idx = class_indices[np.argmax(proba)]
                        pred_prob = np.max(proba)
                        if self.verbose and False:
                            print("proba", proba, "classes", class_indices)
                        # only override if predicted class != mp and prob > threshold
                        if int(pred_idx) != mp and pred_prob >= self.prob_threshold:
                            final_preds[i] = int(pred_idx)
                            overridden = True
                            break
                    else:
                        pred_idx = int(clf.predict(xi)[0])
                        if pred_idx != mp:
                            final_preds[i] = pred_idx
                            overridden = True
                            break
                except Exception as e:
                    # fallback: skip this pair if clf fails
                    if self.verbose:
                        print("Warning: pair clf predict failed:", pair, e)
                    continue
            # end for related
        return final_preds

    # -----------------------
    # shorthand: given main_model (sklearn-like) and X_test, do main.predict then refine
    # -----------------------
    def predict_with_main(self, main_model, X_test):
        main_preds = main_model.predict(X_test)
        return self.refine_predictions(X_test, main_preds)

    # -----------------------
    # persistence
    # -----------------------
    def save(self, prefix_dir):
        os.makedirs(prefix_dir, exist_ok=True)
        # save each pair model
        meta = {"cfg": self.cfg, "classes": self.classes, "pair_meta": self.pair_meta}
        joblib.dump(meta, os.path.join(prefix_dir, "two_stage_meta.joblib"))
        for pair, clf in self.pair_models.items():
            a,b = pair
            fname = os.path.join(prefix_dir, f"pair_{a}_{b}.joblib")
            joblib.dump(clf, fname)
        print("TwoStageRefiner saved to", prefix_dir)

    @classmethod
    def load(cls, prefix_dir, config=None):
        meta_path = os.path.join(prefix_dir, "two_stage_meta.joblib")
        if not os.path.isfile(meta_path):
            raise RuntimeError("two_stage_meta.joblib not found in " + prefix_dir)
        meta = joblib.load(meta_path)
        cfg = config or meta.get("cfg", {})
        obj = cls(cfg)
        obj.classes = meta.get("classes", None)
        obj.class_to_idx = {c:i for i,c in enumerate(obj.classes)} if obj.classes is not None else None
        obj.pair_meta = meta.get("pair_meta", {})
        # load pair models
        obj.pair_models = {}
        for fname in os.listdir(prefix_dir):
            if fname.startswith("pair_") and fname.endswith(".joblib"):
                parts = fname[:-7].split("_")  # pair_a_b
                try:
                    a = int(parts[1]); b = int(parts[2])
                    obj.pair_models[(a,b)] = joblib.load(os.path.join(prefix_dir, fname))
                except Exception:
                    continue
        return obj
