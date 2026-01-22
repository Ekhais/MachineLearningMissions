# enhanced_preprocessor.py
# 增强版特征提取器: BoVW + SPM + 多种纹理与颜色特征
# 新增: GLCM纹理特征, Gabor滤波器, 形状特征(Hu矩), 多颜色空间直方图

import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.decomposition import PCA
import joblib

# -----------------------
# utils
# -----------------------
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Robust read supporting Chinese paths and odd extensions."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, flags)
        return img
    except Exception:
        return None

def resize_keep_aspect(img, side):
    h, w = img.shape[:2]
    if max(h, w) == side:
        top = (side - h) // 2
        bottom = side - h - top
        left = (side - w) // 2
        right = side - w - left
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    scale = side / float(max(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (side - nh) // 2
    bottom = side - nh - top
    left = (side - nw) // 2
    right = side - nw - left
    return cv2.copyMakeBorder(img_r, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

# -----------------------
# detector factory
# -----------------------
def make_detector(use_sift=True, n_orb=1000):
    if use_sift:
        if hasattr(cv2, "SIFT_create"):
            try:
                return cv2.SIFT_create()
            except Exception:
                pass
        if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SIFT_create"):
            try:
                return cv2.xfeatures2d.SIFT_create()
            except Exception:
                pass
    return cv2.ORB_create(nfeatures=n_orb)

# -----------------------
# descriptor extractor with coordinates
# -----------------------
def extract_descriptors_and_coords(img, detector, dense=False, step=16):
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    h, w = gray.shape
    if dense:
        kps = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                kps.append(cv2.KeyPoint(x + step / 2, y + step / 2, step))
        if not kps:
            return None, None
        kps, des = detector.compute(gray, kps)
        if des is None:
            return None, None
        coords = np.array([kp.pt for kp in kps], dtype=np.float32)
    else:
        kps, des = detector.detectAndCompute(gray, None)
        if des is None or len(kps) == 0:
            return None, None
        coords = np.array([kp.pt for kp in kps], dtype=np.float32)
    des = np.asarray(des)
    if des.dtype != np.float32:
        des = des.astype(np.float32)
    return des, coords

# -----------------------
# color histogram (HSV)
# -----------------------
def color_hist_hsv(img, mask=None, bins=(8, 8, 4)):
    if img is None:
        return np.zeros((bins[0] * bins[1] * bins[2],), dtype=np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_bins, s_bins, v_bins = bins
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [h_bins, s_bins, v_bins], [0, 180, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    if hist.sum() > 0:
        hist = hist / (np.linalg.norm(hist) + 1e-12)
    return hist

# -----------------------
# 新增: RGB颜色直方图
# -----------------------
def color_hist_rgb(img, bins=(8, 8, 8)):
    if img is None:
        return np.zeros((bins[0] * bins[1] * bins[2],), dtype=np.float32)
    r_bins, g_bins, b_bins = bins
    hist = cv2.calcHist([img], [0, 1, 2], None, [r_bins, g_bins, b_bins], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    if hist.sum() > 0:
        hist = hist / (np.linalg.norm(hist) + 1e-12)
    return hist

# -----------------------
# 新增: Lab颜色直方图
# -----------------------
def color_hist_lab(img, bins=(8, 8, 8)):
    if img is None:
        return np.zeros((bins[0] * bins[1] * bins[2],), dtype=np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l_bins, a_bins, b_bins = bins
    hist = cv2.calcHist([lab], [0, 1, 2], None, [l_bins, a_bins, b_bins], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float32)
    if hist.sum() > 0:
        hist = hist / (np.linalg.norm(hist) + 1e-12)
    return hist

# -----------------------
# 新增: 颜色矩特征 (均值、标准差、偏度)
# -----------------------
def color_moments(img):
    if img is None:
        return np.zeros(9, dtype=np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    features = []
    for i in range(3):
        channel = hsv[:, :, i].flatten().astype(np.float32)
        mean = np.mean(channel)
        std = np.std(channel)
        # 偏度
        if std > 0:
            skewness = np.mean(((channel - mean) / std) ** 3)
        else:
            skewness = 0.0
        features.extend([mean / 255.0, std / 255.0, skewness])
    return np.array(features, dtype=np.float32)

# -----------------------
# HOG feature (skimage)
# -----------------------
def hog_feature(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9):
    if img is None:
        return np.zeros((1,), dtype=np.float32)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    feat = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block, block_norm='L2-Hys', feature_vector=True)
    return np.asarray(feat, dtype=np.float32)

# -----------------------
# LBP feature (uniform)
# -----------------------
def lbp_histogram(img, P=8, R=1, method='uniform', bins=None):
    if img is None:
        nbins = (P + 2) if bins is None else bins
        return np.zeros((nbins,), dtype=np.float32)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    lbp = local_binary_pattern(gray, P, R, method=method)
    if bins is None:
        nbins = P + 2
    else:
        nbins = bins
    hist, _ = np.histogram(lbp.ravel(), bins=nbins, range=(0, nbins))
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist = hist / (np.linalg.norm(hist) + 1e-12)
    return hist

# -----------------------
# 新增: 多尺度LBP
# -----------------------
def multi_scale_lbp(img, scales=[(8, 1), (16, 2), (24, 3)]):
    """多尺度LBP特征"""
    if img is None:
        total_bins = sum([(p + 2) for p, r in scales])
        return np.zeros(total_bins, dtype=np.float32)

    features = []
    for P, R in scales:
        lbp_hist = lbp_histogram(img, P=P, R=R, method='uniform')
        features.append(lbp_hist)
    return np.concatenate(features).astype(np.float32)

# -----------------------
# 新增: GLCM纹理特征 (灰度共生矩阵)
# -----------------------
def glcm_features(img, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=64):
    """
    提取GLCM纹理特征: contrast, dissimilarity, homogeneity, energy, correlation, ASM
    """
    if img is None:
        n_features = len(distances) * len(angles) * 6
        return np.zeros(n_features, dtype=np.float32)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 量化到指定级别
    gray_quantized = (gray / 256.0 * levels).astype(np.uint8)
    gray_quantized = np.clip(gray_quantized, 0, levels - 1)

    try:
        glcm = graycomatrix(gray_quantized, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

        features = []
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        for prop in props:
            prop_values = graycoprops(glcm, prop)
            features.extend(prop_values.flatten())

        result = np.array(features, dtype=np.float32)
        # 归一化
        if np.linalg.norm(result) > 0:
            result = result / (np.linalg.norm(result) + 1e-12)
        return result
    except Exception:
        n_features = len(distances) * len(angles) * 6
        return np.zeros(n_features, dtype=np.float32)

# -----------------------
# 新增: Gabor滤波器特征
# -----------------------
def gabor_features(img, frequencies=[0.1, 0.2, 0.3, 0.4], n_theta=4):
    """
    提取Gabor滤波器响应的统计特征（均值和标准差）
    """
    if img is None:
        n_features = len(frequencies) * n_theta * 2  # 每个滤波器2个统计值
        return np.zeros(n_features, dtype=np.float32)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = gray.astype(np.float32) / 255.0

    features = []
    thetas = np.linspace(0, np.pi, n_theta, endpoint=False)

    for freq in frequencies:
        for theta in thetas:
            try:
                kernel = np.real(gabor_kernel(freq, theta=theta, sigma_x=3, sigma_y=3))
                filtered = ndi.convolve(gray, kernel, mode='constant')
                features.append(np.mean(filtered))
                features.append(np.std(filtered))
            except Exception:
                features.extend([0.0, 0.0])

    result = np.array(features, dtype=np.float32)
    if np.linalg.norm(result) > 0:
        result = result / (np.linalg.norm(result) + 1e-12)
    return result

# -----------------------
# 新增: Hu矩形状特征
# -----------------------
def hu_moments_feature(img):
    """
    提取Hu不变矩（7个值），对缩放、旋转、平移不变
    """
    if img is None:
        return np.zeros(7, dtype=np.float32)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 二值化获取形状
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    moments = cv2.moments(binary)
    hu_moments = cv2.HuMoments(moments).flatten()

    # 对Hu矩取对数变换（常用技巧，减小数值范围）
    hu_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-12)

    return hu_log.astype(np.float32)

# -----------------------
# 新增: 边缘方向直方图
# -----------------------
def edge_orientation_histogram(img, n_bins=18):
    """
    基于Sobel算子的边缘方向直方图
    """
    if img is None:
        return np.zeros(n_bins, dtype=np.float32)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Sobel梯度
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度方向和幅值
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    orientation = np.arctan2(sobely, sobelx)  # -pi to pi
    orientation = (orientation + np.pi) * 180 / (2 * np.pi)  # 0 to 180

    # 加权直方图
    hist, _ = np.histogram(orientation.flatten(), bins=n_bins, range=(0, 180), weights=magnitude.flatten())
    hist = hist.astype(np.float32)
    if hist.sum() > 0:
        hist = hist / (np.linalg.norm(hist) + 1e-12)

    return hist

# -----------------------
# 新增: 形状上下文特征（简化版）
# -----------------------
def shape_context_features(img):
    """
    提取简化的形状特征：面积比、周长、圆度等
    """
    if img is None:
        return np.zeros(8, dtype=np.float32)

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return np.zeros(8, dtype=np.float32)

    # 取最大轮廓
    cnt = max(contours, key=cv2.contourArea)

    features = []

    # 面积
    area = cv2.contourArea(cnt)
    features.append(area / (img.shape[0] * img.shape[1]))

    # 周长
    perimeter = cv2.arcLength(cnt, True)
    features.append(perimeter / (2 * (img.shape[0] + img.shape[1])))

    # 圆度 (4*pi*area / perimeter^2)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter ** 2)
    else:
        circularity = 0
    features.append(circularity)

    # 边界框宽高比
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / (h + 1e-6)
    features.append(aspect_ratio)

    # 矩形度 (area / bounding_rect_area)
    rect_area = w * h
    if rect_area > 0:
        extent = area / rect_area
    else:
        extent = 0
    features.append(extent)

    # 凸包面积比
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = area / hull_area
    else:
        solidity = 0
    features.append(solidity)

    # 等效直径
    equivalent_diameter = np.sqrt(4 * area / np.pi) / max(img.shape[0], img.shape[1])
    features.append(equivalent_diameter)

    # 轮廓点数量（归一化）
    features.append(len(cnt) / 1000.0)

    return np.array(features, dtype=np.float32)

# -----------------------
# spatial pyramid BoVW encoder
# -----------------------
def bovw_spm_hist(des, coords, kmeans, k, image_shape, levels=(0,)):
    H, W = image_shape[0], image_shape[1]
    if des is None or des.shape[0] == 0:
        total_bins = sum([(2 ** l) * (2 ** l) * k for l in levels])
        return np.zeros((total_bins,), dtype=np.float32)

    words = kmeans.predict(des.astype(np.float32))
    out_hist_parts = []

    for l in levels:
        gx = gy = 2 ** l
        cell_w = W / gx
        cell_h = H / gy
        weight = 1.0 / (2 ** (max(levels) - l + 1)) if l > 0 else 1.0 / (2 ** max(levels))

        for iy in range(gy):
            y0 = iy * cell_h
            y1 = (iy + 1) * cell_h
            for ix in range(gx):
                x0 = ix * cell_w
                x1 = (ix + 1) * cell_w
                mask = (coords[:, 0] >= x0) & (coords[:, 0] < x1) & (coords[:, 1] >= y0) & (coords[:, 1] < y1)
                if np.any(mask):
                    ww = words[mask]
                    hist, _ = np.histogram(ww, bins=np.arange(k + 1))
                    hist = hist.astype(np.float32) * weight
                    if hist.sum() > 0:
                        hist = hist / (np.linalg.norm(hist) + 1e-12)
                else:
                    hist = np.zeros((k,), dtype=np.float32)
                out_hist_parts.append(hist)
    return np.concatenate(out_hist_parts).astype(np.float32)

# -----------------------
# 增强版 Preprocessor
# -----------------------
class EnhancedPreprocessor:
    """
    增强版特征提取器，新增多种纹理和形状特征
    """
    def __init__(self, config: dict):
        self.cfg = config.copy()
        self.image_side = int(config.get("image_side", 256))
        self.use_sift = bool(config.get("use_sift", True))
        self.orb_features = int(config.get("orb_features", 1000))
        self.dense = bool(config.get("dense", False))
        self.step_dense = int(config.get("step_dense", 16))
        self.k = int(config.get("k", 150))
        self.sample_descs = int(config.get("sample_descs", 100000))
        self.random_state = int(config.get("random_state", 0))

        # BoVW features
        self.use_bovw = bool(config.get("use_bovw", True))
        self.spm_levels = tuple(config.get("spm_levels", (0,)))

        # Color features
        self.use_color = bool(config.get("use_color_hist", True))
        self.color_bins = tuple(config.get("color_bins", (8, 8, 4)))
        self.use_color_rgb = bool(config.get("use_color_rgb", False))
        self.use_color_lab = bool(config.get("use_color_lab", False))
        self.use_color_moments = bool(config.get("use_color_moments", True))

        # Texture features
        self.use_hog = bool(config.get("use_hog", True))
        self.hog_params = config.get("hog_params", {"pixels_per_cell": (16, 16), "cells_per_block": (2, 2), "orientations": 9})

        self.use_lbp = bool(config.get("use_lbp", True))
        self.lbp_params = config.get("lbp_params", {"P": 8, "R": 1, "method": "uniform", "bins": None})
        self.use_multi_scale_lbp = bool(config.get("use_multi_scale_lbp", False))

        self.use_glcm = bool(config.get("use_glcm", True))
        self.glcm_params = config.get("glcm_params", {"distances": [1, 2, 3], "angles": [0, 0.785, 1.57, 2.356], "levels": 64})

        self.use_gabor = bool(config.get("use_gabor", True))
        self.gabor_params = config.get("gabor_params", {"frequencies": [0.1, 0.2, 0.3, 0.4], "n_theta": 4})

        # Shape features
        self.use_hu_moments = bool(config.get("use_hu_moments", True))
        self.use_shape_context = bool(config.get("use_shape_context", True))
        self.use_edge_hist = bool(config.get("use_edge_hist", True))

        # PCA and scaler
        self.pca_dim = config.get("pca_dim", None)
        self.scaler_type = config.get("scaler", "normalize")

        # internal
        self.detector = make_detector(use_sift=self.use_sift, n_orb=self.orb_features)
        self.kmeans = None
        self.pca = None
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "normalize":
            self.scaler = Normalizer(norm='l2')
        else:
            self.scaler = None

    def collect_and_sample_descriptors(self, img_list):
        all_descs = []
        filtered = []
        for fp, lbl in tqdm(img_list, desc="Collect descriptors"):
            img = imread_unicode(fp)
            if img is None:
                continue
            img = resize_keep_aspect(img, self.image_side)
            des, coords = extract_descriptors_and_coords(img, self.detector, dense=self.dense, step=self.step_dense)
            if des is None:
                continue
            filtered.append((fp, lbl))
            all_descs.append(des)
        if len(all_descs) == 0:
            raise RuntimeError("No descriptors collected from train images.")
        stacked = np.vstack(all_descs).astype(np.float32)
        if stacked.shape[0] > self.sample_descs:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(stacked.shape[0], self.sample_descs, replace=False)
            sampled = stacked[idx]
        else:
            sampled = stacked
        return filtered, sampled

    def fit(self, img_list):
        if self.use_bovw:
            filtered, sampled = self.collect_and_sample_descriptors(img_list)
            self.kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=1000, random_state=self.random_state)
            print("Fitting kmeans on sampled descriptors (n=%d)..." % sampled.shape[0])
            self.kmeans.fit(sampled.astype(np.float32))
        else:
            filtered = [x for x in img_list]
        return filtered

    def encode_single(self, fp, mask=None):
        img = imread_unicode(fp)
        if img is None:
            return None
        img_res = resize_keep_aspect(img, self.image_side)

        parts = []

        # BoVW (with SPM)
        if self.use_bovw:
            des, coords = extract_descriptors_and_coords(img_res, self.detector, dense=self.dense, step=self.step_dense)
            if des is None:
                bovw_vec = np.zeros((self.k * sum([(2 ** l) * (2 ** l) for l in self.spm_levels]),), dtype=np.float32)
            else:
                bovw_vec = bovw_spm_hist(des, coords, self.kmeans, self.k, img_res.shape[:2], levels=self.spm_levels)
            parts.append(bovw_vec)

        # Color features
        if self.use_color:
            ch = color_hist_hsv(img_res, mask=mask, bins=self.color_bins)
            parts.append(ch)

        if self.use_color_rgb:
            ch_rgb = color_hist_rgb(img_res)
            parts.append(ch_rgb)

        if self.use_color_lab:
            ch_lab = color_hist_lab(img_res)
            parts.append(ch_lab)

        if self.use_color_moments:
            cm = color_moments(img_res)
            parts.append(cm)

        # HOG
        if self.use_hog:
            hog_p = self.hog_params
            hog_vec = hog_feature(img_res, pixels_per_cell=tuple(hog_p.get("pixels_per_cell", (16, 16))),
                                  cells_per_block=tuple(hog_p.get("cells_per_block", (2, 2))),
                                  orientations=int(hog_p.get("orientations", 9)))
            parts.append(hog_vec)

        # LBP
        if self.use_lbp:
            if self.use_multi_scale_lbp:
                lbp_vec = multi_scale_lbp(img_res)
            else:
                lbp_p = self.lbp_params
                lbp_vec = lbp_histogram(img_res, P=int(lbp_p.get("P", 8)), R=float(lbp_p.get("R", 1)),
                                        method=lbp_p.get("method", "uniform"), bins=lbp_p.get("bins", None))
            parts.append(lbp_vec)

        # GLCM
        if self.use_glcm:
            glcm_p = self.glcm_params
            distances = glcm_p.get("distances", [1, 2, 3])
            angles = glcm_p.get("angles", [0, np.pi/4, np.pi/2, 3*np.pi/4])
            levels = glcm_p.get("levels", 64)
            glcm_vec = glcm_features(img_res, distances=distances, angles=angles, levels=levels)
            parts.append(glcm_vec)

        # Gabor
        if self.use_gabor:
            gabor_p = self.gabor_params
            gabor_vec = gabor_features(img_res, frequencies=gabor_p.get("frequencies", [0.1, 0.2, 0.3, 0.4]),
                                       n_theta=gabor_p.get("n_theta", 4))
            parts.append(gabor_vec)

        # Hu moments
        if self.use_hu_moments:
            hu_vec = hu_moments_feature(img_res)
            parts.append(hu_vec)

        # Shape context
        if self.use_shape_context:
            shape_vec = shape_context_features(img_res)
            parts.append(shape_vec)

        # Edge orientation histogram
        if self.use_edge_hist:
            edge_vec = edge_orientation_histogram(img_res)
            parts.append(edge_vec)

        if len(parts) == 0:
            return None
        feat = np.concatenate(parts).astype(np.float32)
        return feat

    def transform_train(self, img_list_filtered):
        X_list = []
        y_list = []
        paths = []
        print("Encoding train images to feature vectors...")
        for fp, lbl in tqdm(img_list_filtered, desc="Encoding train"):
            vec = self.encode_single(fp)
            X_list.append(vec)
            y_list.append(lbl)
            paths.append(fp)

        first_nonnull = next((v for v in X_list if v is not None), None)
        if first_nonnull is None:
            raise RuntimeError("All encoded train features are empty.")
        feat_dim = first_nonnull.shape[0]
        print(f"Feature dimension: {feat_dim}")

        X = np.zeros((len(X_list), feat_dim), dtype=np.float32)
        for i, v in enumerate(X_list):
            if v is None:
                X[i] = np.zeros((feat_dim,), dtype=np.float32)
            else:
                X[i] = v
        y = np.array(y_list, dtype=np.int32)

        if self.pca_dim is not None and self.pca is None:
            print("Fitting PCA to reduce to dim =", self.pca_dim)
            self.pca = PCA(n_components=min(self.pca_dim, feat_dim), random_state=self.random_state)
            X = self.pca.fit_transform(X).astype(np.float32)

        if self.scaler is not None:
            X = self.scaler.fit_transform(X).astype(np.float32)
        return X, y, paths

    def transform_test(self, test_paths):
        X_list = []
        paths = []
        print("Encoding test images...")
        for fp in tqdm(test_paths, desc="Encoding test"):
            vec = self.encode_single(fp)
            X_list.append(vec)
            paths.append(fp)

        first_nonnull = next((v for v in X_list if v is not None), None)
        if first_nonnull is None:
            raise RuntimeError("All test encodings empty.")
        feat_dim = first_nonnull.shape[0]

        X = np.zeros((len(X_list), feat_dim), dtype=np.float32)
        for i, v in enumerate(X_list):
            if v is None:
                X[i] = np.zeros((feat_dim,), dtype=np.float32)
            else:
                X[i] = v

        if self.pca is not None:
            X = self.pca.transform(X).astype(np.float32)
        if self.scaler is not None:
            X = self.scaler.transform(X).astype(np.float32)
        return X, None, paths

    def fit_transform(self, img_list):
        filtered = self.fit(img_list)
        X, y, paths = self.transform_train(filtered)
        return X, y, paths

    def save(self, prefix_dir):
        os.makedirs(prefix_dir, exist_ok=True)
        joblib.dump(self.kmeans, os.path.join(prefix_dir, "kmeans.joblib"))
        if self.pca is not None:
            joblib.dump(self.pca, os.path.join(prefix_dir, "pca.joblib"))
        if self.scaler is not None:
            joblib.dump(self.scaler, os.path.join(prefix_dir, "scaler.joblib"))
        meta = {
            "image_side": self.image_side,
            "use_sift": self.use_sift,
            "dense": self.dense,
            "step_dense": self.step_dense,
            "k": self.k,
            "spm_levels": self.spm_levels,
            "pca_dim": self.pca_dim,
        }
        joblib.dump(meta, os.path.join(prefix_dir, "preprocessor_meta.joblib"))

    @classmethod
    def load(cls, prefix_dir, config):
        pre = cls(config)
        pre.kmeans = joblib.load(os.path.join(prefix_dir, "kmeans.joblib"))
        pca_path = os.path.join(prefix_dir, "pca.joblib")
        if os.path.exists(pca_path):
            pre.pca = joblib.load(pca_path)
        scaler_path = os.path.join(prefix_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            pre.scaler = joblib.load(scaler_path)
        return pre

