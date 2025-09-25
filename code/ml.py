# ml.py  —— deploy-friendly toolbox (multiclass version)
from __future__ import annotations
import os
import json
import joblib
import numpy as np
import pandas as pd
import pywt
from typing import Dict, List, Tuple, Optional

from scipy.stats import skew, kurtosis, entropy
from scipy.fft import fft, fftfreq

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---------- 常數 ----------
FEATURE_COLS: List[str] = [
    "RMS","Skewness","Kurtosis","Entropy","CrestFactor",
    "Freq_Center","RMSF","Spectral_Kurtosis","Spectral_Entropy",
    "Wavelet_D3_Kurtosis","Impulse_Factor","Margin_Factor",
    "Approximate_Entropy","Spectral_Energy","Clearance_Factor",
]
DIRECTIONS: Tuple[str, ...] = ("Xa","Xb","Ya","Yb")

# ---------- I/O ----------
def read_one_txt(file, min_length: int = 18000):
    """讀取單一 txt 檔案，可以處理路徑字串或 Streamlit UploadedFile"""
    import numpy as np, io
    
    if isinstance(file, str):  # 路徑字串
        filename = file
        if not filename.endswith(".txt"):
            return None
        try:
            arr = np.loadtxt(file, skiprows=1, dtype=float)
            if arr.shape[0] < min_length:
                return None
            return arr
        except Exception:
            return None
    
    else:  # Streamlit UploadedFile
        filename = file.name
        if not filename.endswith(".txt"):
            return None
        try:
            arr = np.loadtxt(io.StringIO(file.getvalue().decode("utf-8")),
                             skiprows=1, dtype=float)
            if arr.shape[0] < min_length:
                return None
            return arr
        except Exception:
            return None



def read_split(data_root: str, split: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    split_dir = os.path.join(data_root, split)
    for direction in DIRECTIONS:
        dir_path = os.path.join(split_dir, direction)
        if not os.path.isdir(dir_path):
            continue
        subdirs = sorted(d for d in os.listdir(dir_path) 
                         if os.path.isdir(os.path.join(dir_path, d)))
        if subdirs:
            for load_name in subdirs:
                load_dir = os.path.join(dir_path, load_name)
                load_val = int(load_name) if load_name.isdigit() else None
                files = sorted(f for f in os.listdir(load_dir)
                               if f.endswith(".txt") and not f.startswith("."))
                for i, fn in enumerate(files, 1):
                    fp = os.path.join(load_dir, fn)
                    arr = read_one_txt(fp)
                    if arr is None: continue
                    key = f"{split}_{direction}_{load_name}_{i}"
                    out[key] = {"data": arr, "direction": direction, "load": load_val, "path": fp}
        else:
            files = sorted(f for f in os.listdir(dir_path)
                           if f.endswith(".txt") and not f.startswith("."))
            for i, fn in enumerate(files, 1):
                fp = os.path.join(dir_path, fn)
                arr = read_one_txt(fp)
                if arr is None: continue
                key = f"{split}_{direction}_unknown_{i}"
                out[key] = {"data": arr, "direction": direction, "load": None, "path": fp}
    return out

# ---------- 特徵 ----------
def extract_features(signal: np.ndarray, fs: int = 4880) -> Dict[str, float]:
    s = signal.ravel().astype(float)
    feats: Dict[str, float] = {}

    feats["RMS"] = float(np.sqrt(np.mean(s**2)))
    feats["Skewness"] = float(skew(s))
    feats["Kurtosis"] = float(kurtosis(s))

    pdf, _ = np.histogram(s, bins=100, density=True)
    feats["Entropy"] = float(entropy(pdf + 1e-12))
    feats["CrestFactor"] = float(np.max(np.abs(s)) / (feats["RMS"] + 1e-12))

    fft_vals = np.abs(fft(s))[: len(s)//2]
    fft_freq = fftfreq(len(s), d=1/fs)[: len(s)//2]
    feats["Freq_Center"] = float(np.sum(fft_freq * fft_vals) / (np.sum(fft_vals) + 1e-12))
    feats["RMSF"] = float(np.sqrt(np.sum((fft_freq**2) * (fft_vals**2)) / (np.sum(fft_vals**2) + 1e-12)))
    feats["Spectral_Kurtosis"] = float(kurtosis(fft_vals))
    feats["Spectral_Entropy"] = float(entropy(fft_vals / (np.sum(fft_vals) + 1e-12)))

    feats["Impulse_Factor"] = float(np.max(np.abs(s)) / (np.mean(np.abs(s)) + 1e-12))
    feats["Margin_Factor"]  = float(np.max(np.abs(s)) / ((np.mean(np.sqrt(np.abs(s)))**2) + 1e-12))
    feats["Approximate_Entropy"] = float(np.std(np.diff(s)) / (np.std(s) + 1e-12))

    coeffs = pywt.wavedec(s, "db4", level=3)
    feats["Wavelet_D3_Kurtosis"] = float(kurtosis(coeffs[1]))

    feats["Spectral_Energy"]   = float(np.sum(fft_vals**2))
    feats["Clearance_Factor"]  = float(np.max(np.abs(s)) / (np.mean(np.sqrt(np.abs(s)))**2 + 1e-12))
    return feats

def make_feature_df(raw_dict: Dict[str, Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for key, item in raw_dict.items():
        arr: np.ndarray = item["data"]
        direction: str  = item["direction"]
        load            = item["load"]

        sig = arr.mean(axis=1) if arr.ndim == 2 and arr.shape[1] >= 3 else arr.ravel()
        feats = extract_features(sig)
        feats.update({"Direction": direction, "Load": load, "Key": key})
        rows.append(feats)

    df = pd.DataFrame(rows)
    base_cols = FEATURE_COLS + ["Direction", "Load", "Key"]
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[base_cols]

def extract_features_from_array(arr: np.ndarray) -> np.ndarray:
    """
    對單一 numpy array 計算特徵，並轉成 numpy array (不依賴 json)
    """
    feat_dict = extract_features(arr)
    feat_list = list(feat_dict.values())  # 直接取值，順序可能不保證
    return np.array(feat_list, dtype=float)



# ---------- 健康標籤 ----------
def assign_gt_health(df: pd.DataFrame) -> pd.DataFrame:
    def _rule(row):
        d, load = row["Direction"], row["Load"]
        if d in ("Xa","Xb"):
            if load == 80: return 0
            elif load in (65,95): return 1
            elif load == 130: return 2
        elif d in ("Ya","Yb"):
            if load == 260: return 0
            elif load in (220,300): return 1
            elif load == 380: return 2
        return None
    df = df.copy()
    df["GT_Health"] = df.apply(_rule, axis=1)
    df = df.dropna(subset=["GT_Health"])
    df["GT_Health"] = df["GT_Health"].astype(int)
    return df

# ---------- 訓練 / 推論 ----------
def train_health_model(train_df: pd.DataFrame) -> Tuple[StandardScaler, RandomForestClassifier]:
    X = train_df[FEATURE_COLS].to_numpy()
    y = train_df["GT_Health"].to_numpy().astype(int)
    scaler = StandardScaler().fit(X)
    clf = RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced"
    ).fit(scaler.transform(X), y)
    return scaler, clf

def train_load_model(train_df: pd.DataFrame) -> RandomForestRegressor:
    X = train_df[FEATURE_COLS].to_numpy()
    y = train_df["Load"].to_numpy().astype(float)
    reg = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, y)
    return reg

def predict_health(df: pd.DataFrame, scaler: StandardScaler, clf: RandomForestClassifier) -> Tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].to_numpy()
    proba = clf.predict_proba(scaler.transform(X))
    pred = np.argmax(proba, axis=1)
    return pred, proba

def predict_load(df: pd.DataFrame, reg: RandomForestRegressor) -> np.ndarray:
    X = df[FEATURE_COLS].to_numpy()
    return reg.predict(X)

def get_feature_importance(clf: RandomForestClassifier) -> pd.DataFrame:
    imp = clf.feature_importances_
    return pd.DataFrame({"Feature": FEATURE_COLS, "Importance": imp}).sort_values("Importance", ascending=False)

# ---------- 模型存取 ----------
def save_models(models: Dict[str, object], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if "scaler" in models: joblib.dump(models["scaler"], os.path.join(out_dir, "ml_scaler.joblib"))
    if "clf"    in models: joblib.dump(models["clf"],    os.path.join(out_dir, "ml_classifier.joblib"))
    if "reg"    in models: joblib.dump(models["reg"],    os.path.join(out_dir, "ml_regressor.joblib"))
    json.dump(FEATURE_COLS, open(os.path.join(out_dir, "ml_feature_cols.json"), "w"), ensure_ascii=False, indent=2)

def load_models(out_dir: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    sp = os.path.join(out_dir, "ml_scaler.joblib")
    cp = os.path.join(out_dir, "ml_classifier.joblib")
    rp = os.path.join(out_dir, "ml_regressor.joblib")
    if os.path.exists(sp): out["scaler"] = joblib.load(sp)
    if os.path.exists(cp): out["clf"]    = joblib.load(cp)
    if os.path.exists(rp): out["reg"]    = joblib.load(rp)
    return out

# ---------- 批次推論 ----------
def batch_predict_dir(dir_path: str, models: Dict[str, object]) -> pd.DataFrame:
    raw = {}
    parent = os.path.basename(os.path.dirname(dir_path))  # 方向
    load   = os.path.basename(dir_path)
    files = sorted(f for f in os.listdir(dir_path) if f.endswith(".txt"))
    for i, fn in enumerate(files, 1):
        fp = os.path.join(dir_path, fn)
        arr = read_one_txt(fp)
        if arr is None: continue
        key = f"{parent}_{load}_{i}"
        raw[key] = {"data": arr, "direction": parent, "load": int(load) if load.isdigit() else None, "path": fp}

    df = make_feature_df(raw)
    pred, proba = predict_health(df, models["scaler"], models["clf"])
    est_load    = predict_load(df, models["reg"])
    out = pd.DataFrame({"Key": df["Key"], "Pred": pred, "EstLoad": est_load})
    out = pd.concat([out, pd.DataFrame(proba, columns=["P_Healthy","P_Mild","P_Severe"])], axis=1)
    return out

# ---------- 本地/Colab 測試入口 ----------
if __name__ == "__main__":
    DATA_ROOT = "/content/drive/MyDrive/vibration/data"
    train_raw = read_split(DATA_ROOT, "train")
    train_df  = make_feature_df(train_raw)
    train_df  = assign_gt_health(train_df)

    scaler, clf = train_health_model(train_df)
    reg = train_load_model(train_df)

    save_models({"scaler": scaler, "clf": clf, "reg": reg},
                out_dir="/content/drive/MyDrive/vibration/models")

    imp_df = get_feature_importance(clf)
    print("Top features:\n", imp_df.head())
    print("Saved models.")
