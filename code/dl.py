import os
import numpy as np
import pandas as pd
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error

# ---------------- utils ----------------
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def save_torch(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_torch(model, path, device, feature_bins):
    model.to(device).eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 4, feature_bins, 3, 3, device=device)
        _ = model(dummy)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    return model

# ---------------- read / clean ----------------
def read_one_txt(file, min_length: int = 18000):
    """讀取單一 txt，支援路徑字串與 Streamlit UploadedFile"""
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


def read_data(folder_path):
    """讀取資料夾 (含子資料夾)，輸出 dict"""
    out = {}
    subfolders = sorted(
        f for f in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, f))
    )
    for sub in subfolders:
        sub_dir = os.path.join(folder_path, sub)
        files = sorted(f for f in os.listdir(sub_dir) if f.endswith(".txt"))
        for i, fn in enumerate(files, 1):
            arr = read_one_txt(os.path.join(sub_dir, fn))
            if arr is None:
                continue
            out[f"data_{sub}_{i}"] = arr
    return out

def tune_data(data_dict, min_len=16175, target_len=16175, prefix="data_"):
    cleaned = {}
    for k, v in data_dict.items():
        if not (isinstance(v, np.ndarray) and k.startswith(prefix)):
            continue
        if v.shape[0] < min_len:
            continue
        v = v[:target_len, :]
        cleaned[k] = v
    return cleaned

# ---------------- PSD/CSD ----------------
def _welch_all_axes(arr, fs, nperseg, mask):
    f, Pxx = signal.welch(arr[:, 0], fs=fs, nperseg=nperseg)
    _, Pyy = signal.welch(arr[:, 1], fs=fs, nperseg=nperseg)
    _, Pzz = signal.welch(arr[:, 2], fs=fs, nperseg=nperseg)
    _, Pxy = signal.csd(arr[:, 0], arr[:, 1], fs=fs, nperseg=nperseg)
    _, Pxz = signal.csd(arr[:, 0], arr[:, 2], fs=fs, nperseg=nperseg)
    _, Pyz = signal.csd(arr[:, 1], arr[:, 2], fs=fs, nperseg=nperseg)
    Pxy, Pxz, Pyz = np.abs(Pxy), np.abs(Pxz), np.abs(Pyz)

    if mask is not None:
        m = (f >= mask[0]) & (f <= mask[1])
        f = f[m]; Pxx, Pyy, Pzz = Pxx[m], Pyy[m], Pzz[m]
        Pxy, Pxz, Pyz = Pxy[m], Pxz[m], Pyz[m]
    return f, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz

def _stack_to_matrix(Pxx, Pyy, Pzz, Pxy, Pxz, Pyz):
    N, F = Pxx.shape
    X = np.zeros((N, F, 3, 3), dtype=np.float32)
    X[:, :, 0, 0] = Pxx; X[:, :, 1, 1] = Pyy; X[:, :, 2, 2] = Pzz
    X[:, :, 0, 1] = X[:, :, 1, 0] = Pxy
    X[:, :, 0, 2] = X[:, :, 2, 0] = Pxz
    X[:, :, 1, 2] = X[:, :, 2, 1] = Pyz
    return X

def compute_psd_csd(data_dict, nperseg=256, mask=None, scaler=None):
    Pxxs, Pyys, Pzzs, Pxys, Pxzs, Pyzs, ys = [], [], [], [], [], [], []

    fs = min(len(v) for v in data_dict.values()) / 5.0
    f_ref = None
    for k, arr in data_dict.items():
        try:
            load = int(k.split("_")[1])  # 有 label 的情況
        except:
            load = 0  # test 資料無 label
        f, Pxx, Pyy, Pzz, Pxy, Pxz, Pyz = _welch_all_axes(arr, fs, nperseg, mask)
        if f_ref is None: f_ref = f
        Pxxs.append(Pxx); Pyys.append(Pyy); Pzzs.append(Pzz)
        Pxys.append(Pxy); Pxzs.append(Pxz); Pyzs.append(Pyz)
        ys.append(load)

    Pxx, Pyy, Pzz = np.stack(Pxxs), np.stack(Pyys), np.stack(Pzzs)
    Pxy, Pxz, Pyz = np.stack(Pxys), np.stack(Pxzs), np.stack(Pyzs)
    y = np.array(ys, dtype=np.float32).reshape(-1, 1)

    if scaler is None:
        # 建立訓練集固定 scaler
        Pxx_mean, Pxx_std = Pxx.mean(0), Pxx.std(0)+1e-8
        Pyy_mean, Pyy_std = Pyy.mean(0), Pyy.std(0)+1e-8
        Pzz_mean, Pzz_std = Pzz.mean(0), Pzz.std(0)+1e-8
        Pxy_mean, Pxy_std = Pxy.mean(0), Pxy.std(0)+1e-8
        Pxz_mean, Pxz_std = Pxz.mean(0), Pxz.std(0)+1e-8
        Pyz_mean, Pyz_std = Pyz.mean(0), Pyz.std(0)+1e-8

        Pxx_n = (Pxx-Pxx_mean)/Pxx_std; Pyy_n = (Pyy-Pyy_mean)/Pyy_std
        Pzz_n = (Pzz-Pzz_mean)/Pzz_std; Pxy_n = (Pxy-Pxy_mean)/Pxy_std
        Pxz_n = (Pxz-Pxz_mean)/Pxz_std; Pyz_n = (Pyz-Pyz_mean)/Pyz_std

        X = _stack_to_matrix(Pxx_n, Pyy_n, Pzz_n, Pxy_n, Pxz_n, Pyz_n)

        scaler = {
            "freq": f_ref,
            "Pxx": {"mean": Pxx_mean, "std": Pxx_std},
            "Pyy": {"mean": Pyy_mean, "std": Pyy_std},
            "Pzz": {"mean": Pzz_mean, "std": Pzz_std},
            "Pxy": {"mean": Pxy_mean, "std": Pxy_std},
            "Pxz": {"mean": Pxz_mean, "std": Pxz_std},
            "Pyz": {"mean": Pyz_mean, "std": Pyz_std},
            "y_min": y.min(), "y_max": y.max()
        }
        spectra = {"Pxx": Pxx, "Pyy": Pyy, "Pzz": Pzz,
                   "Pxy": Pxy, "Pxz": Pxz, "Pyz": Pyz}
        return X.astype(np.float32), y.astype(np.float32), scaler, spectra
    else:
        Pxx_n = (Pxx-scaler["Pxx"]["mean"])/(scaler["Pxx"]["std"]+1e-8)
        Pyy_n = (Pyy-scaler["Pyy"]["mean"])/(scaler["Pyy"]["std"]+1e-8)
        Pzz_n = (Pzz-scaler["Pzz"]["mean"])/(scaler["Pzz"]["std"]+1e-8)
        Pxy_n = (Pxy-scaler["Pxy"]["mean"])/(scaler["Pxy"]["std"]+1e-8)
        Pxz_n = (Pxz-scaler["Pxz"]["mean"])/(scaler["Pxz"]["std"]+1e-8)
        Pyz_n = (Pyz-scaler["Pyz"]["mean"])/(scaler["Pyz"]["std"]+1e-8)
        X = _stack_to_matrix(Pxx_n, Pyy_n, Pzz_n, Pxy_n, Pxz_n, Pyz_n)
        spectra = {"Pxx": Pxx, "Pyy": Pyy, "Pzz": Pzz,
                   "Pxy": Pxy, "Pxz": Pxz, "Pyz": Pyz}
        return X.astype(np.float32), y.astype(np.float32), spectra

# ---------------- dataset / model ----------------
def _add_coord_channels(x5d):
    N, _, D, H, W = x5d.size()
    z = torch.linspace(-1, 1, D, device=x5d.device)
    y = torch.linspace(-1, 1, H, device=x5d.device)
    x = torch.linspace(-1, 1, W, device=x5d.device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    zz = zz.unsqueeze(0).unsqueeze(0).expand(N,-1,-1,-1,-1)
    yy = yy.unsqueeze(0).unsqueeze(0).expand(N,-1,-1,-1,-1)
    xx = xx.unsqueeze(0).unsqueeze(0).expand(N,-1,-1,-1,-1)
    return torch.cat([x5d, zz, yy, xx], dim=1)

def make_loader_reg(X, y, batch_size=16, add_coord=True):
    X = torch.from_numpy(X).float().reshape(-1,1,X.shape[1],X.shape[2],X.shape[3])
    if add_coord: X = _add_coord_channels(X)
    y = torch.from_numpy(y).float().view(-1,1)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

class CNN3D_con_coord(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(4,16,(7,3,3),padding=(2,1,1))
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16,32,(5,3,3),padding=(1,1,1))
        self.bn2 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32,64,(3,3,3))
        self.bn3 = nn.BatchNorm3d(64)
        self.pool = nn.MaxPool3d((2,1,1))
        self.fc1=None; self.bn_fc1=nn.BatchNorm1d(256)
        self.drop=nn.Dropout(0.4)
        self.fc2=nn.Linear(256,64); self.fc3=nn.Linear(64,16); self.fc4=nn.Linear(16,1)
    def forward(self,x):
        x=self.pool(F.relu(self.bn1(self.conv1(x))))
        x=self.pool(F.relu(self.bn2(self.conv2(x))))
        x=F.relu(self.bn3(self.conv3(x)))
        x=torch.flatten(x,1)
        if self.fc1 is None: self.fc1=nn.Linear(x.shape[1],256).to(x.device)
        x=F.relu(self.bn_fc1(self.fc1(x))); x=self.drop(x)
        x=F.relu(self.fc2(x)); x=F.relu(self.fc3(x))
        return self.fc4(x)

# ---------------- train / eval ----------------
def train_regression(loader, model, scaler, epochs=300, lr=1e-3, device=None):
    device = device or get_device()
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.7)

    y_min, y_max = scaler["y_min"], scaler["y_max"]

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_norm = (y-y_min)/(y_max-y_min+1e-8)
            out = model(x)
            loss = F.mse_loss(out, y_norm)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

@torch.no_grad()
def evaluate_regression(loader, model, scaler, device=None):
    device = device or get_device()
    model.to(device).eval()
    y_min, y_max = scaler["y_min"], scaler["y_max"]
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        y_real = out*(y_max-y_min)+y_min
        preds.append(y_real.cpu().numpy()); trues.append(y.cpu().numpy())
    return float(mean_squared_error(np.vstack(trues), np.vstack(preds))), np.vstack(trues).ravel(), np.vstack(preds).ravel()

# ---------------- predict ----------------
def predict_dir(dir_path, scaler, model, nperseg=256, mask=None, device=None):
    files = sorted(f for f in os.listdir(dir_path) if f.endswith(".txt"))
    tmp = {}
    for i, fn in enumerate(files, 1):
        arr = read_one_txt(os.path.join(dir_path, fn))
        if arr is None:
            continue
        tmp[f"data_tmp_{i}"] = arr
    X,y,spectra = compute_psd_csd(tmp,nperseg=nperseg,mask=mask,scaler=scaler)
    loader = make_loader_reg(X,y,add_coord=True)
    _,_,y_pred = evaluate_regression(loader,model,scaler,device=device)
    return pd.DataFrame({"Key":list(tmp.keys()),"Pred":y_pred}), spectra

def predict_one_file(file, scaler, model, nperseg=256, mask=None, device=None):
    # file 可能是 UploadedFile 或路徑字串
    if hasattr(file, "getvalue"):  # Streamlit UploadedFile
        arr = read_one_txt(file, min_length=1)  # <--- 放寬條件
    else:  # 當成路徑字串
        arr = read_one_txt(file, min_length=1)

    if arr is None:
        raise ValueError(f"無法讀取檔案: {getattr(file, 'name', file)}")

    tmp = {"data_tmp_1": arr}
    X, y, spectra = compute_psd_csd(tmp, nperseg=nperseg, mask=mask, scaler=scaler)
    loader = make_loader_reg(X, y, add_coord=True)
    _, _, y_pred = evaluate_regression(loader, model, scaler, device=device)
    return float(y_pred[0]), spectra
