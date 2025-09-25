import os
import sys
import json
import joblib
import torch
import shap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# 把 code/ 放進匯入路徑
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))
import dl  # noqa
import ml  # noqa

# ====\\ 路徑
MODEL_DIR       = os.path.join(os.path.dirname(__file__), "..", "models")
DL_MODEL_PATH   = os.path.join(MODEL_DIR, "cnn_model.pth")
DL_SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
ML_SCALER_PATH  = os.path.join(MODEL_DIR, "ml_scaler.joblib")
ML_MODEL_PATH   = os.path.join(MODEL_DIR, "ml_classifier.joblib")
FEATURE_COLS_FP = os.path.join(MODEL_DIR, "ml_feature_cols.json")

DATA_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TRAIN_FEATURES_FP  = os.path.join(DATA_PROCESSED_DIR, "train_features.csv")

# 讀訓練時的特徵欄位順序
with open(FEATURE_COLS_FP, "r", encoding="utf-8") as f:
    FEATURE_COLS = json.load(f)

# ==== 介面 ====
st.set_page_config(page_title="機械手臂震動健康檢測系統", layout="wide")
st.title("機械手臂震動健康檢測系統")

st.markdown("""
**支援檔案格式說明：**  
請上傳 `.txt` 檔，檔案需包含三軸的原始震動訊號
""")

uploaded_files = st.file_uploader("上傳震動資料檔案（可多檔）", type=["txt"], accept_multiple_files=True)
model_type = st.radio("選擇模型", [
    "快速檢測 (Random Forest Classifier)",
    "深度分析 (3D CNN with CoordConv)",
    "異常檢測 (IForest / OCSVM)"
])

def to_light(prob_healthy: float) -> str:
    """將健康度(0~1)轉為燈號"""
    return "🟢" if prob_healthy >= 0.70 else ("🟡" if prob_healthy >= 0.30 else "🔴")


# =========================
# ML：快速檢測
# =========================
if model_type.startswith("快速檢測"):
    if st.button("開始分析"):
        if not uploaded_files:
            st.warning("請先上傳檔案")
            st.stop()

        scaler = joblib.load(ML_SCALER_PATH)
        clf    = joblib.load(ML_MODEL_PATH)

        rows, X_list, names = [], [], []

        for file in uploaded_files:
            arr = ml.read_one_txt(file)
            if arr is None:
                continue
            feat_dict = ml.extract_features(arr)
            feat_vec  = np.array([feat_dict[c] for c in FEATURE_COLS], dtype=float)
            X_list.append(feat_vec)
            names.append(file.name)

        if not X_list:
            st.error("無法從檔案讀出有效資料。")
            st.stop()

        X = np.vstack(X_list)
        Xn = scaler.transform(X)
        proba = clf.predict_proba(Xn)

        p_healthy = proba[:, 0]
        for fname, ph in zip(names, p_healthy):
            rows.append({"檔案": fname, "預測健康度": f"{ph:.2f} {to_light(ph)}"})

        st.subheader("分析結果")
        df_result = pd.DataFrame(rows)
        csv = df_result.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載報表 (CSV)", csv, "結果報表.csv", "text/csv")
        st.dataframe(df_result, use_container_width=True)

        # ---- SHAP ----
        st.subheader("特徵貢獻")
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(Xn)

        new_labels = ["Normal", "Mild Degradation", "Severe Degradation"]

        fig_bar = plt.figure(figsize=(6, 4))
        shap.summary_plot(
            shap_values, Xn, feature_names=FEATURE_COLS,
            plot_type="bar", show=False, max_display=5
        )
        plt.xlabel("")
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(handles, new_labels, loc="best")
        st.pyplot(fig_bar, clear_figure=True)


# =========================
# DL：深度分析
# =========================
elif model_type.startswith("深度分析"):
    if st.button("開始分析"):
        if not uploaded_files:
            st.warning("請先上傳檔案")
            st.stop()

        scaler = joblib.load(DL_SCALER_PATH)
        device = dl.get_device()
        model  = dl.CNN3D_con_coord()
        model  = dl.load_torch(model, DL_MODEL_PATH, device, feature_bins=len(scaler["freq"]))

        preds, files, spectra_list = [], [], []
        for file in uploaded_files:
            pred, spectra = dl.predict_one_file(file, scaler, model, device=device)
            preds.append(float(pred))
            files.append(file.name)
            spectra_list.append(spectra)

        st.subheader("分析結果")
        df_dl = pd.DataFrame({"檔案": files, "預測負荷": [f"{p:.2f}" for p in preds]})
        csv = df_dl.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載報表 (CSV)", csv, "DL_結果.csv", "text/csv")
        st.dataframe(df_dl, use_container_width=True)

        mean_val, std_val = np.mean(preds), np.std(preds)
        threshold = mean_val + 3 * std_val
        st.markdown(
            f"**統計：mean={mean_val:.2f}｜std={std_val:.2f}｜"
            f"min={np.min(preds):.2f}｜max={np.max(preds):.2f}｜outlier threshold={threshold:.2f}**"
        )

        st.subheader("預測分布")
        fig_pred = plt.figure(figsize=(6, 4))
        sns.histplot(preds, kde=True, bins=20, color="skyblue")
        plt.xlabel("Predicted Value")
        plt.ylabel("Frequency")
        st.pyplot(fig_pred, clear_figure=True)

        st.subheader("逐筆預測")
        fig_line, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(range(len(preds)), preds, marker="o")
        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1, label="outlier threshold")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Predicted Value")
        ax.grid(alpha=0.3)
        ax.legend()
        for i, (p, fname) in enumerate(zip(preds, files)):
            if p > threshold:
                ax.scatter(i, p, color="red", s=80, zorder=5)
                ax.text(i + 3, p, fname, fontsize=8, color="red", ha="left", va="center")
        st.pyplot(fig_line, clear_figure=True)

        # ==== PSD/CSD 熱圖（僅 outlier）====
        outlier_idx = [i for i, p in enumerate(preds) if p > threshold]
        if outlier_idx:
            st.subheader("異常樣本 PSD/CSD 熱圖")
            for i in outlier_idx:
                fname, arr = files[i], spectra_list[i]
                with st.expander(f"PSD/CSD 熱圖 - {fname}"):
                    if isinstance(arr, dict):
                        for k2, mat in arr.items():
                            fig = plt.figure(figsize=(6, 3))
                            plt.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
                            plt.title(k2)
                            plt.colorbar()
                            st.pyplot(fig)
                            plt.close(fig)
                    else:
                        st.write("⚠️ 此檔案沒有有效的 PSD/CSD")


# =========================
# 異常檢測 (IForest / OCSVM)
# =========================
elif model_type.startswith("異常檢測"):
    st.subheader("異常檢測 (Isolation Forest / One-Class SVM)")

    cont = st.slider("設定 contamination / nu", 0.01, 0.20, 0.05, 0.01)

    if st.button("開始分析", key="ab_start"):
        if not uploaded_files:
            st.warning("請先上傳檔案！")
            st.stop()

        if not os.path.exists(TRAIN_FEATURES_FP):
            st.error("❌ 缺少訓練特徵檔 train_features.csv，請先生成 (80 與 260 負荷樣本作為 baseline)")
            st.stop()

        # ==== baseline 訓練集 ====
        train_df = pd.read_csv(TRAIN_FEATURES_FP)
        X_train = train_df[["RMS", "Kurtosis"]].to_numpy()

        # ==== 快取測試特徵 ====
        file_names = [f.name for f in uploaded_files]
        if "cached_test" not in st.session_state or st.session_state.get("cached_files") != file_names:
            rows = []
            for file in uploaded_files:
                arr = ml.read_one_txt(file)
                if arr is None:
                    continue
                feat_dict = ml.extract_features(arr)
                rows.append({"Key": file.name, **feat_dict})
            if not rows:
                st.error("❌ 上傳檔案無法計算有效特徵")
                st.stop()
            test_df = pd.DataFrame(rows)
            st.session_state["cached_test"] = test_df
            st.session_state["cached_files"] = file_names
        else:
            test_df = st.session_state["cached_test"]

        X_test = test_df[["RMS", "Kurtosis"]].to_numpy()

        # ==== Isolation Forest ====
        scaler_if = StandardScaler().fit(X_train)
        Xn_train_if = scaler_if.transform(X_train)
        Xn_test_if  = scaler_if.transform(X_test)

        iforest = IsolationForest(n_estimators=200, contamination=cont, random_state=42).fit(Xn_train_if)
        y_pred_if = iforest.predict(Xn_test_if)

        # ==== One-Class SVM ====
        scaler_oc = StandardScaler().fit(X_train)
        Xn_train_oc = scaler_oc.transform(X_train)
        Xn_test_oc  = scaler_oc.transform(X_test)

        ocsvm = OneClassSVM(kernel="rbf", nu=cont, gamma="scale").fit(Xn_train_oc)
        y_pred_oc = ocsvm.predict(Xn_test_oc)

        # ==== Decision Boundary (IForest) ====
        st.write("### Isolation Forest Decision Boundary")
        xx, yy = np.meshgrid(
            np.linspace(Xn_test_if[:, 0].min()-1, Xn_test_if[:, 0].max()+1, 200),
            np.linspace(Xn_test_if[:, 1].min()-1, Xn_test_if[:, 1].max()+1, 200)
        )
        Z_if = iforest.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig_if = plt.figure(figsize=(6, 4))
        plt.contourf(xx, yy, Z_if, cmap=plt.cm.coolwarm, alpha=0.3)
        plt.scatter(Xn_test_if[:, 0], Xn_test_if[:, 1],
                    c=["green" if p == 1 else "red" for p in y_pred_if],
                    edgecolor="k")
        plt.xlabel("RMS (standardized)")
        plt.ylabel("Kurtosis (standardized)")
        plt.title("Isolation Forest Decision Boundary")
        st.pyplot(fig_if)

        # ==== Decision Boundary (OCSVM) ====
        st.write("### One-Class SVM Decision Boundary")
        xx, yy = np.meshgrid(
            np.linspace(Xn_test_oc[:, 0].min()-1, Xn_test_oc[:, 0].max()+1, 200),
            np.linspace(Xn_test_oc[:, 1].min()-1, Xn_test_oc[:, 1].max()+1, 200)
        )
        Z_oc = ocsvm.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig_oc = plt.figure(figsize=(6, 4))
        plt.contourf(xx, yy, Z_oc, levels=np.linspace(Z_oc.min(), Z_oc.max(), 30),
                     cmap=plt.cm.coolwarm, alpha=0.6)
        plt.contour(xx, yy, Z_oc, levels=[0], linewidths=2, colors="k")
        plt.scatter(Xn_test_oc[:, 0], Xn_test_oc[:, 1], c=y_pred_oc,
                    cmap=plt.cm.Paired, edgecolors="k", s=40)
        plt.xlabel("RMS (standardized)")
        plt.ylabel("Kurtosis (standardized)")
        plt.title("One-Class SVM Decision Boundary")
        st.pyplot(fig_oc)

        # ==== 同時異常樣本 ====
        both_anom = test_df[(y_pred_if == -1) & (y_pred_oc == -1)][["Key", "RMS", "Kurtosis"]]
        st.write("### 同時被判定為異常的樣本")
        csv = both_anom.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下載異常樣本 (CSV)", csv, "anomalies.csv", "text/csv")
        st.dataframe(both_anom)
