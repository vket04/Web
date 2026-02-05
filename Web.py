import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import median_filter
import re


# ==========================================
# 1. KIẾN TRÚC MẠNG (Giữ nguyên v3.0)
# ==========================================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = x.mean(dim=-1).view(b, c, 1)
        return x * self.fc(x.mean(dim=-1)).view(b, c, 1)


class ResidualBlockv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlockv3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels), nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels)
        )
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1),
                                          nn.BatchNorm1d(out_channels))

    def forward(self, x):
        return torch.relu(self.se(self.conv(x)) + self.shortcut(x))


class RamanResNetV3(nn.Module):
    def __init__(self, num_targets=4):
        super(RamanResNetV3, self).__init__()
        self.stem = nn.Sequential(nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3), nn.ReLU())
        self.layer1 = ResidualBlockv3(64, 64)
        self.layer2 = ResidualBlockv3(64, 128)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.regressor = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, num_targets))

    def forward(self, x):
        x = self.layer2(self.layer1(self.stem(x)))
        features = torch.cat([self.avg_pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)
        return self.regressor(features)


# ==========================================
# 2. HÀM HỖ TRỢ
# ==========================================
@st.cache_resource
def load_v3_model():
    model = RamanResNetV3(num_targets=4)
    model.eval()
    return model


def preprocess(spectrum):
    clean = median_filter(spectrum, size=3)
    x = clean.reshape(1, -1)
    d1 = savgol_filter(x, 15, 3, deriv=1)
    d2 = savgol_filter(x, 15, 3, deriv=2)
    snv = lambda data: (data - np.mean(data, axis=1, keepdims=True)) / (np.std(data, axis=1, keepdims=True) + 1e-8)
    x_proc = np.stack([snv(x), snv(d1), snv(d2)], axis=1)
    return torch.tensor(x_proc, dtype=torch.float32)


# ==========================================
# 3. GIAO DIỆN CHÍNH
# ==========================================
st.set_page_config(page_title="Raman AI Pro v3.5", layout="wide")

# Khởi tạo mô hình
model = load_v3_model()
sugars = ["Sucrose", "Fructose", "Maltose", "Glucose"]

with st.sidebar:
    st.header("⚙️ Cấu hình")
    up_file = st.file_uploader("Nạp dữ liệu quang phổ (CSV)", type="csv")

    if up_file:
        df_spec = pd.read_csv(up_file)
        all_samples = df_spec.columns[1:].tolist()

        # Công cụ lọc mẫu thông minh
        st.subheader("🔍 Lọc mẫu")
        plates = sorted(list(set([s.split('_')[5] for s in all_samples])))
        sel_plate = st.selectbox("Chọn Plate:", plates)
        p_samples = [s for s in all_samples if s.split('_')[5] == sel_plate]

        wells = sorted(list(set([s.split('_')[4] for s in p_samples])))
        sel_well = st.selectbox("Chọn vị trí Giếng (Well):", wells)

        reps = [s for s in p_samples if s.split('_')[4] == sel_well]
        sample = st.radio("Lần đo (Rep):", reps)

        st.divider()
        st.subheader("🛠️ Công cụ đồ thị")
        show_raw = st.toggle("Hiển thị phổ gốc", value=True)
        show_peaks = st.toggle("Tự động dò đỉnh phổ", value=True)

if up_file and sample:
    spec_data = df_spec[sample].values
    wn = df_spec.iloc[:, 0].values

    tab1, tab2, tab3 = st.tabs(["📊 Kết quả dự đoán", "🔬 Phân tích chi tiết", "🌡️ Bản đồ nồng độ"])

    with tab1:
        st.subheader(f"Mẫu: {sample}")

        # Dự đoán
        with torch.no_grad():
            preds = np.abs(model(preprocess(spec_data)).numpy()[0] * 375.0)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            # Vẽ đồ thị với chức năng Zoom vùng đặc trưng
            fig = go.Figure()
            if show_raw:
                fig.add_trace(
                    go.Scatter(x=wn, y=spec_data, name="Raw Signal", line=dict(color='gray', width=1, dash='dot')))

            clean_data = median_filter(spec_data, 3)
            fig.add_trace(go.Scatter(x=wn, y=clean_data, name="Cleaned", line=dict(color='#00ffa2', width=2)))

            if show_peaks:
                p, _ = find_peaks(clean_data, height=np.mean(clean_data) * 1.2, distance=30)
                fig.add_trace(go.Scatter(x=wn[p], y=clean_data[p], mode='markers', marker=dict(color='red', size=8),
                                         name="Peaks"))

            fig.update_layout(template="plotly_dark", height=450, xaxis_title="Wavenumber (cm-1)",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("### 🧪 Nồng độ dự đoán")
            for i, s in enumerate(sugars):
                st.metric(label=s, value=f"{preds[i]:.2f} µl", delta=f"{(preds[i] - 50):.1f} vs Avg")

            # Bảng tóm tắt
            summary_df = pd.DataFrame({"Thành phần": sugars, "Nồng độ (µl)": np.round(preds, 2)})
            st.dataframe(summary_df, hide_index=True, use_container_width=True)
            st.download_button("📥 Xuất kết quả CSV", summary_df.to_csv(index=False), f"Result_{sample}.csv")

    with tab2:
        st.subheader("🔬 So sánh đối chứng")
        multi_sel = st.multiselect("Chọn các mẫu để so sánh phổ:", all_samples, default=[sample])
        if multi_sel:
            fig_multi = go.Figure()
            for s in multi_sel:
                fig_multi.add_trace(go.Scatter(x=wn, y=df_spec[s], name=s))
            fig_multi.update_layout(template="plotly_dark", height=600)
            st.plotly_chart(fig_multi, use_container_width=True)

    with tab3:
        st.subheader(f"🌡️ Heatmap Plate {sel_plate}")
        target = st.selectbox("Xem phân bố nồng độ loại đường:", sugars)
        s_idx = sugars.index(target)

        # Tạo Heatmap 8x12
        r_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        c_labels = [str(i) for i in range(1, 13)]
        grid = np.zeros((8, 12))

        for s in p_samples:
            try:
                w = s.split('_')[4]
                r_i = r_labels.index(w[0])
                c_i = int(re.search(r'\d+', w).group()) - 1
                # Dự đoán nhanh cho Heatmap
                with torch.no_grad():
                    p_val = np.abs(model(preprocess(df_spec[s].values)).numpy()[0] * 375.0)
                    grid[r_i, c_i] = p_val[s_idx]
            except:
                continue

        fig_heat = px.imshow(grid, x=c_labels, y=r_labels, color_continuous_scale='Viridis', text_auto=".1f")
        fig_heat.update_layout(template="plotly_dark")
        st.plotly_chart(fig_heat, use_container_width=True)

else:
    st.info("👋 Vui lòng nạp file CSV dữ liệu quang phổ để bắt đầu phân tích.")