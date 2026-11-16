import io
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# 1. CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_A_PATH = (
    PROJECT_ROOT / "outputs/notebook_05/models/original/best_model.keras"
)
MODEL_B_PATH = (
    PROJECT_ROOT / "outputs/notebook_05/models/roi_cropped/best_model.keras"
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

CLASS_DISPLAY = {
    "glioma": ("Glioma", "#f43f5e", "rgba(244, 63, 94, 0.14)"),
    "meningioma": ("Meningioma", "#f59e0b", "rgba(245, 158, 11, 0.14)"),
    "notumor": ("No Tumor", "#10b981", "rgba(16, 185, 129, 0.14)"),
    "pituitary": ("Pituitary", "#8b5cf6", "rgba(139, 92, 246, 0.14)"),
}

IMAGE_SIZE = (224, 224)

# Options: "resnet" | "rescale" | "none"
PREPROCESSING_MODE = "resnet"

# ROI settings (matched to Notebook 04)
ROI_PADDING = 0.08
ROI_KERNEL_SIZE = 15
ROI_MIN_AREA = 0.01

FUSION_WEIGHT_A = 0.5
FUSION_WEIGHT_B = 0.5


# ============================================================
# 2. STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NeuroVision · Brain MRI Diagnostics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 3. DRIBBBLE-INSPIRED BENTO THEME & NATIVE WIDGET STYLING
# ============================================================

st.markdown(
    """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
    --bg-base: #070b14;
    --bg-surface: #0f1624;
    --bg-elevated: #151f32;
    --border-subtle: rgba(148, 163, 184, 0.14);
    --border-accent: rgba(45, 212, 191, 0.35);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent-teal: #2dd4bf;
    --accent-blue: #3b82f6;
}

.stApp {
    background:
        radial-gradient(circle at 85% 5%, rgba(59, 130, 246, 0.14), transparent 32%),
        radial-gradient(circle at 10% 15%, rgba(45, 212, 191, 0.09), transparent 28%),
        var(--bg-base);
    color: var(--text-primary);
    font-family: 'Inter', sans-serif;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, h4 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    letter-spacing: -0.02em;
}

.block-container {
    max-width: 1440px;
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
}

section[data-testid="stSidebar"] {
    background: #0a101d;
    border-right: 1px solid var(--border-subtle);
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(160deg, rgba(17, 25, 40, 0.92), rgba(11, 17, 30, 0.95));
    border: 1px solid var(--border-subtle) !important;
    border-radius: 18px !important;
    padding: 1.15rem 1.25rem !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

div[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, 0.65);
    border: 1.5px dashed rgba(59, 130, 246, 0.42);
    border-radius: 14px;
    padding: 0.75rem;
    transition: border-color 0.2s ease, background 0.2s ease;
}

div[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-teal);
    background: rgba(20, 30, 52, 0.85);
}

.stButton > button[kind="primary"],
.stButton > button {
    background: linear-gradient(135deg, #0d9488 0%, #2563eb 100%);
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 0.95rem;
    padding: 0.7rem 1.25rem;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.28);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 26px rgba(45, 212, 191, 0.3);
    border-color: rgba(255, 255, 255, 0.25);
    color: #ffffff;
}

button[data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--text-secondary);
    padding: 0.55rem 1rem;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-teal) !important;
}

div[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 0.85rem 1rem;
}

div[data-testid="stMetricLabel"] {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
}

div[data-testid="stMetricValue"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--text-primary);
}

.nv-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 11px;
    border-radius: 999px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    border: 1px solid rgba(45, 212, 191, 0.28);
    background: rgba(45, 212, 191, 0.1);
    color: #5eead4;
    margin-right: 6px;
}

.nv-kicker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5eead4;
    margin-bottom: 4px;
}

.nv-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.85rem;
    font-weight: 800;
    color: #f8fafc;
    line-height: 1.2;
    margin: 0;
}

.nv-sub {
    color: #94a3b8;
    font-size: 0.92rem;
    margin-top: 6px;
    margin-bottom: 0;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>""",
    unsafe_allow_html=True,
)


# ============================================================
# 4. MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return tf.keras.models.load_model(model_path, compile=False)


# ============================================================
# 5. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(rgb_image):
    resized = cv2.resize(
        rgb_image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )
    image = resized.astype(np.float32)

    if PREPROCESSING_MODE == "resnet":
        image = tf.keras.applications.resnet50.preprocess_input(image)
    elif PREPROCESSING_MODE == "rescale":
        image = image / 255.0
    elif PREPROCESSING_MODE == "none":
        pass
    else:
        raise ValueError("Invalid preprocessing configuration.")

    return np.expand_dims(image, axis=0)


# ============================================================
# 6. ROI EXTRACTION
# ============================================================

def extract_roi(rgb_image):
    height, width = rgb_image.shape[:2]
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, threshold = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel_size = max(3, ROI_KERNEL_SIZE | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        closed,
        connectivity=8,
    )

    if count <= 1:
        return rgb_image.copy(), (0, 0, width, height)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_index = int(np.argmax(areas)) + 1
    area = stats[largest_index, cv2.CC_STAT_AREA]

    if area < height * width * ROI_MIN_AREA:
        return rgb_image.copy(), (0, 0, width, height)

    x = stats[largest_index, cv2.CC_STAT_LEFT]
    y = stats[largest_index, cv2.CC_STAT_TOP]
    w = stats[largest_index, cv2.CC_STAT_WIDTH]
    h = stats[largest_index, cv2.CC_STAT_HEIGHT]

    pad_x = int(w * ROI_PADDING)
    pad_y = int(h * ROI_PADDING)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    roi = rgb_image[y1:y2, x1:x2]
    if roi.size == 0:
        return rgb_image.copy(), (0, 0, width, height)

    return roi, (x1, y1, x2, y2)


# ============================================================
# 7. PREDICTION
# ============================================================

def predict(model, image):
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)
    probabilities = np.asarray(predictions).squeeze().astype(np.float64)

    if probabilities.ndim != 1 or len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected four class scores, got {probabilities.shape}"
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Model returned invalid prediction values.")

    return probabilities


def get_class_name(probabilities):
    return CLASS_NAMES[int(np.argmax(probabilities))]


# ============================================================
# 8. GRAD-CAM
# ============================================================

def find_target_layer(container, preferred_name="conv5_block3_out"):
    try:
        return container.get_layer(preferred_name)
    except Exception:
        pass

    for layer in reversed(container.layers):
        out_shape = getattr(layer, "output_shape", None)
        if out_shape is None and hasattr(layer, "output"):
            out_shape = tuple(layer.output.shape)
        if out_shape is not None and len(out_shape) == 4 and not isinstance(
            layer, tf.keras.layers.InputLayer
        ):
            return layer

    raise ValueError(f"Could not find a 4D target conv layer in {container.name}.")


def inspect_architecture(model, preferred_name="conv5_block3_out"):
    for idx, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.Model):
            target_layer = find_target_layer(layer, preferred_name)
            return "nested", idx, layer, target_layer

    target_layer = find_target_layer(model, preferred_name)
    return "flat", None, None, target_layer


@st.cache_resource(show_spinner=False)
def build_gradcam_runner(model_path):
    model = load_model(model_path)
    arch_type, backbone_idx, backbone, target_layer = inspect_architecture(model)

    if arch_type == "flat":
        unified_model = tf.keras.Model(
            inputs=model.input,
            outputs=[target_layer.output, model.output],
        )
        return ("flat", unified_model, [], [])

    backbone_extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=[target_layer.output, backbone.output],
    )

    pre_layers = [
        layer
        for layer in model.layers[:backbone_idx]
        if not isinstance(layer, tf.keras.layers.InputLayer)
    ]
    post_layers = list(model.layers[backbone_idx + 1 :])

    return ("nested", backbone_extractor, pre_layers, post_layers)


def generate_gradcam(rgb_image, model_path, class_index):
    arch_type, extractor, pre_layers, post_layers = build_gradcam_runner(model_path)
    input_tensor = tf.convert_to_tensor(
        preprocess_image(rgb_image),
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        if arch_type == "flat":
            conv_outputs, predictions = extractor(input_tensor, training=False)
        else:
            x = input_tensor
            for layer in pre_layers:
                x = layer(x, training=False)

            conv_outputs, backbone_out = extractor(x, training=False)

            out = backbone_out
            for layer in post_layers:
                out = layer(out, training=False)
            predictions = out

        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients could not be calculated.")

    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    heatmap = tf.reduce_sum(weights * conv_outputs, axis=-1)
    heatmap = tf.nn.relu(heatmap)[0]
    heatmap = heatmap / (
        tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    )
    return heatmap.numpy()


def create_overlay(rgb_image, heatmap):
    height, width = rgb_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (width, height))
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0, 1))
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_image, 0.62, colored, 0.38, 0)


def map_roi_cam_to_original(original, roi_heatmap, bbox):
    x1, y1, x2, y2 = bbox
    height, width = original.shape[:2]
    mapped = np.zeros((height, width), dtype=np.float32)
    mapped[y1:y2, x1:x2] = cv2.resize(
        roi_heatmap,
        (x2 - x1, y2 - y1),
    )
    return mapped


# ============================================================
# 9. VISUALIZATION HELPERS
# ============================================================

def render_progress_row(label, percentage, color="#2dd4bf"):
    pct = float(np.clip(percentage, 0.0, 100.0))
    html = (
        f'<div style="margin-bottom:14px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
        f'<span style="color:#e2e8f0;font-weight:600;font-size:0.9rem;">{label}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;color:{color};font-weight:700;font-size:0.88rem;">{pct:.2f}%</span>'
        f'</div>'
        f'<div style="width:100%;height:8px;background:rgba(30,41,59,0.9);border-radius:999px;overflow:hidden;">'
        f'<div style="width:{pct}%;height:100%;background:{color};border-radius:999px;"></div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def show_probability_chart(probabilities):
    labels = [CLASS_DISPLAY[name][0] for name in CLASS_NAMES]
    colors = [CLASS_DISPLAY[name][1] for name in CLASS_NAMES]

    df = pd.DataFrame({
        "Class": labels,
        "Score": probabilities * 100,
        "Color": colors,
    }).sort_values("Score", ascending=True)

    fig = px.bar(
        df,
        x="Score",
        y="Class",
        orientation="h",
        text=df["Score"].map(lambda x: f"{x:.1f}%"),
        color="Class",
        color_discrete_map=dict(zip(df["Class"], df["Color"])),
        range_x=[0, 112],
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=12, color="#e2e8f0"),
        marker_line_width=0,
        cliponaxis=False,
    )

    fig.update_layout(
        height=270,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#cbd5e1"),
        margin=dict(l=8, r=28, t=10, b=10),
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.1)",
            zeroline=False,
            ticksuffix="%",
        ),
        yaxis=dict(title="", showgrid=False),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )


# ============================================================
# 10. SIDEBAR CONTROL PANEL
# ============================================================

with st.sidebar:
    st.markdown("## 🧠 NeuroVision")
    st.caption("Dual-Stream Brain MRI Diagnostic Suite")
    st.divider()

    st.markdown("**INFERENCE ARCHITECTURE**")
    mode = st.radio(
        "Prediction mode",
        [
            "Model Fusion - Ensemble",
            "Model A - Original",
            "Model B - ROI",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**PIPELINE TELEMETRY**")

    with st.container(border=True):
        st.markdown("**Model A** · Full-Frame ResNet50")
        st.caption("Captures global cranial & structural context")
        st.markdown("**Model B** · ROI-Cropped ResNet50")
        st.caption("Focuses on Otsu-extracted brain parenchyma")
        if mode == "Model Fusion - Ensemble":
            st.caption(
                f"Fusion Policy: Soft Voting ({int(FUSION_WEIGHT_A*100)}% A / {int(FUSION_WEIGHT_B*100)}% B)"
            )

    st.divider()
    st.markdown(
        '<span class="nv-badge">● LOCAL IN-MEMORY</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Scans are processed locally in RAM. Research & educational prototype only."
    )


# ============================================================
# 11. COMPACT TOP HEADER BAR
# ============================================================

header_left, header_right = st.columns([1.6, 1.0], vertical_alignment="center")

with header_left:
    st.markdown(
        '<div class="nv-kicker">AI-ASSISTED NEUROIMAGING WORKSPACE</div>'
        '<h1 class="nv-title">Brain MRI Tumor Classification</h1>'
        '<p class="nv-sub">Dual ResNet50 inference with Otsu ROI localization and Grad-CAM visual attribution.</p>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        '<div style="text-align:right;padding-top:8px;">'
        '<span class="nv-badge">RESNET50 × 2</span>'
        '<span class="nv-badge">224×224 INPUT</span>'
        '<span class="nv-badge">GRAD-CAM</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.write("")


# ============================================================
# 12. MAIN BENTO WORKSPACE (LEFT: INPUT | RIGHT: DIAGNOSTICS)
# ============================================================

left_col, right_col = st.columns([1.0, 1.55], gap="large")

with left_col:
    with st.container(border=True):
        st.markdown("#### 01 · Scan Acquisition")
        uploaded_file = st.file_uploader(
            "Drop MRI slice here (PNG, JPG, BMP)",
            type=["png", "jpg", "jpeg", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded_file is None:
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("upload_hash", None)

            st.info(
                "Upload an axial, coronal, or sagittal T1/T2 brain MRI image to begin analysis."
            )
            st.markdown("**Target Diagnostic Classes**")
            c_a, c_b = st.columns(2)
            with c_a:
                st.caption("🔴 **Glioma**")
                st.caption("🟡 **Meningioma**")
            with c_b:
                st.caption("🟢 **No Tumor**")
                st.caption("🟣 **Pituitary**")
        else:
            image_bytes = uploaded_file.getvalue()
            current_hash = hash(image_bytes)

            if st.session_state.get("upload_hash") != current_hash:
                st.session_state["upload_hash"] = current_hash
                st.session_state.pop("analysis_result", None)

            original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            original_rgb = np.asarray(original_image, dtype=np.uint8)
            h, w = original_rgb.shape[:2]

            st.image(
                original_rgb,
                caption=f"Specimen: {uploaded_file.name}",
                use_container_width=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Width", f"{w} px")
            m2.metric("Height", f"{h} px")
            m3.metric("Input", f"{IMAGE_SIZE[0]}²")

            st.write("")
            analyze = st.button(
                "✦ Run Diagnostic Analysis",
                type="primary",
                use_container_width=True,
            )

            if analyze:
                progress = st.progress(0, text="Initializing inference...")
                try:
                    predictions = {}
                    roi_image, roi_bbox = extract_roi(original_rgb)

                    need_a = mode in [
                        "Model A - Original",
                        "Model Fusion - Ensemble",
                    ]
                    need_b = mode in [
                        "Model B - ROI",
                        "Model Fusion - Ensemble",
                    ]

                    with st.spinner("Running neural network inference..."):
                        if need_a:
                            progress.progress(20, text="Loading Model A (Original)...")
                            model_a = load_model(MODEL_A_PATH)
                            progress.progress(40, text="Running Model A forward pass...")
                            predictions["Model A"] = predict(model_a, original_rgb)

                        if need_b:
                            progress.progress(55, text="Extracting ROI & loading Model B...")
                            model_b = load_model(MODEL_B_PATH)
                            progress.progress(70, text="Running Model B forward pass...")
                            predictions["Model B"] = predict(model_b, roi_image)

                        if mode == "Model A - Original":
                            final_scores = predictions["Model A"]
                        elif mode == "Model B - ROI":
                            final_scores = predictions["Model B"]
                        else:
                            final_scores = (
                                FUSION_WEIGHT_A * predictions["Model A"]
                                + FUSION_WEIGHT_B * predictions["Model B"]
                            )

                        progress.progress(82, text="Computing Grad-CAM attention maps...")
                        gradcam_results = {}
                        gradcam_errors = {}

                        if need_a:
                            try:
                                cam_a = generate_gradcam(
                                    original_rgb,
                                    MODEL_A_PATH,
                                    int(np.argmax(predictions["Model A"])),
                                )
                                gradcam_results["Model A (Original)"] = create_overlay(
                                    original_rgb,
                                    cam_a,
                                )
                            except Exception as error:
                                gradcam_errors["Model A"] = str(error)

                        if need_b:
                            try:
                                cam_b = generate_gradcam(
                                    roi_image,
                                    MODEL_B_PATH,
                                    int(np.argmax(predictions["Model B"])),
                                )
                                mapped_cam = map_roi_cam_to_original(
                                    original_rgb,
                                    cam_b,
                                    roi_bbox,
                                )
                                gradcam_results["Model B (ROI-Mapped)"] = create_overlay(
                                    original_rgb,
                                    mapped_cam,
                                )
                            except Exception as error:
                                gradcam_errors["Model B"] = str(error)

                        st.session_state["analysis_result"] = {
                            "mode": mode,
                            "scores": final_scores,
                            "individual": predictions,
                            "gradcams": gradcam_results,
                            "gradcam_errors": gradcam_errors,
                            "original": original_rgb,
                            "roi": roi_image,
                            "roi_bbox": roi_bbox,
                        }

                        progress.progress(100, text="Analysis complete.")
                        progress.empty()

                except Exception as error:
                    progress.empty()
                    st.error(f"Analysis failed: {error}")
                    st.exception(error)


# ============================================================
# 13. RIGHT BENTO COLUMN: DIAGNOSTIC INTELLIGENCE SUITE
# ============================================================

with right_col:
    result = st.session_state.get("analysis_result")

    if result is None:
        with st.container(border=True):
            st.markdown("#### 02 · Diagnostic Intelligence")
            st.caption("Awaiting scan execution")
            st.write("")
            st.markdown(
                "Once you upload an MRI scan and click **Run Diagnostic Analysis**, this workspace will populate with:"
            )
            st.markdown(
                "- **Primary Classification & Confidence** across all 4 neuro-oncology categories\n"
                "- **Model Consensus Check** comparing full-frame vs. ROI-cropped predictions\n"
                "- **Grad-CAM Visual Attribution** highlighting discriminative spatial features\n"
                "- **Automated ROI Extraction** showing the Otsu bounding-box crop fed to Model B"
            )
    else:
        scores = result["scores"]
        raw_class = get_class_name(scores)
        display_name, accent_hex, accent_bg = CLASS_DISPLAY[raw_class]
        top_score = float(np.max(scores)) * 100.0

        with st.container(border=True):
            banner_html = (
                f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">'
                f'<div>'
                f'<div class="nv-kicker" style="color:{accent_hex};">PRIMARY CLASSIFICATION · {result["mode"].upper()}</div>'
                f'<div style="font-family:\'Plus Jakarta Sans\',sans-serif;font-size:2.1rem;font-weight:800;color:#f8fafc;line-height:1.15;">{display_name}</div>'
                f'</div>'
                f'<div style="background:{accent_bg};border:1px solid {accent_hex};border-radius:14px;padding:10px 18px;text-align:right;">'
                f'<div style="font-size:0.7rem;letter-spacing:0.08em;color:#cbd5e1;font-weight:600;">CONFIDENCE</div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:1.55rem;font-weight:700;color:{accent_hex};">{top_score:.2f}%</div>'
                f'</div>'
                f'</div>'
            )
            st.markdown(banner_html, unsafe_allow_html=True)

            st.write("")
            k1, k2, k3 = st.columns(3)
            k1.metric("Predicted Class", display_name)
            k2.metric("Top Confidence", f"{top_score:.2f}%")

            if (
                "Model A" in result["individual"]
                and "Model B" in result["individual"]
            ):
                pred_a = CLASS_DISPLAY[
                    get_class_name(result["individual"]["Model A"])
                ][0]
                pred_b = CLASS_DISPLAY[
                    get_class_name(result["individual"]["Model B"])
                ][0]
                consensus = "Agreed" if pred_a == pred_b else "Divergent"
                k3.metric("Model Consensus", consensus, f"A: {pred_a} · B: {pred_b}")
            else:
                k3.metric("Inference Stream", "Single Model", result["mode"])

        with st.container(border=True):
            tab_probs, tab_cam, tab_compare, tab_roi = st.tabs([
                "📊 Class Probabilities",
                "🔥 Grad-CAM Attribution",
                "⚖️ Model A vs B",
                "✂️ ROI Extraction",
            ])

            with tab_probs:
                c_chart, c_bars = st.columns([1.25, 0.95], gap="medium")
                with c_chart:
                    show_probability_chart(scores)
                with c_bars:
                    st.markdown("**Ranked Confidence Breakdown**")
                    st.write("")
                    for cls_key, val in sorted(
                        zip(CLASS_NAMES, scores),
                        key=lambda x: x[1],
                        reverse=True,
                    ):
                        lbl, clr, _ = CLASS_DISPLAY[cls_key]
                        render_progress_row(lbl, float(val) * 100.0, clr)

            with tab_cam:
                st.caption(
                    "Grad-CAM highlights spatial regions driving the model's decision. It is a visual attribution aid, not a clinical segmentation mask."
                )
                if result["gradcams"]:
                    cam_cols = st.columns(len(result["gradcams"]))
                    for idx, (cam_title, cam_img) in enumerate(
                        result["gradcams"].items()
                    ):
                        with cam_cols[idx]:
                            st.image(
                                cam_img,
                                caption=cam_title,
                                use_container_width=True,
                            )
                for name, err in result["gradcam_errors"].items():
                    st.warning(f"{name} Grad-CAM unavailable: {err}")
                if not result["gradcams"]:
                    st.info(
                        "Grad-CAM could not be generated for this model architecture."
                    )

            with tab_compare:
                if len(result["individual"]) > 1:
                    comp_cols = st.columns(len(result["individual"]))
                    for idx, (m_name, m_scores) in enumerate(
                        result["individual"].items()
                    ):
                        with comp_cols[idx]:
                            m_cls = get_class_name(m_scores)
                            m_disp, m_clr, _ = CLASS_DISPLAY[m_cls]
                            st.markdown(f"#### {m_name}")
                            st.metric(
                                "Top Prediction",
                                m_disp,
                                f"{np.max(m_scores)*100:.2f}% confidence",
                            )
                            st.write("")
                            for cls_key, val in zip(CLASS_NAMES, m_scores):
                                lbl, clr, _ = CLASS_DISPLAY[cls_key]
                                render_progress_row(lbl, float(val) * 100.0, clr)
                else:
                    only_name = next(iter(result["individual"].keys()))
                    st.info(
                        f"Currently running in single-stream mode (**{only_name}**). Select **Model Fusion - Ensemble** in the sidebar to compare both models side by side."
                    )

            with tab_roi:
                x1, y1, x2, y2 = result["roi_bbox"]
                st.caption(
                    f"Otsu thresholding + morphological closing bounding box: (x1={x1}, y1={y1}) → (x2={x2}, y2={y2})"
                )
                r1, r2 = st.columns(2)
                with r1:
                    st.image(
                        result["original"],
                        caption="Original Full-Frame MRI",
                        use_container_width=True,
                    )
                with r2:
                    st.image(
                        result["roi"],
                        caption="Extracted ROI Input (Model B)",
                        use_container_width=True,
                    )