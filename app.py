"""Streamlit app: classify an image as a cat or a dog (MobileNetV2)."""
import io
import textwrap
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # no GUI window, safe for servers
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from PIL import Image, ImageOps, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "cat_dog_mobilenetv2.keras"

IMG_SIZE = 224
CLASS_NAMES = ["cat", "dog"]  # 0 = cat, 1 = dog (label order in TFDS cats_vs_dogs)
LABEL_NAME = {"cat": "Cat", "dog": "Dog"}
EMOJI = {"cat": "🐱", "dog": "🐶"}
SURE_THRESHOLD = 0.90  # at or above this: "Very confident"
MIN_THRESHOLD = 0.70   # below this: "Not sure"


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    import tensorflow as tf  # imported here so the page renders quickly

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess(image: Image.Image) -> np.ndarray:
    """Same as training: resize to 224x224, pixel values 0-255.
    (Rescaling to [-1, 1] is already INSIDE the model.)"""
    import tensorflow as tf

    image = ImageOps.exif_transpose(image).convert("RGB")
    arr = np.asarray(image, dtype="float32")
    arr = tf.image.resize(arr, (IMG_SIZE, IMG_SIZE)).numpy()
    return arr[np.newaxis, ...]


def confidence_level(confidence: float) -> str:
    if confidence >= SURE_THRESHOLD:
        return "Very confident"
    if confidence >= MIN_THRESHOLD:
        return "Fairly confident"
    return "Not sure"


def build_conclusion(label: str, p_cat: float, p_dog: float) -> str:
    confidence = max(p_cat, p_dog)
    level = confidence_level(confidence)
    if level == "Not sure":
        return (
            f"The model is not sure (highest confidence is only {confidence:.1%}). "
            "The image may not be a cat or a dog, or the subject may be unclear."
        )
    return (
        f"This image is predicted to be a {LABEL_NAME[label].lower()} {EMOJI[label]} "
        f"with {confidence:.1%} confidence ({level.lower()})."
    )


def predict_image(image: Image.Image) -> dict:
    model = load_model()
    p_dog = float(model.predict(preprocess(image), verbose=0)[0][0])
    p_dog = min(max(p_dog, 0.0), 1.0)
    p_cat = 1.0 - p_dog
    label = "dog" if p_dog >= 0.5 else "cat"
    confidence = max(p_cat, p_dog)
    return {
        "label": label,
        "confidence": confidence,
        "confidence_level": confidence_level(confidence),
        "cat_score": p_cat,
        "dog_score": p_dog,
        "conclusion": build_conclusion(label, p_cat, p_dog),
    }


@st.cache_data(show_spinner="Predicting...")
def predict_bytes(data: bytes) -> dict:
    return predict_image(Image.open(io.BytesIO(data)))


def read_csv_safe(name: str, index_col=None):
    path = MODEL_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=index_col)


def csv_signature() -> tuple:
    """List of (name, modified time) for all CSVs; used so the image cache
    refreshes when a CSV file is replaced."""
    if not MODEL_DIR.exists():
        return ()
    return tuple((p.name, p.stat().st_mtime) for p in sorted(MODEL_DIR.glob("*.csv")))


# ---------------------------------------------------------------- matplotlib reports
def fig_to_png(fig: Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return buf.getvalue()


def load_report_data():
    summary = read_csv_safe("metrics_summary.csv")
    cm_df = read_csv_safe("confusion_matrix.csv", index_col=0)
    if summary is None or cm_df is None:
        return None
    m = dict(zip(summary["metric"], summary["value"]))
    class_names = [c.replace("pred_", "") for c in cm_df.columns]
    return m, cm_df.to_numpy(dtype=int), class_names


def draw_confusion_matrix(fig: Figure, ax, cm: np.ndarray, class_names, fontsize=14):
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=fontsize)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def build_final_report(m: dict, cm: np.ndarray, class_names) -> Figure:
    accuracy = m["test_accuracy"] * 100
    error = m["error_rate"] * 100
    macro_p = m["macro_precision"] * 100
    macro_r = m["macro_recall"] * 100
    macro_f1 = m["macro_f1"] * 100
    loss = m["test_loss"]
    total = int(m["total_test_images"])
    correct = int(m["correct"])
    incorrect = int(m["incorrect"])

    if accuracy >= 95:
        performance = "Excellent test performance"
    elif accuracy >= 85:
        performance = "Good test performance"
    elif accuracy >= 70:
        performance = "Moderate test performance"
    else:
        performance = "Further model improvement is needed"

    conclusion = (
        f"The MobileNetV2 model achieved {accuracy:.2f}% test accuracy, "
        f"with a test loss of {loss:.4f}. It correctly classified {correct} "
        f"out of {total} test images and incorrectly classified {incorrect}. "
        f"The error rate was {error:.2f}%. Macro precision was {macro_p:.2f}%, "
        f"macro recall was {macro_r:.2f}%, and macro F1-score was {macro_f1:.2f}%. "
        f"Overall, the model showed {performance.lower()}. "
        "Results are from a held-out test set that was not used for "
        "training or model selection."
    )

    fig = Figure(figsize=(16, 12))
    fig.suptitle(
        "FINAL PROJECT REPORT\nCAT AND DOG IMAGE CLASSIFICATION USING MobileNetV2",
        fontsize=17, fontweight="bold", y=0.98,
    )

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.axis("off")
    model_info = (
        "MODEL INFORMATION\n\n"
        "Architecture : MobileNetV2\n"
        "Approach     : Transfer Learning\n"
        "Framework    : TensorFlow / Keras\n"
        "Task         : Binary Image Classification\n"
        f"Classes      : {class_names[0]} and {class_names[1]}\n"
        "Loss         : Binary Cross-Entropy\n"
        "Optimizer    : Adam\n"
        "Regularization: Data Augmentation, Dropout, Early Stopping"
    )
    ax1.text(0.02, 0.95, model_info, va="top", fontsize=11, linespacing=1.5)
    ax1.set_title("1. Methodology", loc="left", fontweight="bold")

    ax2 = fig.add_subplot(2, 2, 2)
    names = ["Accuracy", "Error Rate", "Macro Precision", "Macro Recall", "Macro F1-score"]
    values = [accuracy, error, macro_p, macro_r, macro_f1]
    bars = ax2.barh(names, values)
    ax2.set_xlim(0, 115)
    ax2.set_xlabel("Percentage (%)")
    ax2.set_title("2. Evaluation Metrics", fontweight="bold")
    ax2.invert_yaxis()
    for bar, value in zip(bars, values):
        ax2.text(value + 1, bar.get_y() + bar.get_height() / 2,
                 f"{value:.2f}%", va="center", fontsize=9)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("3. Confusion Matrix", fontweight="bold")
    draw_confusion_matrix(fig, ax3, cm, class_names)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    ax4.set_title("4. Overall Conclusion", loc="left", fontweight="bold")
    ax4.text(0.02, 0.95, textwrap.fill(conclusion, width=62),
             va="top", fontsize=11, linespacing=1.6)
    ax4.text(0.02, 0.12,
             f"Test Loss: {loss:.4f}\n"
             f"Correct Predictions: {correct}/{total}\n"
             f"Incorrect Predictions: {incorrect}/{total}",
             va="top", fontsize=10, fontweight="bold")

    fig.text(0.5, 0.015,
             "Note: Results reflect a held-out test set not used for model selection.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig


def build_dashboard(m: dict, report: pd.DataFrame, cm: np.ndarray, class_names) -> Figure:
    accuracy = m["test_accuracy"] * 100
    error = m["error_rate"] * 100
    loss = m["test_loss"]
    total = int(m["total_test_images"])
    correct = int(m["correct"])
    incorrect = int(m["incorrect"])

    precision = [report.loc[c, "precision"] * 100 for c in class_names]
    recall = [report.loc[c, "recall"] * 100 for c in class_names]
    f1 = [report.loc[c, "f1-score"] * 100 for c in class_names]

    fig = Figure(figsize=(18, 11))
    fig.suptitle("MOBILENETV2 — MODEL STATISTICAL REPORT", fontsize=18, fontweight="bold")
    axes = fig.subplots(2, 3)

    ax = axes[0, 0]
    bars = ax.bar(["Accuracy", "Error Rate"], [accuracy, error])
    ax.set_title("Prediction Performance")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 110)
    for bar, value in zip(bars, [accuracy, error]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{value:.2f}%", ha="center", fontweight="bold")

    ax = axes[0, 1]
    ax.pie([correct, incorrect], labels=["Correct", "Incorrect"],
           autopct="%1.2f%%", startangle=90)
    ax.set_title(f"Prediction Distribution (N={total})")

    ax = axes[0, 2]
    ax.bar(["Test Loss"], [loss])
    ax.set_title("Test Loss (Binary Cross-Entropy)")
    ax.set_ylabel("Loss")
    ax.text(0, loss, f"{loss:.4f}", ha="center", va="bottom", fontweight="bold")

    ax = axes[1, 0]
    ax.set_title("Confusion Matrix")
    draw_confusion_matrix(fig, ax, cm, class_names)

    ax = axes[1, 1]
    x = np.arange(len(class_names))
    width = 0.25
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-score")
    ax.set_title("Classification Metrics by Class")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.axis("off")
    summary = (
        "MODEL SUMMARY\n\n"
        "Architecture: MobileNetV2\n"
        "Method: Transfer Learning\n"
        "Task: Cat vs Dog Classification\n\n"
        f"Test images: {total}\n"
        f"Correct: {correct}\n"
        f"Incorrect: {incorrect}\n\n"
        f"Accuracy: {accuracy:.2f}%\n"
        f"Error rate: {error:.2f}%\n"
        f"Test loss: {loss:.4f}\n\n"
        "Metrics are based on the held-out test set.\n"
        "Model selection used the validation set only."
    )
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, va="top",
            fontsize=10, linespacing=1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def build_training_curves(history: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(12, 5))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(history["epoch"], history["accuracy"], marker="o", label="Training")
    ax1.plot(history["epoch"], history["val_accuracy"], marker="o", label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax2.plot(history["epoch"], history["loss"], marker="o", label="Training")
    ax2.plot(history["epoch"], history["val_loss"], marker="o", label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner="Building final report...")
def final_report_png(sig: tuple) -> Optional[bytes]:
    data = load_report_data()
    if data is None:
        return None
    m, cm, class_names = data
    return fig_to_png(build_final_report(m, cm, class_names))


@st.cache_data(show_spinner="Building dashboard...")
def dashboard_png(sig: tuple) -> Optional[bytes]:
    data = load_report_data()
    report = read_csv_safe("classification_report.csv", index_col=0)
    if data is None or report is None:
        return None
    m, cm, class_names = data
    return fig_to_png(build_dashboard(m, report, cm, class_names))


@st.cache_data(show_spinner="Building training curves...")
def curves_png(sig: tuple) -> Optional[bytes]:
    history = read_csv_safe("training_history.csv")
    if history is None:
        return None
    return fig_to_png(build_training_curves(history))


# ---------------------------------------------------------------- UI
def render_result_card(name: str, data: bytes, res: dict) -> None:
    with st.container(border=True):
        left, right = st.columns([1, 1.3])
        with left:
            st.image(data, caption=name, width=280)
        with right:
            if res["confidence_level"] == "Not sure":
                st.subheader("❓ Not sure")
                st.warning(res["conclusion"])
            else:
                st.subheader(f"{EMOJI[res['label']]} {LABEL_NAME[res['label']]}")
                st.success(res["conclusion"])
            st.progress(res["cat_score"], text=f"Cat 🐱: {res['cat_score']:.2%}")
            st.progress(res["dog_score"], text=f"Dog 🐶: {res['dog_score']:.2%}")


def render_prediction_tab() -> None:
    files = st.file_uploader(
        "Upload cat or dog images (you can select several)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if not files:
        st.info("No images yet. Upload an image to see its prediction.")
        return

    rows = []
    for f in files:
        data = f.getvalue()
        try:
            res = predict_bytes(data)
        except (UnidentifiedImageError, OSError):
            st.error(f"{f.name}: not a valid image file.")
            continue
        render_result_card(f.name, data, res)
        rows.append({
            "file": f.name,
            "prediction": "uncertain" if res["confidence_level"] == "Not sure"
                          else LABEL_NAME[res["label"]].lower(),
            "confidence_level": res["confidence_level"],
            "confidence": round(res["confidence"], 4),
            "cat_score": round(res["cat_score"], 4),
            "dog_score": round(res["dog_score"], 4),
            "conclusion": res["conclusion"],
        })

    if not rows:
        return

    df = pd.DataFrame(rows)
    st.divider()
    st.subheader("Conclusion")
    n_cat = int((df["prediction"] == "cat").sum())
    n_dog = int((df["prediction"] == "dog").sum())
    n_unsure = int((df["prediction"] == "uncertain").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total images", len(df))
    c2.metric("Cats 🐱", n_cat)
    c3.metric("Dogs 🐶", n_dog)
    c4.metric("Not sure ❓", n_unsure)

    st.dataframe(df, hide_index=True)
    st.download_button(
        "⬇️ Download predictions (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="predictions.csv",
        mime="text/csv",
    )


def render_model_tab() -> None:
    summary = read_csv_safe("metrics_summary.csv")
    if summary is None:
        st.warning("metrics_summary.csv was not found in the model/ folder.")
        return

    sig = csv_signature()
    m = dict(zip(summary["metric"], summary["value"]))
    st.subheader("Model scores on the test set")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['test_accuracy']:.2%}")
    c2.metric("Macro F1", f"{m['macro_f1']:.2%}")
    c3.metric("Loss", f"{m['test_loss']:.4f}")
    c4.metric("Test images", f"{int(m['total_test_images'])}")
    st.caption(
        f"Correct {int(m['correct'])} of {int(m['total_test_images'])} test images "
        f"· wrong {int(m['incorrect'])} · error rate {m['error_rate']:.2%}"
    )

    st.divider()
    st.subheader("Final report")
    png = final_report_png(sig)
    if png:
        st.image(png)
        st.download_button("⬇️ Download final report (PNG)", data=png,
                           file_name="final_report.png", mime="image/png")
    else:
        st.warning("The final report needs metrics_summary.csv and confusion_matrix.csv.")

    png = dashboard_png(sig)
    if png:
        st.subheader("Statistical dashboard")
        st.image(png)
        st.download_button("⬇️ Download dashboard (PNG)", data=png,
                           file_name="statistical_dashboard.png", mime="image/png")

    png = curves_png(sig)
    if png:
        st.subheader("Training curves")
        st.image(png)

    with st.expander("Show data tables (CSV)"):
        report = read_csv_safe("classification_report.csv", index_col=0)
        if report is not None:
            st.write("Precision, recall, F1 per class")
            st.dataframe(report.round(4))
        cm = read_csv_safe("confusion_matrix.csv", index_col=0)
        if cm is not None:
            st.write("Confusion matrix")
            st.dataframe(cm)
        history = read_csv_safe("training_history.csv")
        if history is not None:
            st.write("Training history")
            st.dataframe(history, hide_index=True)

    st.info(
        "This model only knows two classes (cat and dog). Images of other animals "
        "or objects will still be forced into one of them, and the numbers above "
        "only apply to images similar to the test data."
    )


def main() -> None:
    st.set_page_config(page_title="Cat or Dog?", page_icon="🐾")
    st.title("🐾 Cat or Dog Detector")
    st.caption("MobileNetV2 (transfer learning) · trained on Google Colab")

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()

    tab_predict, tab_model = st.tabs(["Image prediction", "Model scores"])
    with tab_predict:
        render_prediction_tab()
    with tab_model:
        render_model_tab()


if __name__ == "__main__":
    main()