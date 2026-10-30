"""Streamlit dashboard for the Smart Parking Space Detection demo."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import streamlit as st

from src.pipeline import ParkingPipeline

DATA_DIR = Path("data/samples")
SPOTS_DIR = Path("spots")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


METHOD_LABELS = {
    "classifier": "Spot classifier (CNN): is this spot occupied?",
    "yolo": "YOLO vehicle detection + overlap",
}


@st.cache_resource
def load_pipeline(spots_path: str, method: str, settings: tuple[tuple[str, object], ...]) -> ParkingPipeline:
    return ParkingPipeline(spots_path, method=method, **dict(settings))


def list_files(directory: Path, extensions: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in extensions)


def main() -> None:
    st.set_page_config(page_title="Smart Parking Space Detection", layout="wide")
    st.title("Smart Parking Space Detection")
    st.caption(
        "Fixed-camera parking lot occupancy from predefined parking-spot polygons, "
        "using a per-spot CNN classifier or YOLOv8 vehicle detection."
    )

    spot_files = list_files(SPOTS_DIR, {".json"})
    image_files = list_files(DATA_DIR, IMAGE_EXTENSIONS)
    video_files = list_files(DATA_DIR, VIDEO_EXTENSIONS)

    with st.sidebar:
        st.header("Setup")
        if not spot_files:
            st.warning(
                "No spot definitions found in `spots/`. Run "
                "`python -m src.calibrate <image> <output.json>` first."
            )
            return
        media_type = st.radio("Source type", ["Image", "Video"])
        media_files = image_files if media_type == "Image" else video_files
        media_path = st.selectbox(f"Sample {media_type.lower()}", media_files, format_func=str)

        # Spot polygons only make sense on the camera view they were drawn for,
        # so default to the spots file named like the selected image/video.
        matching = [i for i, p in enumerate(spot_files) if media_path and p.stem == media_path.stem]
        spots_path = st.selectbox(
            "Parking-spot definition",
            spot_files,
            index=matching[0] if matching else 0,
            format_func=str,
        )
        if media_path and spots_path.stem != media_path.stem:
            st.warning("This spot file was not made for the selected image, so results will be meaningless.")

        method = st.radio("Detection method", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
        if method == "classifier":
            settings = {
                "probability_threshold": st.slider("Occupied probability threshold", 0.05, 0.95, 0.5, 0.05),
                "model_path": st.text_input("Classifier weights", value="models/spot_classifier.pt"),
            }
        else:
            settings = {
                "overlap_threshold": st.slider("Occupied overlap threshold", 0.05, 0.9, 0.3, 0.05),
                "confidence": st.slider(
                    "Vehicle detection confidence",
                    0.05,
                    0.9,
                    0.35,
                    0.05,
                    help="Lower it if visible cars are missed (common in high, overhead views).",
                ),
                "model_path": st.text_input("YOLO model weights", value="yolov8n.pt"),
            }
        settings_key = tuple(sorted(settings.items()))

    if media_type == "Image":
        if not image_files:
            st.info("No sample images found in `data/samples/`.")
            return
        image_path = media_path
        if st.button("Run detection"):
            pipeline = load_pipeline(str(spots_path), method, settings_key)
            frame = cv2.imread(str(image_path))
            result = pipeline.process_frame(frame)
            st.image(
                cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB),
                caption=f"Occupied: {result.occupied_count}/{result.total_count}",
                width="stretch",
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Total spots", result.total_count)
            col2.metric("Occupied", result.occupied_count)
            col3.metric("Free", result.free_count)

    else:
        if not video_files:
            st.info("No sample videos found in `data/samples/`.")
            return
        video_path = media_path
        if st.button("Run on video"):
            pipeline = load_pipeline(str(spots_path), method, settings_key)
            cap = cv2.VideoCapture(str(video_path))
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()
            try:
                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    result = pipeline.process_frame(frame)
                    frame_placeholder.image(
                        cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB),
                        width="stretch",
                    )
                    metrics_placeholder.write(
                        f"**Occupied:** {result.occupied_count}/{result.total_count}  "
                        f"**Free:** {result.free_count}/{result.total_count}"
                    )
                    time.sleep(0.03)
            finally:
                cap.release()


if __name__ == "__main__":
    main()
