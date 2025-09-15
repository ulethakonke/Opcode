import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io, zipfile, json

# ------------------------
# Helpers
# ------------------------
def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)

def extract_palette(img, k=8):
    """Cluster colors to create palette."""
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    return centers, labels.reshape(img.shape[:2])

def image_to_symbols(img, k=8):
    """Convert image into symbolic ops."""
    palette, labelmap = extract_palette(img, k)
    h, w = img.shape[:2]
    symbols = {
        "HEADER": {"width": w, "height": h},
        "PALETTE": [to_serializable(c) for c in palette],
        "OPS": []
    }

    # detect polygons for each color
    for color_idx, col in enumerate(palette):
        mask = (labelmap == color_idx).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 20:  # skip tiny noise
                continue
            pts = cnt.squeeze().tolist()
            if isinstance(pts[0], int):  # ensure list of points
                pts = [pts]
            symbols["OPS"].append({
                "type": "POLYGON",
                "points": to_serializable(pts),
                "color": to_serializable(col.tolist())
            })

    return symbols

def render_symbols(symbols):
    """Decode symbolic representation back into image."""
    w = symbols["HEADER"]["width"]
    h = symbols["HEADER"]["height"]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    for op in symbols["OPS"]:
        if op["type"] == "POLYGON":
            pts = np.array(op["points"], np.int32)
            pts = pts.reshape((-1, 1, 2))
            col = tuple(op["color"])
            cv2.fillPoly(canvas, [pts], col)

    return canvas

# ------------------------
# Streamlit UI
# ------------------------
st.title("🌀 Symbolic Image Codec v2")

uploaded = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    st.subheader("Original Image")
    st.image(img_np, use_column_width=True)

    # Encode
    symbolic = image_to_symbols(img_np, k=6)

    # Decode
    decoded = render_symbols(symbolic)

    st.subheader("Reconstructed Image")
    st.image(decoded, use_column_width=True)

    # Download zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("symbolic.json", json.dumps(symbolic, indent=2, default=to_serializable))
        _, enc_png = cv2.imencode(".png", decoded)
        zf.writestr("decoded.png", enc_png.tobytes())
    st.download_button("Download Encoded Package", buf.getvalue(), "symbolic_codec.zip")
