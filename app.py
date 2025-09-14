import streamlit as st
from PIL import Image
import numpy as np
import io
import zipfile

# ---------- Helper functions ----------
def encode_image(img: Image.Image):
    """Encode image to a symbolic opcode format (simplified example)."""
    img = img.convert("RGB")
    pixels = np.array(img)
    h, w, _ = pixels.shape

    # Build a simple palette
    unique_colors = []
    color_map = {}
    data = []

    for y in range(h):
        for x in range(w):
            color = tuple(pixels[y, x])
            if color not in color_map:
                color_map[color] = len(unique_colors)
                unique_colors.append(color)
            data.append(color_map[color])

    # Symbolic header and palette
    encoded = bytearray()
    encoded.append(0x00)  # HEADER opcode
    encoded += w.to_bytes(2, 'little') + h.to_bytes(2, 'little')
    encoded.append(0x01)  # PALETTE opcode
    encoded.append(len(unique_colors))
    for c in unique_colors:
        encoded += bytes(c)
    # Pixel data
    encoded += bytearray(data)
    encoded.append(0xFF)  # END opcode
    return encoded, unique_colors, (w, h)

def decode_image(encoded_data, palette, size):
    """Decode symbolic image back to a PIL image."""
    w, h = size
    img_array = np.zeros((h, w, 3), dtype=np.uint8)
    # Simple decoder: just map indices to colors sequentially
    pixel_data = encoded_data[len(palette)*3 + 3 + 5:-1]  # skip header/palette
    img_array = np.array([palette[idx] for idx in pixel_data], dtype=np.uint8).reshape(h, w, 3)
    return Image.fromarray(img_array)

# ---------- Streamlit UI ----------
st.title("Concrete Symbolic Image Codec")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_column_width=True)

    # Encode
    encoded, palette, size = encode_image(img)
    st.success(f"Image encoded successfully! Encoded size: {len(encoded)} bytes")

    # Decode for preview
    decoded_img = decode_image(encoded, palette, size)
    st.image(decoded_img, caption="Decoded Preview", use_column_width=True)

    # Prepare downloadable ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        # Symbolic file
        zip_file.writestr("encoded_image.bin", encoded)
        # PNG preview
        png_buffer = io.BytesIO()
        decoded_img.save(png_buffer, format="PNG")
        zip_file.writestr("decoded_preview.png", png_buffer.getvalue())
    
    zip_buffer.seek(0)
    st.download_button(
        label="Download Encoded + Preview ZIP",
        data=zip_buffer,
        file_name="symbolic_image.zip",
        mime="application/zip"
    )
