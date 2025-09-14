import streamlit as st
from PIL import Image
import numpy as np
import io

st.title("Concrete Symbolic Image Encoder")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    
    # Optional: Downsample colors to 128 for large images (avoid huge palettes)
    img = img.convert("P", palette=Image.ADAPTIVE, colors=128).convert("RGB")
    
    st.image(img, caption="Original Image", use_column_width=True)

    def encode_image(img: Image.Image):
        w, h = img.size
        pixels = np.array(img)
        data = []

        # HEADER
        data.append(0x00)  # opcode HEADER
        data.extend(w.to_bytes(2, "little"))
        data.extend(h.to_bytes(2, "little"))

        # Build palette
        flat_pixels = pixels.reshape(-1, 3)
        unique_colors = np.unique(flat_pixels, axis=0).tolist()
        
        # Clamp palette to 255
        palette_size = min(len(unique_colors), 255)
        unique_colors = unique_colors[:palette_size]

        data.append(palette_size)  # PALETTE size
        for color in unique_colors:
            data.extend(color)  # r,g,b

        # Simple FILL example: fill first color everywhere
        first_color_index = 0
        data.append(0x02)  # FILL opcode
        data.extend((0).to_bytes(2, "little"))  # x
        data.extend((0).to_bytes(2, "little"))  # y
        data.extend(w.to_bytes(2, "little"))    # width
        data.extend(h.to_bytes(2, "little"))    # height
        data.append(first_color_index)

        # END
        data.append(0xFF)
        data.extend((0x12345678).to_bytes(4, "little"))

        return bytes(data)

    try:
        encoded = encode_image(img)
        st.success(f"Image encoded successfully! Encoded size: {len(encoded)} bytes")
        st.download_button(
            label="Download Encoded File",
            data=encoded,
            file_name="encoded_image.bin",
            mime="application/octet-stream"
        )
    except Exception as e:
        st.error(f"Encoding failed: {e}")
