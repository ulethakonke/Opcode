# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import zipfile

# ---------- Image Encoding Functions ----------
def encode_image(img: Image.Image):
    img = img.convert('RGB')
    w, h = img.size
    pixels = np.array(img)
    
    # Get unique colors
    flat_pixels = pixels.reshape(-1, 3)
    unique_colors, inverse_indices = np.unique(flat_pixels, axis=0, return_inverse=True)
    
    if len(unique_colors) > 65535:
        st.warning(f"Image has too many unique colors ({len(unique_colors)}). Consider reducing colors.")
    
    encoded = bytearray()
    
    # HEADER
    encoded.append(0x00)
    encoded += w.to_bytes(2, 'little') + h.to_bytes(2, 'little')
    
    # PALETTE
    encoded.append(0x01)
    encoded += len(unique_colors).to_bytes(2, 'little')  # 2 bytes for palette length
    for color in unique_colors:
        encoded += bytes(color)
    
    # RAW pixels (using palette indices)
    encoded.append(0x07)  # BLOCK opcode
    encoded += w.to_bytes(2, 'little') + h.to_bytes(2, 'little')
    for idx in inverse_indices:
        encoded.append(idx if idx < 256 else 255)  # Clip if >255 for BLOCK
    
    # END
    encoded.append(0xFF)
    encoded += (0x12345678).to_bytes(4, 'little')
    
    return encoded, unique_colors, (w, h)

# ---------- Streamlit UI ----------
st.title("Concrete Symbolic Image Codec")

uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_column_width=True)
    
    if st.button("Encode Image"):
        try:
            encoded, palette, size = encode_image(img)
            st.success(f"Image encoded successfully! Encoded size: {len(encoded)} bytes.")
            
            # Prepare ZIP download
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                zf.writestr("encoded_image.bin", encoded)
            zip_buffer.seek(0)
            
            st.download_button(
                label="Download Encoded ZIP",
                data=zip_buffer,
                file_name="encoded_image.zip",
                mime="application/zip"
            )
        except Exception as e:
            st.error(f"Encoding failed: {e}")
