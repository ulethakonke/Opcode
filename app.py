# app.py
import streamlit as st
from PIL import Image
import numpy as np
import io
import zipfile

# ---------- Encode ----------
def encode_image(img: Image.Image):
    img = img.convert('RGB')
    w, h = img.size
    pixels = np.array(img)
    
    # Flatten + unique colors
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
    encoded += len(unique_colors).to_bytes(2, 'little')
    for color in unique_colors:
        encoded += bytes(color)
    
    # RAW pixels (indices)
    encoded.append(0x07)
    encoded += w.to_bytes(2, 'little') + h.to_bytes(2, 'little')
    for idx in inverse_indices:
        encoded.append(idx if idx < 256 else 255)
    
    # END
    encoded.append(0xFF)
    encoded += (0x12345678).to_bytes(4, 'little')
    
    return encoded, unique_colors, (w, h)

# ---------- Decode ----------
def decode_image(encoded: bytes):
    pc = 0
    canvas = None
    palette = []
    
    while pc < len(encoded):
        opcode = encoded[pc]; pc += 1
        
        if opcode == 0x00:  # HEADER
            w = int.from_bytes(encoded[pc:pc+2], 'little'); pc += 2
            h = int.from_bytes(encoded[pc:pc+2], 'little'); pc += 2
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        elif opcode == 0x01:  # PALETTE
            n = int.from_bytes(encoded[pc:pc+2], 'little'); pc += 2
            palette = []
            for _ in range(n):
                r, g, b = encoded[pc], encoded[pc+1], encoded[pc+2]
                pc += 3
                palette.append((r, g, b))
        
        elif opcode == 0x07:  # BLOCK
            w_blk = int.from_bytes(encoded[pc:pc+2], 'little'); pc += 2
            h_blk = int.from_bytes(encoded[pc:pc+2], 'little'); pc += 2
            pixels = encoded[pc:pc + w_blk*h_blk]; pc += w_blk*h_blk
            pixels = np.array(pixels).reshape((h_blk, w_blk))
            for y in range(h_blk):
                for x in range(w_blk):
                    idx = pixels[y, x]
                    canvas[y, x] = palette[idx] if idx < len(palette) else (0,0,0)
        
        elif opcode == 0xFF:  # END
            break
    
    return Image.fromarray(canvas)

# ---------- Streamlit UI ----------
st.title("Concrete Symbolic Image Codec")

uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_column_width=True)
    
    if st.button("Encode & Decode Image"):
        try:
            # Encode
            encoded, palette, size = encode_image(img)
            st.success(f"Image encoded! Encoded size: {len(encoded)} bytes (original {uploaded_file.size} bytes).")
            
            # Decode back
            decoded_img = decode_image(encoded)
            st.image(decoded_img, caption="Decoded Image (from .bin)", use_column_width=True)
            
            # Prepare ZIP with .bin + decoded PNG
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                zf.writestr("encoded_image.bin", encoded)
                png_buffer = io.BytesIO()
                decoded_img.save(png_buffer, format="PNG")
                zf.writestr("decoded_image.png", png_buffer.getvalue())
            zip_buffer.seek(0)
            
            st.download_button(
                label="Download ZIP (encoded .bin + decoded .png)",
                data=zip_buffer,
                file_name="encoded_package.zip",
                mime="application/zip"
            )
        except Exception as e:
            st.error(f"Encoding failed: {e}")
