import streamlit as st
from PIL import Image
import numpy as np
import io

# --------------------------
# Symbolic Codec Definition
# --------------------------

HEADER = 0x01
PALETTE = 0x02
FILL = 0x03
END = 0xFF

def encode_image(image: Image.Image):
    """
    Encode an image into our toy symbolic format.
    Currently:
      - builds a palette of unique colors (up to 256)
      - FILLs the whole image with the most common color
    """
    data = bytearray()

    w, h = image.size
    data.append(HEADER)
    data.extend(w.to_bytes(2, "big"))
    data.extend(h.to_bytes(2, "big"))

    # Build palette (deduplicate colors)
    pixels = list(image.getdata())
    unique_colors = list(dict.fromkeys(pixels))  # preserves order
    if len(unique_colors) > 256:
        unique_colors = unique_colors[:256]  # toy limit

    data.append(PALETTE)
    data.append(len(unique_colors))
    for (r, g, b) in unique_colors:
        data.extend([r, g, b])

    # Pick most frequent color for FILL
    color_counts = {}
    for c in pixels:
        color_counts[c] = color_counts.get(c, 0) + 1
    dominant_color = max(color_counts, key=color_counts.get)
    color_index = unique_colors.index(dominant_color)

    # FILL whole frame
    data.append(FILL)
    data.extend([0, 0])      # x,y
    data.extend([w, h])      # width, height
    data.append(color_index) # palette index

    # END
    data.append(END)

    return bytes(data)


def decode_image(data: bytes):
    """
    Decode from toy symbolic format back into a PIL Image.
    """
    stream = io.BytesIO(data)

    b = stream.read(1)
    if not b or b[0] != HEADER:
        raise ValueError("Missing HEADER")

    w = int.from_bytes(stream.read(2), "big")
    h = int.from_bytes(stream.read(2), "big")

    img = np.zeros((h, w, 3), dtype=np.uint8)

    # PALETTE
    if stream.read(1)[0] != PALETTE:
        raise ValueError("Missing PALETTE")
    ncolors = stream.read(1)[0]
    palette = []
    for _ in range(ncolors):
        rgb = tuple(stream.read(3))
        palette.append(rgb)

    # FILL
    b = stream.read(1)
    if b[0] == FILL:
        x = stream.read(1)[0]
        y = stream.read(1)[0]
        fw = stream.read(1)[0]
        fh = stream.read(1)[0]
        ci = stream.read(1)[0]
        color = palette[ci]
        img[y:y+fh, x:x+fw] = color

    # END
    if stream.read(1)[0] != END:
        raise ValueError("Missing END")

    return Image.fromarray(img)


# --------------------------
# Streamlit UI
# --------------------------

st.title("🌀 Symbolic Image Codec Prototype")
st.write("Upload any PNG/JPG to test symbolic encoding/decoding.")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.subheader("Original Image")
    st.image(img, width=300)

    # Encode
    encoded = encode_image(img)

    # Decode
    decoded = decode_image(encoded)

    st.subheader("Decoded Image")
    st.image(decoded, width=300)

    st.subheader("Encoded Byte Stream (hex)")
    st.code(encoded.hex(" "), language="text")

    st.success(
        f"Compressed size: {len(encoded)} bytes (vs raw {img.size[0]*img.size[1]*3} bytes)"
    )
