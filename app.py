# app.py
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import io
import json
import zipfile
from collections import deque, Counter
import math

st.set_page_config(layout="wide")
st.title("Symbolic Image Encoder — V2 (RECT / CIRCLE / GRADIENT / DETAIL)")

# ---------- Utilities ----------
def to_rgb_array(img: Image.Image):
    return np.array(img.convert("RGB"))

def quantize_image(img: Image.Image, n_colors=8):
    p = img.convert("P", palette=Image.ADAPTIVE, colors=n_colors)
    palette = p.getpalette()[:3 * n_colors]
    colors = [tuple(palette[i:i+3]) for i in range(0, len(palette), 3)]
    idx = np.array(p)
    return idx, colors  # idx: HxW indices into colors

def most_frequent_color_index(idx):
    flat = idx.flatten()
    cnt = Counter(flat)
    return cnt.most_common(1)[0][0]

# Simple 4-neighbor connected-component labeling
def connected_components(mask):
    h, w = mask.shape
    labels = -np.ones((h, w), dtype=int)
    label = 0
    components = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and labels[y, x] == -1:
                # BFS
                q = deque()
                q.append((y, x))
                labels[y, x] = label
                coords = []
                while q:
                    cy, cx = q.popleft()
                    coords.append((cx, cy))
                    for ny, nx in ((cy-1, cx),(cy+1, cx),(cy, cx-1),(cy, cx+1)):
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == -1:
                            labels[ny, nx] = label
                            q.append((ny, nx))
                components.append(coords)
                label += 1
    return components

def bbox_of_coords(coords):
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    return minx, miny, maxx - minx + 1, maxy - miny + 1

def compute_perimeter(mask_region):
    # mask_region is 0/1 numpy array
    h, w = mask_region.shape
    perim = 0
    for y in range(h):
        for x in range(w):
            if mask_region[y,x]:
                # if any neighbor is zero or out of bounds, add to perimeter
                for ny, nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
                    if not (0 <= ny < h and 0 <= nx < w and mask_region[ny, nx]):
                        perim += 1
    return perim

def crop_image(img, bbox):
    x, y, w, h = bbox
    return img.crop((x, y, x+w, y+h))

# Try to detect vertical/horizontal linear gradient
def detect_gradient(rgb_arr):
    h, w, _ = rgb_arr.shape
    # compute row means and column means
    row_means = rgb_arr.mean(axis=1).mean(axis=1)  # shape (h,)
    col_means = rgb_arr.mean(axis=0).mean(axis=1)  # shape (w,)
    # fit linear regression line and compute R^2
    def fit_r2(arr):
        n = len(arr)
        x = np.arange(n)
        A = np.vstack([x, np.ones(n)]).T
        m, c = np.linalg.lstsq(A, arr, rcond=None)[0]
        pred = m*x + c
        ss_res = ((arr - pred)**2).sum()
        ss_tot = ((arr - arr.mean())**2).sum()
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
        return r2, m
    r2_row, slope_row = fit_r2(row_means)
    r2_col, slope_col = fit_r2(col_means)
    # decide orientation
    if r2_row > 0.9 and abs(slope_row) > 0.02:
        return ("vertical", r2_row)
    if r2_col > 0.9 and abs(slope_col) > 0.02:
        return ("horizontal", r2_col)
    return (None, 0.0)

# ---------- Symbolic encoder ----------
def symbolic_encode(img: Image.Image, n_colors=8, min_component_area=20):
    w, h = img.size
    idx, palette = quantize_image(img, n_colors=n_colors)
    bg_index = most_frequent_color_index(idx)
    palette_list = palette  # list of (r,g,b)
    symbols = []
    used_mask = np.zeros_like(idx, dtype=bool)

    # 1) Background fill
    bg_color = palette_list[bg_index]
    symbols.append({"op":"FILL", "x":0, "y":0, "w":w, "h":h, "color": bg_color})
    # mark background as used
    used_mask[idx == bg_index] = True

    # 2) Gradient detection (if background not perfectly uniform)
    rgb = to_rgb_array(img)
    orientation, r2 = detect_gradient(rgb)
    if orientation and r2 > 0.95:
        # approximate gradient endpoints by row means
        if orientation == "vertical":
            top_color = tuple(rgb[0,:,:].mean(axis=0).astype(int))
            bottom_color = tuple(rgb[-1,:,:].mean(axis=0).astype(int))
            symbols.append({"op":"GRADIENT", "x":0, "y":0, "w":w, "h":h, "c1":top_color, "c2":bottom_color, "dir":"vertical"})
            # If gradient is used we won't rely on bg fill
            used_mask[:, :] = True

    # 3) Find remaining components (non-bg)
    mask_nonbg = ~used_mask
    components = connected_components(mask_nonbg)
    detail_blocks = []
    for comp in components:
        if len(comp) < min_component_area:
            continue
        bbox = bbox_of_coords(comp)
        x, y, cw, ch = bbox
        # make mask for component within bbox
        region_mask = np.zeros((ch, cw), dtype=np.uint8)
        for cx, cy in comp:
            region_mask[cy - y, cx - x] = 1
        area = region_mask.sum()
        perim = compute_perimeter(region_mask)
        circularity = (4 * math.pi * area / (perim*perim)) if perim > 0 else 0
        # If bounding rect filled ratio close -> RECT
        bbox_area = cw*ch
        fill_ratio = area / bbox_area
        # Decide circle vs rect vs detail
        if circularity > 0.5 and fill_ratio > 0.6:
            # approximate circle center and radius
            xs = [c[0] for c in comp]
            ys = [c[1] for c in comp]
            cx = int(sum(xs)/len(xs))
            cy = int(sum(ys)/len(ys))
            # radius as average distance to centroid
            r = int(np.mean([math.hypot(x-cx,y-cy) for x,y in comp]))
            symbols.append({"op":"CIRCLE", "x":cx, "y":cy, "r":r, "color": tuple(rgb[cy, cx])})
            for px, py in comp:
                used_mask[py, px] = True
        elif fill_ratio > 0.7:
            symbols.append({"op":"RECT", "x":x, "y":y, "w":cw, "h":ch, "color": tuple(rgb[y + ch//2, x + cw//2])})
            for px, py in comp:
                used_mask[py, px] = True
        else:
            # fallback: create detail block (crop + save PNG later)
            crop = crop_image(img, bbox)
            detail_blocks.append({"bbox":bbox, "image":crop})
            for px, py in comp:
                used_mask[py, px] = True

    # 4) Anything still unmarked -> detail in tiles
    remaining = np.where(~used_mask)
    if remaining[0].size > 0:
        # create a coarse tiling (e.g., 64x64 blocks) and include any blocks with nonzero remaining pixels
        tile = 64
        for ty in range(0, h, tile):
            for tx in range(0, w, tile):
                sub = ~used_mask[ty:ty+tile, tx:tx+tile]
                if sub.size == 0:
                    continue
                if np.any(sub):
                    # crop region
                    bx = tx
                    by = ty
                    bw = min(tile, w - tx)
                    bh = min(tile, h - ty)
                    crop = crop_image(img, (bx, by, bw, bh))
                    detail_blocks.append({"bbox": (bx, by, bw, bh), "image": crop})
                    used_mask[by:by+bh, bx:bx+bw] = True

    # 5) Compose final symbol list
    # Remove duplicate DETAIL blocks by bbox
    # Convert colors to tuples (already done)
    symbols_out = []
    for s in symbols:
        symbols_out.append(s)

    # Add DETAIL entries
    for i, d in enumerate(detail_blocks):
        symbols_out.append({"op":"DETAIL", "id": f"detail_{i}", "x": d["bbox"][0], "y": d["bbox"][1], "w": d["bbox"][2], "h": d["bbox"][3]})

    return {"width": w, "height": h, "palette": [tuple(c) for c in palette], "symbols": symbols_out}, detail_blocks

# ---------- Reconstructor ----------
def reconstruct_from_symbols(symbolic, detail_blocks):
    w = symbolic["width"]
    h = symbolic["height"]
    canvas = Image.new("RGB", (w, h), (0,0,0))
    draw = ImageDraw.Draw(canvas)
    for s in symbolic["symbols"]:
        op = s["op"]
        if op == "FILL":
            draw.rectangle([s["x"], s["y"], s["x"]+s["w"]-1, s["y"]+s["h"]-1], fill=tuple(s["color"]))
        elif op == "GRADIENT":
            # simple vertical linear gradient
            x, y, W, H = s["x"], s["y"], s["w"], s["h"]
            c1 = np.array(s["c1"], dtype=int)
            c2 = np.array(s["c2"], dtype=int)
            for row in range(H):
                t = row / max(1, H-1)
                col = tuple(((1-t)*c1 + t*c2).astype(int))
                draw.line([(x, y+row), (x+W-1, y+row)], fill=col)
        elif op == "RECT":
            draw.rectangle([s["x"], s["y"], s["x"]+s["w"]-1, s["y"]+s["h"]-1], fill=tuple(s["color"]))
        elif op == "CIRCLE":
            cx, cy, r = s["x"], s["y"], s["r"]
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=tuple(s["color"]))
        elif op == "DETAIL":
            # find block image
            pass

    # paste detail blocks
    for i, d in enumerate(detail_blocks):
        bbox = d["bbox"]
        crop = d["image"]
        canvas.paste(crop, (bbox[0], bbox[1]))
    return canvas

# ---------- Streamlit UI ----------
st.markdown("Upload an image and the encoder will attempt to describe it with symbolic primitives. Complex regions become DETAIL blocks (PNG).")
col1, col2 = st.columns([1,1])

with col1:
    uploaded = st.file_uploader("Upload image (png/jpg/jpeg)", type=["png","jpg","jpeg"])
    n_colors = st.slider("Quantize colors (palette size)", 4, 32, 8)
    min_comp_area = st.slider("Min component area for shapes (px)", 5, 50, 20)
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Original", use_column_width=True)

with col2:
    if uploaded and st.button("Encode symbolically"):
        with st.spinner("Encoding..."):
            symbolic, detail_blocks = symbolic_encode(img, n_colors=n_colors, min_component_area=min_comp_area)
            recon = reconstruct_from_symbols(symbolic, detail_blocks)
            st.image(recon, caption="Reconstructed preview", use_column_width=True)
            st.success(f"Symbols: {len(symbolic['symbols'])}, Detail blocks: {len(detail_blocks)}")
            # prepare zip
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("symbolic.json", json.dumps(symbolic, indent=2))
                # write detail block images
                for i, d in enumerate(detail_blocks):
                    pngb = io.BytesIO()
                    d["image"].save(pngb, format="PNG")
                    zf.writestr(f"detail_{i}.png", pngb.getvalue())
                # write reconstructed preview
                previewb = io.BytesIO()
                recon.save(previewb, format="PNG")
                zf.writestr("reconstructed_preview.png", previewb.getvalue())
            buf.seek(0)
            st.download_button("Download symbolic ZIP", data=buf, file_name="symbolic_package.zip", mime="application/zip")
            st.json(symbolic)
