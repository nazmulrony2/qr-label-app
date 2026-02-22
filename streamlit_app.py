# qr_label_pdf_standalone.py
import sys
from pathlib import Path
# Comment out or remove this line if running as standalone file
# sys.path.append(str(Path(__file__).resolve().parents[1]))

import io
import math
import json
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.utils import ImageReader

# ────────────────────────────────────────────────
#  If you DON'T have a real db.py module yet,
#  comment out these two lines and the save function calls
# ────────────────────────────────────────────────
# from db import init_db, save_qr_label_set
# init_db()           # ← comment out if no db

# Fake / dummy save function so the code doesn't crash
def save_qr_label_set(set_name, settings_json, items):
    st.warning("Database save is disabled in standalone mode.")
    return 999  # fake ID

# ────────────────────────────────────────────────

st.set_page_config(page_title="QR Label PDF", layout="wide")
st.title("🏷️ QR Label PDF Generator")

# st.write(...) welcome message remains the same
st.write(
    "Upload multiple **PNG** QR codes and set the QR text. "
    "Adjust layout settings, see a **JPG preview**, generate PDF, and **save the set** for later (if db enabled)."
)

# ────────────────────────────────────────────────
# The rest of your code can stay almost identical
# from here ↓
# ────────────────────────────────────────────────

files = st.file_uploader("Upload QR PNG images", type=["png"], accept_multiple_files=True)

def truncate_text(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    if n <= 1:
        return "…"
    return s[: n - 1] + "…"

def compute_layout(page_size, margin, gap, label_w, label_h):
    page_w, page_h = page_size
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin
    step_x = label_w + gap
    step_y = label_h + gap
    cols = max(1, int((usable_w + gap) // step_x))
    rows = max(1, int((usable_h + gap) // step_y))
    per_page = cols * rows
    return cols, rows, per_page

def draw_image_fit_cover_pil(base_img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    im = base_img.copy()
    im_w, im_h = im.size
    if im_w <= 0 or im_h <= 0:
        return im
    scale = max(target_w / im_w, target_h / im_h)
    new_w = int(round(im_w * scale))
    new_h = int(round(im_h * scale))
    im = im.resize((max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS)

    left = max(0, (im.width - target_w) // 2)
    top  = max(0, (im.height - target_h) // 2)
    im = im.crop((left, top, left + target_w, top + target_h))
    return im

# Your render_preview_first_page_jpg function (with small fix)
def render_preview_first_page_jpg(
    labels,
    page_size,
    margin_inch,
    gap_inch,
    label_w_pt,
    label_h_pt,
    qr_h_pt,
    img_pad_pt,
    font_size_pt,
    border_pt,
    show_inner_qr_border,
    max_chars,
    dpi,
    full_page: bool
):
    page_w_pt, page_h_pt = page_size
    margin_pt = margin_inch * inch
    gap_pt    = gap_inch   * inch

    cols, rows, per_page = compute_layout(page_size, margin_pt, gap_pt, label_w_pt, label_h_pt)

    def pt_to_px(pt):
        return int(round((pt / 72.0) * dpi))

    label_w_px = pt_to_px(label_w_pt)
    label_h_px = pt_to_px(label_h_pt)
    qr_h_px    = pt_to_px(qr_h_pt)
    pad_px     = pt_to_px(img_pad_pt)
    gap_px     = pt_to_px(gap_pt)
    margin_px  = pt_to_px(margin_pt)

    if full_page:
        canvas_w = pt_to_px(page_w_pt)
        canvas_h = pt_to_px(page_h_pt)
        sheet = Image.new("RGB", (max(1, canvas_w), max(1, canvas_h)), "white")
        origin_x = margin_px
        origin_y = margin_px
    else:
        pad_outer_px = int(round(0.15 * dpi))
        grid_w = cols * label_w_px + max(0, cols-1) * gap_px + 2 * pad_outer_px
        grid_h = rows * label_h_px + max(0, rows-1) * gap_px + 2 * pad_outer_px
        sheet = Image.new("RGB", (max(1, grid_w), max(1, grid_h)), "white")
        origin_x = pad_outer_px
        origin_y = pad_outer_px

    # Try to load a nice font — fallback to default
    try:
        font_px = max(8, int(round(font_size_pt * dpi / 72)))
        font = ImageFont.truetype("arialbd.ttf", font_px)          # or "timesbd.ttf", "DejaVuSans-Bold.ttf"
    except Exception:
        try:
            font = ImageFont.truetype("arial.ttf", font_px)
        except Exception:
            font = ImageFont.load_default()

    dr = ImageDraw.Draw(sheet)

    max_labels = min(len(labels), per_page if full_page else 9999)
    idx = 0

    for r in range(rows):
        for c in range(cols):
            if idx >= max_labels:
                break

            x = origin_x + c * (label_w_px + gap_px)
            y = origin_y + r * (label_h_px + gap_px)

            bw = max(1, int(round(border_pt * dpi / 72)))   # better scaling

            # Outer border
            dr.rectangle([x, y, x + label_w_px - 1, y + label_h_px - 1], outline="black", width=bw)

            qr_top    = y
            qr_bottom = y + qr_h_px - 1

            if show_inner_qr_border:
                dr.rectangle([x, qr_top, x + label_w_px - 1, qr_bottom], outline="black", width=bw)

            # Cut guide
            cut_y = y + qr_h_px
            dr.line([x, cut_y, x + label_w_px - 1, cut_y], fill="black", width=max(1, bw // 2))

            # Paste QR
            inner_x = x + pad_px
            inner_y = qr_top + pad_px
            inner_w = max(1, label_w_px - 2 * pad_px)
            inner_h = max(1, qr_h_px    - 2 * pad_px)

            try:
                qr = Image.open(io.BytesIO(labels[idx]["img_bytes"])).convert("RGB")
                qr_fit = draw_image_fit_cover_pil(qr, inner_w, inner_h)
                sheet.paste(qr_fit, (inner_x, inner_y))
            except Exception:
                dr.rectangle([inner_x, inner_y, inner_x + inner_w - 1, inner_y + inner_h - 1],
                             outline="red", width=3)

            # Text
            txt = truncate_text(labels[idx].get("qr_text", ""), int(max_chars))
            bbox = dr.textbbox((0, 0), txt, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x + (label_w_px - tw) // 2
            text_top = y + qr_h_px
            text_h_px = label_h_px - qr_h_px
            ty = text_top + max(0, (text_h_px - th) // 2)
            dr.text((tx, ty), txt, fill="black", font=font)

            idx += 1

        if idx >= max_labels:
            break

    out = io.BytesIO()
    sheet.save(out, format="JPEG", quality=88, optimize=True)
    return out.getvalue()

# The make_pdf(...) function remains almost unchanged
# (I only added LANCZOS resampling and minor safety)

def make_pdf(
    labels,
    page_size,
    margin_in,
    gap_in,
    qr_box_in,
    text_box_in,
    img_pad_in,
    font_size_pt,
    border_pt,
    show_inner_qr_border,
    max_chars
):
    margin   = float(margin_in) * inch
    gap      = float(gap_in)    * inch
    qr_box   = float(qr_box_in) * inch
    text_h   = float(text_box_in) * inch
    img_pad  = float(img_pad_in) * inch

    inner_w = qr_box - 2 * img_pad
    inner_h = qr_box - 2 * img_pad
    if inner_w <= 0 or inner_h <= 0:
        st.error("Image padding too large for selected QR box size.")
        return b""

    label_w = qr_box
    label_h = qr_box + text_h

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=page_size)
    page_w, page_h = page_size

    cols, rows, per_page = compute_layout(page_size, margin, gap, label_w, label_h)

    def draw_one(x, y, qr_text, img_bytes):
        c.setLineWidth(float(border_pt))

        c.rect(x, y, label_w, label_h, stroke=1, fill=0)

        if show_inner_qr_border:
            c.rect(x, y + text_h, label_w, qr_box, stroke=1, fill=0)

        c.setLineWidth(float(border_pt) * 0.6)
        c.line(x, y + text_h, x + label_w, y + text_h)
        c.setLineWidth(float(border_pt))

        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
            img_reader = ImageReader(img)
            inner_x = x + img_pad
            inner_y = y + text_h + img_pad
            c.drawImage(
                img_reader,
                inner_x, inner_y,
                width=inner_w, height=inner_h,
                preserveAspectRatio=True,
                anchor="c",
                mask="auto",
            )
        except Exception:
            pass

        c.setFont("Times-Bold", int(font_size_pt))
        text = truncate_text(qr_text, int(max_chars))
        baseline_y = y + text_h / 2 - (font_size_pt / 72.0 * inch) / 2 + 2   # small baseline tweak
        c.drawCentredString(x + label_w / 2, baseline_y, text)

    total = len(labels)
    pages = math.ceil(total / per_page) if total > 0 else 1

    idx = 0
    for p in range(pages):
        for r in range(rows):
            for col in range(cols):
                if idx >= total:
                    break
                x = margin + col * (label_w + gap)
                top_y = page_h - margin - r * (label_h + gap)
                y = top_y - label_h
                draw_one(x, y, labels[idx]["qr_text"], labels[idx]["img_bytes"])
                idx += 1
            if idx >= total:
                break
        if idx < total:
            c.showPage()

    c.save()
    return buf.getvalue()

# ────────────────────────────────────────────────
# Main logic (almost same as yours)
# ────────────────────────────────────────────────

if not files:
    st.info("Upload one or more PNG QR images to continue.")
    st.stop()

initial = [{"file_name": f.name, "qr_text": f"QR:{f.name.rsplit('.',1)[0]}"} for f in files]
df = pd.DataFrame(initial)

st.subheader("QR text for each image (editable)")
edited = st.data_editor(df, use_container_width=True, num_rows="fixed")

# Layout settings (same)
st.subheader("Layout Settings")

c1, c2, c3, c4 = st.columns(4)
page_size_name   = c1.selectbox("Page size", ["A4", "Letter"], index=0)
page_margin_in   = c2.number_input("Page margin (inch)", 0.0, 2.0, 0.40, 0.05)
label_gap_in     = c3.number_input("Gap between labels (inch)", 0.0, 1.5, 0.25, 0.05)
preview_dpi      = c4.number_input("Preview DPI", 100, 400, 200, 25)

c5, c6, c7, c8 = st.columns(4)
qr_box_in   = c5.number_input("QR box size (inch)", 0.5, 3.0, 1.25, 0.05)
text_box_in = c6.number_input("Text box height (inch)", 0.10, 1.0, 0.24, 0.01)
img_pad_in  = c7.number_input("Image padding inside QR box (inch)", 0.0, 0.5, 0.05, 0.01)
font_size_pt= c8.number_input("Font size (pt)", 6, 40, 12, 1)

c9, c10, c11, c12 = st.columns(4)
border_pt          = c9.number_input("Border thickness (pt)", 0.5, 6.0, 1.5, 0.5)
show_inner_qr_border = c10.checkbox("Show inner QR box border", value=True)
max_chars          = c11.number_input("Max text characters (truncate)", 8, 60, 22, 1)
use_full_page_preview = c12.checkbox("Preview full page (slower)", value=False)

# Prepare labels list
file_map = {f.name: f for f in files}
labels = []
for _, row in edited.iterrows():
    fname = row["file_name"]
    if fname in file_map:
        labels.append({
            "qr_text": str(row["qr_text"] or ""),
            "img_bytes": file_map[fname].getvalue()
        })

page_size = A4 if page_size_name == "A4" else letter

# PDF generation
st.subheader("Generate PDF")

if "last_pdf_bytes" not in st.session_state:
    st.session_state.last_pdf_bytes = None

if st.button("Generate PDF"):
    try:
        with st.spinner("Generating PDF..."):
            pdf_bytes = make_pdf(
                labels=labels,
                page_size=page_size,
                margin_in=page_margin_in,
                gap_in=label_gap_in,
                qr_box_in=qr_box_in,
                text_box_in=text_box_in,
                img_pad_in=img_pad_in,
                font_size_pt=font_size_pt,
                border_pt=border_pt,
                show_inner_qr_border=show_inner_qr_border,
                max_chars=max_chars,
            )
            if pdf_bytes:
                st.session_state.last_pdf_bytes = pdf_bytes
                st.success("PDF ready.")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

if st.session_state.last_pdf_bytes:
    st.download_button(
        "↓ Download QR Labels PDF",
        data=st.session_state.last_pdf_bytes,
        file_name="qr_labels.pdf",
        mime="application/pdf",
    )

# Save (dummy version if no db)
st.subheader("Save this QR Label Set")
set_name = st.text_input("Set name", "My QR Labels")
if st.button("Save Label Set"):
    try:
        items = []
        for f, lab in zip(files, labels):
            items.append({
                "file_name": f.name,
                "qr_text": lab["qr_text"],
                "image_png_bytes": f.getvalue(),
            })
        set_id = save_qr_label_set(
            set_name=set_name.strip() or "My QR Labels",
            settings_json=json.dumps({}),  # minimal
            items=items
        )
        st.success(f"Saved (ID = {set_id})")
    except Exception as e:
        st.error(f"Save failed: {e}")

# Preview
st.subheader("JPG Preview — First page")

label_w_pt = qr_box_in * inch
label_h_pt = (qr_box_in + text_box_in) * inch
qr_h_pt    = qr_box_in * inch
img_pad_pt = img_pad_in * inch

with st.spinner("Rendering preview..."):
    preview_bytes = render_preview_first_page_jpg(
        labels=labels,
        page_size=page_size,
        margin_inch=page_margin_in,
        gap_inch=label_gap_in,
        label_w_pt=label_w_pt,
        label_h_pt=label_h_pt,
        qr_h_pt=qr_h_pt,
        img_pad_pt=img_pad_pt,
        font_size_pt=font_size_pt,
        border_pt=border_pt,
        show_inner_qr_border=show_inner_qr_border,
        max_chars=max_chars,
        dpi=preview_dpi,
        full_page=use_full_page_preview,
    )

st.image(preview_bytes, use_container_width=True)