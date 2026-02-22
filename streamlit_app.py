# streamlit_app.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # allow: from db import ...

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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from db import init_db, save_qr_label_set  # persistent storage (can be stubbed)

st.set_page_config(page_title="QR Label PDF", layout="wide")
st.title("🏷️ QR Label PDF Generator")

init_db()

st.write(
    "Upload multiple **PNG** QR codes and set the QR text. "
    "Adjust layout settings, see a **JPG preview**, generate PDF, and **save the set** for later."
)

files = st.file_uploader("Upload QR PNG images", type=["png"], accept_multiple_files=True)

# ---------------- Helpers ----------------
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
    """Resize to cover target box (no empty space), then center-crop."""
    im = base_img.copy()
    im_w, im_h = im.size
    if im_w <= 0 or im_h <= 0:
        return im
    scale = max(target_w / im_w, target_h / im_h)
    new_w = int(round(im_w * scale))
    new_h = int(round(im_h * scale))
    im = im.resize((max(1, new_w), max(1, new_h)))

    left = max(0, (im.width - target_w) // 2)
    top = max(0, (im.height - target_h) // 2)
    im = im.crop((left, top, left + target_w, top + target_h))
    return im

def register_pdf_font() -> str:
    """
    Times New Roman Bold (TTF) register করে PDF-এ ব্যবহার করার জন্য।
    Streamlit Cloud এ OS font থাকে না, তাই repo-তে fonts/timesbd.ttf রাখুন।
    """
    base = Path(__file__).resolve().parent
    candidates = [
        base / "fonts" / "timesbd.ttf",                 # common name
        base / "fonts" / "Times New Roman Bold.ttf",    # alternate name
        base / "fonts" / "LiberationSerif-Bold.ttf",    # open fallback (Times-like)
    ]

    for fp in candidates:
        if fp.exists():
            font_name = "TNR_BOLD_TTF"
            try:
                pdfmetrics.registerFont(TTFont(font_name, str(fp)))
                return font_name
            except Exception:
                continue

    return "Times-Bold"  # fallback (built-in)

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
    """JPG preview of first page."""
    page_w_pt, page_h_pt = page_size
    margin_pt = margin_inch * inch
    gap_pt = gap_inch * inch

    cols, rows, per_page = compute_layout(page_size, margin_pt, gap_pt, label_w_pt, label_h_pt)

    def pt_to_px(pt):
        return int(round((pt / 72.0) * dpi))

    label_w_px = pt_to_px(label_w_pt)
    label_h_px = pt_to_px(label_h_pt)
    qr_h_px = pt_to_px(qr_h_pt)
    pad_px = pt_to_px(img_pad_pt)
    gap_px = pt_to_px(gap_pt)
    margin_px = pt_to_px(margin_pt)

    if full_page:
        canvas_w = pt_to_px(page_w_pt)
        canvas_h = pt_to_px(page_h_pt)
        sheet = Image.new("RGB", (max(1, canvas_w), max(1, canvas_h)), "white")
        dr = ImageDraw.Draw(sheet)
        dr.rectangle([0, 0, sheet.width - 1, sheet.height - 1], outline="black", width=1)
        origin_x = margin_px
        origin_y = margin_px
    else:
        pad_outer_px = int(round(0.15 * dpi))
        grid_w = cols * label_w_px + (cols - 1) * gap_px + 2 * pad_outer_px
        grid_h = rows * label_h_px + (rows - 1) * gap_px + 2 * pad_outer_px
        sheet = Image.new("RGB", (max(1, grid_w), max(1, grid_h)), "white")
        origin_x = pad_outer_px
        origin_y = pad_outer_px

    # Preview font (best-effort). PDF uses Times New Roman Bold TTF if present.
    try:
        font_px = max(10, int(round((font_size_pt / 72.0) * dpi)))
        font = ImageFont.truetype("timesbd.ttf", font_px)
    except Exception:
        font = ImageFont.load_default()

    max_labels = min(len(labels), per_page)
    idx = 0

    for r in range(rows):
        for c in range(cols):
            if idx >= max_labels:
                break

            x = origin_x + c * (label_w_px + gap_px)
            y = origin_y + r * (label_h_px + gap_px)

            dr = ImageDraw.Draw(sheet)
            bw = max(1, int(round(border_pt)))  # px approx

            # Outer label border
            dr.rectangle([x, y, x + label_w_px - 1, y + label_h_px - 1], outline="black", width=bw)

            # QR area is TOP portion with height qr_h_px
            qr_top = y
            qr_bottom = y + qr_h_px - 1

            if show_inner_qr_border:
                dr.rectangle([x, qr_top, x + label_w_px - 1, qr_bottom], outline="black", width=bw)

            # --- CUT GUIDE LINE (below QR box) ---
            cut_y = y + qr_h_px
            dr.line(
                [x, cut_y, x + label_w_px - 1, cut_y],
                fill="black",
                width=max(1, bw // 2)
            )

            # Image inside QR box with padding
            inner_x = x + pad_px
            inner_y = qr_top + pad_px
            inner_w = max(1, label_w_px - 2 * pad_px)
            inner_h = max(1, qr_h_px - 2 * pad_px)

            try:
                qr = Image.open(io.BytesIO(labels[idx]["img_bytes"])).convert("RGB")
                qr_fit = draw_image_fit_cover_pil(qr, inner_w, inner_h)
                sheet.paste(qr_fit, (inner_x, inner_y))
            except Exception:
                pass

            # Text area below QR
            txt = truncate_text(labels[idx].get("qr_text", ""), int(max_chars))
            bbox = dr.textbbox((0, 0), txt, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x + (label_w_px - tw) // 2
            text_top = y + qr_h_px
            text_h_px = max(1, label_h_px - qr_h_px)
            ty = text_top + max(0, (text_h_px - th) // 2)
            dr.text((tx, ty), txt, fill="black", font=font)

            idx += 1

        if idx >= max_labels:
            break

    out = io.BytesIO()
    sheet.save(out, format="JPEG", quality=92)
    return out.getvalue()

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
    margin = float(margin_in) * inch
    gap = float(gap_in) * inch

    qr_box = float(qr_box_in) * inch
    text_h = float(text_box_in) * inch
    img_pad = float(img_pad_in) * inch

    # Safety: prevent negative inner area
    inner_w = qr_box - 2 * img_pad
    inner_h = qr_box - 2 * img_pad
    if inner_w <= 0 or inner_h <= 0:
        raise ValueError("Image padding is too large for the QR box size. Reduce padding or increase QR box.")

    label_w = qr_box
    label_h = qr_box + text_h

    FONT_NAME_PDF = register_pdf_font()  # ✅ Times New Roman Bold TTF (if provided), else fallback

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=page_size)
    page_w, page_h = page_size

    cols, rows, per_page = compute_layout(page_size, margin, gap, label_w, label_h)

    def draw_one(x, y, qr_text, img_bytes):
        c.setLineWidth(float(border_pt))

        # Outer label border
        c.rect(x, y, label_w, label_h, stroke=1, fill=0)

        # QR box border at TOP (same width as label)
        if show_inner_qr_border:
            c.rect(x, y + text_h, label_w, qr_box, stroke=1, fill=0)

        # --- CUT GUIDE LINE (below QR box) ---
        c.setLineWidth(float(border_pt) * 0.6)  # slightly thinner
        c.line(x, y + text_h, x + label_w, y + text_h)

        # Restore border width
        c.setLineWidth(float(border_pt))

        # QR image with padding (inside the top box)
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
            img_reader = ImageReader(img)
            inner_x = x + img_pad
            inner_y = y + text_h + img_pad
            c.drawImage(
                img_reader,
                inner_x,
                inner_y,
                width=inner_w,
                height=inner_h,
                preserveAspectRatio=True,
                anchor="c",
                mask="auto",
            )
        except Exception:
            pass

        # ---------------- TEXT (FIXED) ----------------
        text = truncate_text(qr_text, int(max_chars))

        # Font metrics (ascent/descent in 1000-em units)
        ascent_1000 = pdfmetrics.getAscent(FONT_NAME_PDF)
        descent_1000 = pdfmetrics.getDescent(FONT_NAME_PDF)  # usually negative
        units_1000 = ascent_1000 - descent_1000
        if units_1000 <= 0:
            units_1000 = 1000

        # ✅ Fit-to-box height (keeps your chosen font_size_pt as minimum)
        box_h_pt = float(text_h)  # points
        target_text_h = box_h_pt * 0.70
        fit_size = (target_text_h * 1000.0) / units_1000
        effective_size = max(float(font_size_pt), float(fit_size))

        # ✅ Width-fit (only if too wide)
        max_w = float(label_w) - (0.10 * inch)
        while pdfmetrics.stringWidth(text, FONT_NAME_PDF, effective_size) > max_w and effective_size > 6:
            effective_size -= 0.5

        c.setFont(FONT_NAME_PDF, effective_size)

        # ✅ Accurate vertical centering using metrics
        ascent = (ascent_1000 * effective_size) / 1000.0
        descent = (descent_1000 * effective_size) / 1000.0  # negative
        text_height = ascent - descent
        baseline_y = y + (box_h_pt - text_height) / 2.0 - descent

        c.drawCentredString(x + label_w / 2, baseline_y, text)
        # ------------------------------------------------

    total = len(labels)
    pages = max(1, math.ceil(total / per_page)) if total else 1

    idx = 0
    for _ in range(pages):
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

# ---------------- Main ----------------
if not files:
    st.info("Upload one or more PNG QR images to continue.")
    st.stop()

# --- QR text editor (TOP) ---
initial = []
for f in files:
    base = f.name.rsplit(".", 1)[0]
    initial.append({"file_name": f.name, "qr_text": f"QR:{base}"})

df = pd.DataFrame(initial)

st.subheader("QR text for each image (editable)")
edited = st.data_editor(df, use_container_width=True, num_rows="fixed")

# -------- Layout Settings (BELOW editor) --------
st.subheader("Layout Settings")

# Default values YOU provided
c1, c2, c3, c4 = st.columns(4)
with c1:
    page_size_name = st.selectbox("Page size", ["A4", "Letter"], index=0, key="ps")
with c2:
    page_margin_in = st.number_input("Page margin (inch)", min_value=0.0, max_value=2.0, value=0.40, step=0.05, key="pm")
with c3:
    label_gap_in = st.number_input("Gap between labels (inch)", min_value=0.0, max_value=1.5, value=0.25, step=0.05, key="lg")
with c4:
    preview_dpi = st.number_input("Preview DPI", min_value=100, max_value=400, value=200, step=25, key="dpi")

c5, c6, c7, c8 = st.columns(4)
with c5:
    qr_box_in = st.number_input("QR box size (inch)", min_value=0.5, max_value=3.0, value=1.25, step=0.05, key="qb")
with c6:
    text_box_in = st.number_input("Text box height (inch)", min_value=0.10, max_value=1.0, value=0.24, step=0.01, key="tb")
with c7:
    img_pad_in = st.number_input("Image padding inside QR box (inch)", min_value=0.0, max_value=0.5, value=0.05, step=0.01, key="ip")
with c8:
    font_size_pt = st.number_input("Font size (pt)", min_value=6, max_value=40, value=12, step=1, key="fs")

c9, c10, c11, c12 = st.columns(4)
with c9:
    border_pt = st.number_input("Border thickness (pt)", min_value=0.5, max_value=6.0, value=1.50, step=0.5, key="bt")
with c10:
    show_inner_qr_border = st.checkbox("Show inner QR box border", value=True, key="iqb")
with c11:
    max_chars = st.number_input("Max text characters (truncate)", min_value=8, max_value=60, value=22, step=1, key="mc")
with c12:
    use_full_page_preview = st.checkbox("Preview full page (slower)", value=False, key="fp")

# Build labels from edited table (keep file order)
file_map = {f.name: f for f in files}
labels = []
for _, row in edited.iterrows():
    f = file_map.get(row["file_name"])
    if not f:
        continue
    labels.append({"qr_text": str(row["qr_text"]) if row["qr_text"] is not None else "", "img_bytes": f.getvalue()})

page_size = A4 if page_size_name == "A4" else letter

# ---------------- Generate PDF (and KEEP bytes after rerun) ----------------
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
        st.session_state.last_pdf_bytes = pdf_bytes
        st.success("PDF generated. Click Download below.")
    except Exception as e:
        st.session_state.last_pdf_bytes = None
        st.error(f"PDF generation failed: {e}")

# Always show download if available
if st.session_state.last_pdf_bytes:
    st.download_button(
        "Download QR Labels PDF",
        data=st.session_state.last_pdf_bytes,
        file_name="qr_labels.pdf",
        mime="application/pdf",
    )

# ---------------- Save Label Set (PERSISTENT) ----------------
st.subheader("Save this QR Label Set (store for later use)")

set_col1, set_col2 = st.columns([2, 1])
with set_col1:
    set_name = st.text_input("Set name", value="My QR Labels")
with set_col2:
    save_set = st.button("Save Label Set")

settings = {
    "page_size": page_size_name,
    "page_margin_in": float(page_margin_in),
    "label_gap_in": float(label_gap_in),
    "preview_dpi": int(preview_dpi),
    "qr_box_in": float(qr_box_in),
    "text_box_in": float(text_box_in),
    "img_pad_in": float(img_pad_in),
    "font_size_pt": int(font_size_pt),
    "border_pt": float(border_pt),
    "max_chars": int(max_chars),
    "show_inner_qr_border": bool(show_inner_qr_border),
    "use_full_page_preview": bool(use_full_page_preview),
}

if save_set:
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
            settings_json=json.dumps(settings),
            items=items
        )
        st.success(f"Saved Label Set to database. Set ID = {set_id}")
    except Exception as e:
        st.error(f"Failed to save label set: {e}")

# ---------------- JPG Preview ----------------
st.subheader("JPG Preview (before generating PDF) — First page")

label_w_pt = float(qr_box_in) * inch
label_h_pt = (float(qr_box_in) + float(text_box_in)) * inch
qr_h_pt = float(qr_box_in) * inch
img_pad_pt = float(img_pad_in) * inch

with st.spinner("Rendering preview..."):
    preview_jpg = render_preview_first_page_jpg(
        labels=labels,
        page_size=page_size,
        margin_inch=float(page_margin_in),
        gap_inch=float(label_gap_in),
        label_w_pt=label_w_pt,
        label_h_pt=label_h_pt,
        qr_h_pt=qr_h_pt,
        img_pad_pt=img_pad_pt,
        font_size_pt=int(font_size_pt),
        border_pt=float(border_pt),
        show_inner_qr_border=bool(show_inner_qr_border),
        max_chars=int(max_chars),
        dpi=int(preview_dpi),
        full_page=bool(use_full_page_preview),
    )

st.image(preview_jpg, use_container_width=True)