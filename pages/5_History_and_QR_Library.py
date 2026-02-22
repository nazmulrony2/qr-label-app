import json
import streamlit as st
import pandas as pd
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from PIL import Image
import io
import math

from db import init_db, fetch_change_log, list_qr_label_sets, load_qr_label_set

st.set_page_config(page_title="History & QR Library", layout="wide")
st.title("🗂️ History (Remarks) + QR Label Library")

init_db()

# ------------------- Change Log -------------------
st.header("1) Inventory Change Log (with Remarks)")

limit = st.number_input("Show last N changes", min_value=50, max_value=5000, value=500, step=50)
log_df = fetch_change_log(int(limit))

if log_df.empty:
    st.info("No change history found yet. Commit changes with remarks from Inventory Editor.")
else:
    st.dataframe(log_df, use_container_width=True)

# ------------------- QR Library -------------------
st.header("2) Saved QR Label Sets")

sets_df = list_qr_label_sets()
if sets_df.empty:
    st.info("No saved QR label sets yet. Save from the QR Label page.")
    st.stop()

st.dataframe(sets_df, use_container_width=True)

set_id = st.selectbox("Select a Label Set ID to load", options=sets_df["set_id"].tolist())

meta, items = load_qr_label_set(int(set_id))
settings = json.loads(meta["settings_json"])

st.subheader(f"Loaded Set: {meta['set_name']} (ID {meta['set_id']})")
st.caption(f"Created at: {meta['created_at']}")
st.json(settings)

# Preview thumbnails
st.subheader("Items preview")
preview_cols = st.columns(4)
for i, it in enumerate(items[:12]):  # show first 12
    img = Image.open(io.BytesIO(it["image_png_bytes"])).convert("RGB")
    with preview_cols[i % 4]:
        st.image(img, caption=it["qr_text"], use_container_width=True)

# --- regenerate PDF from saved set ---
def make_pdf_from_items(items, settings):
    page_size = A4 if settings.get("page_size") == "A4" else letter
    page_margin_in = float(settings.get("page_margin_in", 0.40))
    label_gap_in = float(settings.get("label_gap_in", 0.25))
    qr_box_in = float(settings.get("qr_box_in", 1.25))
    text_box_in = float(settings.get("text_box_in", 0.24))
    img_pad_in = float(settings.get("img_pad_in", 0.05))
    font_size_pt = int(settings.get("font_size_pt", 12))
    border_pt = float(settings.get("border_pt", 1.5))
    max_chars = int(settings.get("max_chars", 22))
    show_inner = bool(settings.get("show_inner_qr_border", True))

    QR_BOX = qr_box_in * inch
    TEXT_H = text_box_in * inch
    PAD = img_pad_in * inch
    LABEL_W = QR_BOX
    LABEL_H = QR_BOX + TEXT_H

    margin = page_margin_in * inch
    gap = label_gap_in * inch

    page_w, page_h = page_size
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin

    step_x = LABEL_W + gap
    step_y = LABEL_H + gap
    cols = max(1, int((usable_w + gap) // step_x))
    rows = max(1, int((usable_h + gap) // step_y))
    per_page = cols * rows

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=page_size)
    c.setLineWidth(border_pt)

    def trunc(s):
        s = (s or "").strip()
        return s if len(s) <= max_chars else s[:max_chars - 1] + "…"

    idx = 0
    total = len(items)
    pages = max(1, math.ceil(total / per_page)) if total else 1

    for _ in range(pages):
        for r in range(rows):
            for cc in range(cols):
                if idx >= total:
                    break
                x = margin + cc * step_x
                top_y = page_h - margin - r * step_y
                y = top_y - LABEL_H

                # label border
                c.rect(x, y, LABEL_W, LABEL_H, stroke=1, fill=0)

                # qr border top
                if show_inner:
                    c.rect(x, y + TEXT_H, LABEL_W, QR_BOX, stroke=1, fill=0)

                # image padded
                it = items[idx]
                try:
                    img = Image.open(io.BytesIO(it["image_png_bytes"])).convert("RGBA")
                    img_reader = ImageReader(img)
                    inner_x = x + PAD
                    inner_y = y + TEXT_H + PAD
                    inner_w = LABEL_W - 2 * PAD
                    inner_h = QR_BOX - 2 * PAD
                    c.drawImage(img_reader, inner_x, inner_y, width=inner_w, height=inner_h,
                                preserveAspectRatio=True, anchor="c", mask="auto")
                except Exception:
                    pass

                # text
                c.setFont("Times-Bold", font_size_pt)
                baseline_y = y + (TEXT_H - (font_size_pt / 72.0) * inch) / 2
                c.drawCentredString(x + LABEL_W / 2, baseline_y, trunc(it["qr_text"]))

                idx += 1

            if idx >= total:
                break
        if idx < total:
            c.showPage()

    c.save()
    return buf.getvalue()

st.subheader("Regenerate PDF from saved set")
if st.button("Generate PDF from this saved set"):
    pdf_bytes = make_pdf_from_items(items, settings)
    st.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name=f"qr_labels_set_{meta['set_id']}.pdf",
        mime="application/pdf",
    )
