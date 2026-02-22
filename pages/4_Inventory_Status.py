import pandas as pd
import streamlit as st

from db import init_db, load_inventory

st.set_page_config(page_title="Inventory Status", layout="wide")
st.title("📊 Inventory Status / Pivot Summary")

init_db()

# Restore data from session or DB
if "edited_df" in st.session_state:
    df = st.session_state.edited_df
else:
    _, db_edited = load_inventory()
    if db_edited is None:
        st.warning("No inventory found. Upload in Inventory Editor first.")
        st.page_link("pages/1_Inventory_Editor.py", label="Go to Inventory Editor", icon="📋")
        st.stop()
    df = db_edited
    st.session_state.edited_df = db_edited
    st.session_state.inventory_ready = True

PRIORITY_COLS = [
    "Qr_Code",
    "Type",
    "Asset_Type",
    "Brand_Name",
    "Location",
    "Allocate_To",
    "Allocated_Department",
    "Allocated_Division",
    "Mc_Stutas",
    "Conditions",
]

present_cols = [c for c in PRIORITY_COLS if c in df.columns]

st.caption(f"Rows: {len(df):,}")

# Quick filters on priority columns
st.sidebar.header("Quick Filters (Priority Columns)")
filtered = df.copy()

for col in present_cols:
    vals = filtered[col].fillna("(blank)").astype(str)
    options = sorted(vals.unique().tolist())
    selected = st.sidebar.multiselect(col, options, default=[])
    if selected:
        filtered = filtered[vals.isin(selected)]

st.subheader("Pivot / Crosstab")

# Default example: Mc_Stutas by Location
default_index = "Location" if "Location" in present_cols else (present_cols[0] if present_cols else None)
default_cols = "Mc_Stutas" if "Mc_Stutas" in present_cols else (present_cols[1] if len(present_cols) > 1 else None)

c1, c2, c3 = st.columns([1, 1, 1])
with c1:
    index_col = st.selectbox("Rows (Index)", options=present_cols, index=present_cols.index(default_index) if default_index in present_cols else 0)
with c2:
    col_col = st.selectbox("Columns", options=present_cols, index=present_cols.index(default_cols) if default_cols in present_cols else min(1, len(present_cols)-1))
with c3:
    show_totals = st.checkbox("Show totals", value=True)

if index_col == col_col:
    st.warning("Choose different fields for Rows and Columns.")
    st.stop()

# Make pivot count table
pivot = pd.pivot_table(
    filtered,
    index=index_col,
    columns=col_col,
    aggfunc="size",
    fill_value=0
)

# Add totals
if show_totals:
    pivot["Total"] = pivot.sum(axis=1)
    total_row = pivot.sum(axis=0).to_frame().T
    total_row.index = ["Total"]
    pivot = pd.concat([pivot, total_row], axis=0)

st.dataframe(pivot.reset_index(), use_container_width=True)

st.subheader("Common views")

views = []
if "Location" in present_cols and "Mc_Stutas" in present_cols:
    views.append(("Mc_Stutas by Location", "Location", "Mc_Stutas"))
if "Type" in present_cols and "Location" in present_cols:
    views.append(("Type by Location", "Type", "Location"))
if "Brand_Name" in present_cols and "Location" in present_cols:
    views.append(("Brand_Name by Location", "Brand_Name", "Location"))
if "Allocated_Department" in present_cols and "Mc_Stutas" in present_cols:
    views.append(("Mc_Stutas by Allocated_Department", "Allocated_Department", "Mc_Stutas"))

for title, r, c in views:
    with st.expander(title):
        p = pd.pivot_table(filtered, index=r, columns=c, aggfunc="size", fill_value=0)
        st.dataframe(p.reset_index(), use_container_width=True)
