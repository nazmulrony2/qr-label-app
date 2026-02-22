import io
from datetime import datetime
import pandas as pd
import streamlit as st

from db import (
    init_db,
    save_inventory,
    load_inventory,
    clear_inventory_and_changes,
    get_last_update_time,
    insert_change_rows,
)

st.set_page_config(page_title="Inventory Editor", layout="wide")
st.title("📋 Inventory Editor (Persistent + Change Remark Log)")

init_db()

PRIORITY_COLS = [
    "Qr_Code", "Type", "Asset_Type", "Brand_Name",
    "Location", "Allocate_To", "Allocated_Department", "Allocated_Division",
    "Mc_Stutas", "Conditions"
]

uploaded = st.file_uploader("Upload inventory Excel (.xlsx)", type=["xlsx"])

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="inventory")
    return out.getvalue()

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin(["", "nan", "None"]), col] = None
    return df

def diff_cells(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    # Align
    old = old_df.reindex_like(new_df)
    new = new_df
    changed = new.ne(old) & ~(new.isna() & old.isna())
    if not changed.any().any():
        return pd.DataFrame(columns=["row_index", "row_id_value", "column_name", "old_value", "new_value", "remark"])

    id_col = "ID" if "ID" in new.columns else None
    rows = []
    for ridx in new.index:
        if ridx not in changed.index:
            continue
        row_changed_cols = changed.loc[ridx]
        if not bool(row_changed_cols.any()):
            continue
        row_id_value = str(new.at[ridx, id_col]) if id_col and pd.notna(new.at[ridx, id_col]) else None
        for col in new.columns:
            if bool(row_changed_cols.get(col, False)):
                ov = old.at[ridx, col] if ridx in old.index else None
                nv = new.at[ridx, col]
                rows.append({
                    "row_index": int(ridx) if str(ridx).isdigit() else None,
                    "row_id_value": row_id_value,
                    "column_name": col,
                    "old_value": None if pd.isna(ov) else str(ov),
                    "new_value": None if pd.isna(nv) else str(nv),
                    "remark": ""
                })
    return pd.DataFrame(rows)

# ---- DB controls ----
db_original, db_edited = load_inventory()
last_updated = get_last_update_time()

top = st.columns([2, 2, 1])
with top[0]:
    st.caption(f"DB last updated: {last_updated or '—'}")
with top[1]:
    st.caption("Upload a new Excel to replace DB, or edit stored inventory.")
with top[2]:
    if st.button("Clear Inventory + Changes DB"):
        clear_inventory_and_changes()
        st.session_state.clear()
        st.rerun()

# ---- Load data ----
if uploaded is not None:
    df = normalize_df(pd.read_excel(uploaded))
    st.session_state.original_df = df.copy()
    st.session_state.edited_df = df.copy()
    st.session_state.inventory_ready = True

    save_inventory(st.session_state.original_df, st.session_state.edited_df)
    st.session_state.last_committed_df = st.session_state.edited_df.copy()

    st.success("Uploaded and saved to database.")
else:
    if "edited_df" not in st.session_state and db_edited is not None:
        st.session_state.original_df = db_original
        st.session_state.edited_df = db_edited
        st.session_state.inventory_ready = True
        st.session_state.last_committed_df = db_edited.copy()
        st.info("Loaded inventory from database.")
    elif "edited_df" not in st.session_state:
        st.info("Upload an Excel file to begin.")
        st.stop()

original_df: pd.DataFrame = st.session_state.original_df
edited_df: pd.DataFrame = st.session_state.edited_df

# Sidebar filters (priority first)
st.sidebar.header("Filters")
filter_any = st.sidebar.checkbox("Enable filters for all columns", value=False)

priority_present = [c for c in PRIORITY_COLS if c in edited_df.columns]
remaining = [c for c in edited_df.columns if c not in priority_present]
filter_cols = priority_present + (remaining if filter_any else [])

active_filters = {}
for col in filter_cols:
    s = edited_df[col].fillna("(blank)").astype(str)
    options = sorted(s.unique().tolist())
    selected = st.sidebar.multiselect(col, options, default=[])
    if selected:
        active_filters[col] = selected

view_df = edited_df.copy()
for col, selected in active_filters.items():
    s = view_df[col].fillna("(blank)").astype(str)
    view_df = view_df[s.isin(selected)]

# Column selection (defaults = priority)
default_visible = priority_present if priority_present else list(edited_df.columns)

visible_cols = st.multiselect(
    "Choose columns to display (tick/untick)",
    options=list(edited_df.columns),
    default=default_visible,
)

if not visible_cols:
    st.warning("Select at least one column.")
    st.stop()

# --- Editor ---
st.subheader("Edit table")
edited_view = st.data_editor(
    view_df[visible_cols],
    use_container_width=True,
    num_rows="dynamic",
    key="inventory_editor",
)

# Update full edited_df with edited columns only
edited_df.loc[edited_view.index, edited_view.columns] = edited_view
st.session_state.edited_df = edited_df
st.session_state.inventory_ready = True

# Persist inventory state (always)
save_inventory(original_df, edited_df)

st.markdown("---")

# --- Pending changes + remarks ---
st.subheader("Changes with Remark (commit to log)")

if "last_committed_df" not in st.session_state:
    st.session_state.last_committed_df = edited_df.copy()

pending = diff_cells(st.session_state.last_committed_df, edited_df)

if pending.empty:
    st.info("No new changes since last commit.")
else:
    st.write("Add remark for each change, then click **Commit changes with remarks**.")
    pending_edited = st.data_editor(
        pending,
        use_container_width=True,
        num_rows="fixed",
        key="pending_changes_editor",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        commit = st.button("Commit changes with remarks")
    with col2:
        st.caption("This will store the changed cells + remarks in database (safe even after session reset).")

    if commit:
        now = datetime.utcnow().isoformat()
        changes = []
        for _, r in pending_edited.iterrows():
            changes.append({
                "changed_at": now,
                "row_index": None if pd.isna(r.get("row_index")) else int(r.get("row_index")),
                "row_id_value": None if pd.isna(r.get("row_id_value")) else str(r.get("row_id_value")),
                "column_name": str(r["column_name"]),
                "old_value": None if pd.isna(r.get("old_value")) else str(r.get("old_value")),
                "new_value": None if pd.isna(r.get("new_value")) else str(r.get("new_value")),
                "remark": "" if pd.isna(r.get("remark")) else str(r.get("remark")),
            })
        insert_change_rows(changes)
        st.session_state.last_committed_df = edited_df.copy()
        st.success("Committed changes with remarks to database.")

st.markdown("---")
c1, c2 = st.columns(2)
with c1:
    out_bytes = to_excel_bytes(st.session_state.edited_df)
    st.download_button(
        "Download edited Excel",
        data=out_bytes,
        file_name="inventory_edited.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
with c2:
    if st.button("Reset edits back to original"):
        st.session_state.edited_df = original_df.copy()
        save_inventory(original_df, st.session_state.edited_df)
        st.session_state.last_committed_df = st.session_state.edited_df.copy()
        st.rerun()
