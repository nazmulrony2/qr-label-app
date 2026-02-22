import streamlit as st

st.set_page_config(page_title="Inventory App", layout="wide")
st.title("Inventory App")

st.write("Choose a page:")

st.page_link("pages/1_Inventory_Editor.py", label="📋 Inventory Editor", icon="📋")
st.page_link("pages/2_QR_Label_PDF.py", label="🏷️ QR Label PDF Generator", icon="🏷️")
