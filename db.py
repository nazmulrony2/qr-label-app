# db.py

def init_db() -> None:
    # কেন: Streamlit Cloud এ DB লাগবে না, কিন্তু init_db() কল আছে—তাই no-op।
    return

def save_qr_label_set(set_name: str, settings_json: str, items: list) -> int:
    # কেন: UI তে Save button আছে—DB ছাড়া যেন crash না করে।
    return 0