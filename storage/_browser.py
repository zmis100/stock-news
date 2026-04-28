"""브라우저 LocalStorage 어댑터 (싱글톤).

여러 storage 모듈이 같은 LocalStorage 인스턴스를 공유하도록.
"""
import streamlit as st
from streamlit_local_storage import LocalStorage


def get_ls() -> LocalStorage:
    """LocalStorage 인스턴스 반환 (rerun 간 재사용)."""
    if "_ls_singleton" not in st.session_state:
        st.session_state["_ls_singleton"] = LocalStorage()
    return st.session_state["_ls_singleton"]
