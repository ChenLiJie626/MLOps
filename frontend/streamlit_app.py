import streamlit as st

from streamlit_gallery import apps, components
from streamlit_gallery.utils.page import page_group

def main():
    page = page_group("p")

    with st.sidebar:
        st.title("🎈 阿斯克勒庇俄斯")

        with st.expander("✨ APPS", True):
            page.item("Start", apps.gallery, default=True)

        with st.expander("🧩 COMPONENTS", True):
            page.item("Training⭐", components.elements)
            page.item("Predict⭐", components.prediction)
        

    page.show()

if __name__ == "__main__":
    st.set_page_config(page_title="阿斯克勒庇俄斯", page_icon="🎈", layout="wide")
    main()
