import streamlit as st

from streamlit_gallery import apps, components, prediction
from streamlit_gallery.utils.page import page_group

def main():
    page = page_group("p")

    with st.sidebar:
        st.title("🎈 Okld's Gallery")

        with st.expander("✨ APPS", True):
            page.item("Start", apps.gallery, default=True)

        with st.expander("🧩 COMPONENTS", True):
            page.item("Training⭐", components.elements)
            page.item("Predict⭐", prediction.prediction)
        

    page.show()

if __name__ == "__main__":
    st.set_page_config(page_title="Streamlit Gallery by Okld", page_icon="🎈", layout="wide")
    main()
