import streamlit as st

from pathlib import Path
import streamlit.components.v1 as components


def main():
    st.markdown("""
        <style>
               .block-container
                {
                    padding-top: 0rem;
                    padding-right: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                    padding-bottom: 0rem;
                    margin-top: 0rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)
    path = str(Path(__file__).parents[0])+"/"
    
    # 读取HTML文件的内容
    with open(path+'index.html', 'r', encoding='utf-8') as file:
        html_content = file.read()

    # 使用Streamlit组件嵌入HTML内容
    components.html(html_content, height=800)


if __name__ == "__main__":
    main()
