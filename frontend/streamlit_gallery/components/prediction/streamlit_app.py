import streamlit as st

from streamlit_gallery.utils.readme import readme
from streamlit_quill import st_quill
import time
import os

def main():
    
    # 创建临时目录
    temp_dir = "tempDir"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    # 设置页面标题和描述
    st.title('预测模型界面')
    st.write('上传文件，等待处理，并下载预测结果文件。')

    # 上传文件
    uploaded_file = st.file_uploader('上传文件')

    if uploaded_file is not None:
        # 保存上传的文件
        with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("文件上传成功!")

        # 显示处理按钮
        if st.button('开始处理'):
            with st.spinner('文件处理中，请稍候...'):
                # 模拟发送请求到后端并等待响应
                time.sleep(20)  # 模拟后端处理时间
                
                # 假设后端返回了处理后的文件路径
                result_file_path = os.path.join("tempDir", "result_" + uploaded_file.name)

                # 保存假设的处理结果文件
                with open(result_file_path, "w") as f:
                    f.write("这是处理后的文件内容。")

                st.success('文件处理完成！')
                
                # 提供下载链接
                with open(result_file_path, "rb") as f:
                    st.download_button('下载预测结果文件', f, file_name="00_pred.nii.gz")  # "result_" + uploaded_file.name

                # 展示六个图片框
                st.write("处理后的图片展示:")
                image_files = ["front.png", "back.png", "left.png", "right.png", "top.png", "bottom.png"]
                cols = st.columns(3)
                for i, image_file in enumerate(image_files):
                    image_path = os.path.join(os.path.join(temp_dir, "res_img"), image_file)  # 假设图片存放在tempDir目录下
                    if os.path.exists(image_path):
                        cols[i % 3].image(image_path, caption=image_file.split('.')[0].capitalize(), use_column_width=True)
                    else:
                        cols[i % 3].write(f"{image_file} not found.")




if __name__ == "__main__":
    main()