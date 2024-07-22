import logging
import sys
from contextlib import contextmanager
from io import StringIO
import json
import random
import threading
import time
import streamlit as st
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from pathlib import Path
from streamlit import session_state as state
from streamlit_elements import elements, sync, event
from types import SimpleNamespace
from .dashboard import Dashboard, Editor, Card, DataGrid, Radar, Pie, Player
from streamlit_echarts import st_echarts
import os
import urllib
import requests

'''get the backend url'''
if os.environ.get("BACKEND_URL") is not None:
    BACKEND_URL = os.environ.get("BACKEND_URL")
else:
    BACKEND_URL = "http://0.0.0.0:8000"

MODELS_URL = urllib.parse.urljoin(BACKEND_URL, "models")
TRAIN_URL = urllib.parse.urljoin(BACKEND_URL, "train")
PREDICT_URL = urllib.parse.urljoin(BACKEND_URL, "predict")
DELETE_URL = urllib.parse.urljoin(BACKEND_URL, "delete")



lock = threading.Lock()
# ECharts 配置
def get_echarts_options(accuracy_data):
    return {
        "title": {"text": "训练准确率"},
        "xAxis": {"type": "category", "data": [f"epoch{i}" for i in range(1, len(accuracy_data) + 1)]},
        "yAxis": {"type": "value"},
        "series": [{"data": accuracy_data, "type": "line"}],
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
    }

def setup_dashboard():
    if "w" not in state:
        board = Dashboard()
        w = SimpleNamespace(
            dashboard=board,
            editor=Editor(board, 0, 0, 6, 11, minW=3, minH=3),
            player=Player(board, 0, 12, 6, 7, minH=4),
            pie=Pie(board, 6, 0, 6, 7, minW=3, minH=4),
            radar=Radar(board, 0, 0, 12, 7, minW=4, minH=4),
            card=Card(board, 6, 7, 3, 7, minW=2, minH=4),
            data_grid=DataGrid(board, 6, 13, 6, 7, minH=4),
        )
        state.w = w

       
    else:
        w = state.w

def display_elements():
    with elements("demo"): # type: ignore
        event.Hotkey("ctrl+s", sync(), bindInputs=True, overrideDefault=True)
        with state.w.dashboard(rowHeight=57):
            state.w.player()
            state.w.data_grid(state.w.player.get_content())

@contextmanager
def st_redirect(src, dst_placeholder, height=200):
    output_func = lambda text: dst_placeholder.text_area("训练日志", text, height=height)
    
    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if get_script_run_ctx():
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write

@contextmanager
def st_stderr(dst_placeholder):
    with st_redirect(sys.stderr, dst_placeholder):
        yield

def log_and_chart():
    col1, col2 = st.columns(2)
    log_placeholder = col1.empty()  # 日志输出区域
    chart_placeholder = col2.empty()  # 图表输出区域
    
    url = TRAIN_URL
    response = requests.post(url).json()

    with st_stderr(log_placeholder):
        accuracy_data = []
        for epoch in range(1, 5):
            time.sleep(1)
            accuracy = random.uniform(70, 100)
            accuracy_data.append(accuracy)
            logging.warning(f"Epoch {epoch}: Train Accuracy: {accuracy:.2f}%")
            with chart_placeholder:
                st_echarts(get_echarts_options(accuracy_data))
    st.session_state["is_training"] = False
    st.balloons()
    st.success("训练完成！")

def main():
    setup_dashboard()
    display_elements()
    if st.session_state.is_training and not lock.locked():
        with lock:
            with st.spinner('Wait for it...'):
                log_and_chart()


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
