import json
from streamlit_elements import media, mui, sync, lazy
from .dashboard import Dashboard
import streamlit as st
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

class Player(Dashboard.Item):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_address, self.name, self.role, self.gpu = "", "", "", ""
        url = urllib.parse.urljoin(BACKEND_URL, "update_user")
        response = requests.post(url, data=json.dumps({"server_address": self.server_address, "name": self.name, "role": self.role, "gpu": self.gpu})).json()
        self.users_list = response["users_list"]

    def _set_address(self, event):
        #sync()
        self.server_address = event.target.value

    def _set_name(self, event):
        #sync()
        self.name = event.target.value
    
    def get_content(self):
        result = []
        result += [{"id": i, "server_address": user["server_address"], "name": user["name"], "role": user["role"], "gpu": user["gpu"]} for i, user in enumerate(self.users_list)]
        return json.dumps(result)

    def __call__(self):
        with mui.Paper(key=self._key, sx={"display": "flex", "flexDirection": "column", "borderRadius": 3, "overflow": "hidden"}, elevation=1):
            with self.title_bar(padding="10px 15px", dark_switcher=False):
                mui.icon.OndemandVideo()
                mui.Typography("Role:")
            self._render_input("Server IP", lazy(self._set_address))
            self._render_input("Name", lazy(self._set_name))
            self._render_select("Role", lazy(sync(None, "role")), [("Server", "server"), ("Client", "client")])
            self._render_select("GPU", lazy(sync(None, "gpu")), [("NVIDIA GeForce RTX 4090", "4090"), ("NVIDIA GeForce RTX 3060", "3060"), ("NVIDIA TITAN RTX", "titan")])
            with mui.CardActions:
                mui.Button("添加用户", onClick=self._handle_button_click)

    def _render_input(self, label, on_change):
        with mui.Box(sx={"padding": "10px 15px", "display": "flex", "alignItems": "center"}):
            mui.Typography(label, sx={"marginRight": "10px", "width": "100px", "textAlign": "right"})
            mui.TextField(label="", variant="outlined", fullWidth=True, sx={"flexGrow": 1}, onChange=on_change)

    def _render_select(self, label, on_change, options):
        with mui.Box(sx={"padding": "10px 15px", "display": "flex", "alignItems": "center", "marginTop": "10px"}):
            mui.Typography(label, sx={"marginRight": "10px", "width": "100px", "textAlign": "right"})
            mui.Select(defaultValue=None, fullWidth=True, onChange=on_change, children=[mui.MenuItem(text, value=value) for text, value in options])
    
    def _render_button(self, text):
        with mui.Box(sx={"padding": "10px 15px", "marginTop": "10px", "display": "flex", "justifyContent": "center"}):
            # mui.Button(text, variant="contained", color="primary", onClick=sync())
            '''click to backend to add device'''
            mui.Button(text, variant="contained", color="primary", onClick=self._handle_button_click())
    
    def _handle_button_click(self):
        sync()
        # 获取表单中的输入信息
        self.role = st.session_state.role.props.value if hasattr(st.session_state.role, "props") else st.session_state.role
        self.gpu = st.session_state.gpu.props.value if hasattr(st.session_state.gpu, "props") else st.session_state.gpu
        
        # 输出获取到的信息
        print("Server Address:", self.server_address)
        print("Name:", self.name)
        print("Role:", self.role)
        print("Server GPU:", self.gpu)

        url = urllib.parse.urljoin(BACKEND_URL, "update_user")
        print(url)
        response = requests.post(url, data=json.dumps({"server_address": self.server_address, "name": self.name, "role": self.role, "gpu": self.gpu})).json()
        self.users_list = response["users_list"]
        print(self.users_list)

