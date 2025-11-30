import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

st.set_page_config(layout="wide")

# 读取数据
df = pd.read_csv("outputs/umap.csv")

# 左右布局
left, right = st.columns([2.5, 1])

st.title("甲骨文 UMAP（右侧显示图像）")

with left:
    fig = px.scatter(
        df,
        x="umap_x",
        y="umap_y",
        color="label",
        custom_data=["file"],
        height=800
    )

    # 显示 Plotly 图
    hovered = st.plotly_chart(fig, use_container_width=True)

# 初始化状态
if "hover_img" not in st.session_state:
    st.session_state.hover_img = None

# 监听前端 hover
st.components.v1.html(
    """
    <script>
    window.addEventListener("DOMContentLoaded", () => {
        const iframe = window.parent.document.querySelector('iframe[title="stPlotlyChart"]');
        if (!iframe) return;

        iframe.onload = () => {
            const plot = iframe.contentWindow.document.querySelector('.plotly');

            plot.on('plotly_hover', function(data){
                let file = data.points[0].customdata[0];
                window.parent.postMessage({hover_file: file}, "*");
            });
        };
    });

    window.addEventListener("message", (event) => {
        if (event.data.hover_file) {
            window.parent.document.dispatchEvent(
                new CustomEvent("updateHover", {detail: event.data.hover_file})
            );
        }
    });
    </script>
    """,
    height=0
)

# 用 JS 触发 Python 更新 session_state
hover_file = st.experimental_get_query_params().get("hover", None)
if hover_file:
    st.session_state.hover_img = hover_file[0]

with right:
    st.subheader("悬停图像：")

    if st.session_state.hover_img:
        img = Image.open(st.session_state.hover_img)
        st.image(img, width=350)
    else:
        st.info("请将鼠标移动到左侧散点图上。")
