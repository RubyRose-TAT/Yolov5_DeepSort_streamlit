import streamlit as st

import time
import os
import sys
import argparse
from pathlib import Path
from io import StringIO
from PIL import Image

from track import *
from yolov5.detect import *
from Project_Introduction import introduction

# 进度条
st.markdown(
    """
    <style>
        .stProgress > div > div > div > div {
            background-image: linear-gradient(to right, #99ff99 , #00ccff);
        }
    </style>""",
    unsafe_allow_html=True,
)  

def get_subdirs(b='.'):
    '''
        返回特定路径中的所有子目录
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        返回runs目录下最新文件夹
    '''
    return max(get_subdirs(os.path.join('runs', 'track', 'yolov5', 'weights')), key=os.path.getmtime)

def get_detection_folder_y():
    return max(get_subdirs(os.path.join('yolov5','runs', 'detect')), key=os.path.getmtime)

if __name__ == '__main__':

    st.title('Yolov5_DeepSort_Streamlit App')

    opt = parse_opt_d()
    opt_y = parse_opt()

    source = ("项目介绍", "图片检测", "视频检测")
    source_index = st.sidebar.selectbox("功能选择:", range(
        len(source)), format_func=lambda x: source[x])
    
    if source_index == 0:
        introduction()
    
    # opt
    opt.img = 1280
    opt.classes = 0
    opt_y.classes = 0

    if source_index != 0:
        yolo_model_options = ("crowdhuman_yolov5m", "yolov5x", "yolov5x6", "yolov5s", "yolov5s6", "yolov5n", "yolov5n6", "yolov5m", "yolov5m6", "yolov5l", "yolov5l6")
        yolo_weights = st.sidebar.selectbox(
            label="yolov5模型:", options=yolo_model_options,
        )
        opt.yolo_model = 'yolov5/weights/' + yolo_weights + '.pt'
        opt_y.weights = 'yolov5/weights/' + yolo_weights + '.pt'
    
    if source_index == 2:
        deep_sort_model_options = ("osnet_x0_25_market1501", "osnet_x0_5_market1501", "osnet_x0_75_msmt17", "osnet_x1_0_msmt17")
        deep_sort_weights = st.sidebar.selectbox(
            label="DeepSort模型:", options=deep_sort_model_options,
        )
        opt.deep_sort_model = 'deep_sort/deep/checkpoint/' + deep_sort_weights + '.pth'

    if source_index == 1:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片:", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'inference/images/{uploaded_file.name}')
                opt_y.source = f'inference/images/{uploaded_file.name}'
        else:
            is_valid = False
    if source_index == 2:
        uploaded_file = st.sidebar.file_uploader("上传视频:", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("inference", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'inference/videos/{uploaded_file.name}'
        else:
            is_valid = False
    if source_index != 0:
        if is_valid:
            st.markdown("---")

            if st.button('开始检测'):
                if source_index == 1:
                    with torch.no_grad():
                        main(opt_y)
                    with st.spinner(text='Preparing Images'):
                        for img in os.listdir(get_detection_folder_y()):
                            st.image(str(Path(f'{get_detection_folder_y()}') / img))

                        st.balloons()
                if source_index == 2:
                    with torch.no_grad():
                        detect(opt)

                    with st.spinner(text='Preparing Video'):
                        for vid in os.listdir(get_detection_folder()):
                            st.video(str(Path(f'{get_detection_folder()}') / vid))

                        st.markdown("---")

                        st.balloons()