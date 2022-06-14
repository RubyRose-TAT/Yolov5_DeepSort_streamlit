import streamlit as st
from PIL import Image

def introduction():
    st.sidebar.markdown('---')
    st.sidebar.image(Image.open('inference/data/bus.jpg'))

    st.markdown('---')

    st.markdown("""

                # 1. 相关技术

                ### 1.1 web应用搭建

                基于streamlit搭建web应用。Streamlit是一个专门针对机器学习和数据科学应用而生的开发框架,它能够快速的帮助我们创建定制化的用户交互应用


                ### 1.2 目标检测算法

                在一帧图片中检测出行人的位置,不同行人之间不进行区分。

                ###### 1.2.1 传统两阶段目标检测模型
                先提取候选框,再对每一个候选框进行逐一的甄别,速度慢。

                ###### 1.2.1 单阶段目标检测模型(YOLO)
                传入图片就能直接得到结果。
                

                ### 1.3 目标跟踪算法

                在检测任务的基础上进一步实现了每一个特定目标的跟踪。不仅定位到了行人这一类目标出现的位置,而且在不同画面中去针对特定的人进行了匹配,从而实现对每一个目标的跟踪效果。

                # 2. YOLOv5

    """)
    st.image(Image.open('inference/data/yolov5.png'), caption='')
    st.markdown("""
                **COCO AP val:** 表示在COCO val2017数据集上的5000张图像,经过推理得到的 mAP 评估指标。

                **GPU Speed:** 在COCO val2017数据集上,使用V100实例下,每幅图像的平均推理时间
    """)
    with st.expander(""):
            st.markdown("""
                        mAP: mean Average Precision, 即各类别AP的平均值

                        AP: PR曲线下面积

                        PR曲线: Precision-Recall曲线

                        Precision: TP / (TP + FP)

                        Recall: TP / (TP + FN)

                        TP: IoU>0.5的检测框数量(同一Ground Truth只计算一次)

                        FP: IoU<=0.5的检测框,或者是检测到同一个GT的多余检测框的数量

                        FN: 没有检测到的GT的数量
    """)
    st.markdown("""
                不同大小模型以及输入尺度对应的mAP、推理速度、参数数量以及理论计算量(每秒所进行的浮点数运算次数)FLOPs:
    """)
    st.image(Image.open('inference/data/0.png'), caption='')
    st.markdown("""
                # 3. DeepSort
                Deepsort是实现目标跟踪的算法,从sort演变而来。其使用卡尔曼滤波器预测所检测对象的运动轨迹,匈牙利算法将它们与新检测的目标匹配。

                deepsort之所以可以大量避免IDSwitch,是因为deepsort算法中特征提取网络可以将目标检测框中的特征提取出来并保存,在目标被遮挡后又从新出现后,利用前后的特征对比可以将遮挡的后又出现的目标和遮挡之前的追踪的目标从新找到,大大减少了目标在遮挡后,追踪失败的可能。
                
                ### 3.1 多目标追踪的主要步骤

                    1.获取原始视频帧

                    2.利用目标检测器对视频帧中的目标进行检测

                    3.将检测到的目标的框中的特征提取出来,该特征包括表观特征(方便特征对比避免ID switch)和运动特征(运动特征方便卡尔曼滤波对其进行预测)

                    4.计算前后两帧目标之前的匹配程度(利用匈牙利算法和级联匹配),为每个追踪到的目标分配ID。

                ### 3.2 sort算法

                    Deepsort的前身是sort算法,sort算法的核心是卡尔曼滤波算法和匈牙利算法。

                    卡尔曼滤波算法作用：该算法的主要作用就是当前的一系列运动变量去预测下一时刻的运动变量,但是第一次的检测结果用来初始化卡尔曼滤波的运动变量。

                    匈牙利算法的作用：简单来讲就是解决分配问题,就是把一群检测框和卡尔曼预测的框做分配,让卡尔曼预测的框找到和自己最匹配的检测框,达到追踪的效果。
    """)
    with st.expander("sort流程"):
        st.markdown(' sort工作流程如下图所示:')
        st.image(Image.open('inference/data/sort.png'), caption=' Detections是通过目标检测到的框,Tracks是轨迹信息')
        st.markdown("""
                整个算法的工作流程如下：

                (1)将第一帧检测到的结果创建其对应的Tracks。将卡尔曼滤波的运动变量初始化,通过卡尔曼滤波预测其对应的框

                (2)将该帧目标检测的框框和上一帧通过Tracks预测的框框一一进行IOU匹配,再通过IOU匹配的结果计算其代价矩阵(cost matrix,其计算方式是1-IOU)

                (3)将(2)中得到的所有的代价矩阵作为匈牙利算法的输入,得到线性的匹配的结果,这时候我们得到的结果有三种:
                第一种是Tracks失配(Unmatched Tracks),我们直接将失配的Tracks删除;
                第二种是Detections失配(Unmatched Detections),我们将这样的Detections初始化为一个新的Tracks(new Tracks);
                第三种是检测框和预测的框框配对成功,这说明我们前一帧和后一帧追踪成功,将其对应的Detections通过卡尔曼滤波更新其对应的Tracks变量

                (4)反复循环(2)-(3)步骤,直到视频帧结束

        """)

    st.markdown("""
                ### 3.3 Deepsort算法

                由于sort算法还是比较粗糙的追踪算法,当物体发生遮挡的时候,特别容易丢失自己的ID。
                而Deepsort算法在sort算法的基础上增加了级联匹配(Matching Cascade)和新轨迹的确认(confirmed)。Tracks分为确认态(confirmed),和不确认态(unconfirmed),新产生的Tracks是不确认态的;
                不确认态的Tracks必须要和Detections连续匹配一定的次数(默认是3)才可以转化成确认态。确认态的Tracks必须和Detections连续失配一定次数(默认30次),才会被删除
    
    """)

    with st.expander("Deepsort流程"):
        st.markdown(' Deepsort算法的工作流程如下图所示')
        st.image(Image.open('inference/data/sort.png'), caption='')
        st.markdown("""
                    整个算法的工作流程如下：

                    (1)将第一帧次检测到的结果创建其对应的Tracks。将卡尔曼滤波的运动变量初始化,通过卡尔曼滤波预测其对应的框框。这时候的Tracks一定是unconfirmed的。
                    
                    (2)将该帧目标检测的框框和第上一帧通过Tracks预测的框框一一进行IOU匹配,再通过IOU匹配的结果计算其代价矩阵(cost matrix,其计算方式是1-IOU)。
                    
                    (3)将(2)中得到的所有的代价矩阵作为匈牙利算法的输入,得到线性的匹配的结果,这时候我们得到的结果有三种,第一种是Tracks失配(Unmatched Tracks),我们直接将失配的Tracks(因为这个Tracks是不确定态了,如果是确定态的话则要连续达到一定的次数(默认30次)才可以删除)删除;
                    第二种是Detections失配(Unmatched Detections),我们将这样的Detections初始化为一个新的Tracks(new Tracks);第三种是检测框和预测的框框配对成功,这说明我们前一帧和后一帧追踪成功,将其对应的Detections通过卡尔曼滤波更新其对应的Tracks变量。
                    
                    (4)反复循环(2)-(3)步骤,直到出现确认态(confirmed)的Tracks或者视频帧结束。
                    
                    (5)通过卡尔曼滤波预测其确认态的Tracks和不确认态的Tracks对应的框框。将确认态的Tracks的框框和是Detections进行级联匹配(之前每次只要Tracks匹配上都会保存Detections其的外观特征和运动信息,默认保存前100帧,利用外观特征和运动信息和Detections进行级联匹配,这么做是因为确认态(confirmed)的Tracks和Detections匹配的可能性更大)。
                    
                    (6)进行级联匹配后有三种可能的结果。第一种,Tracks匹配,这样的Tracks通过卡尔曼滤波更新其对应的Tracks变量。第二第三种是Detections和Tracks失配,这时将之前的不确认状态的Tracks和失配的Tracks一起和Unmatched Detections一一进行IOU匹配,再通过IOU匹配的结果计算其代价矩阵(cost matrix,其计算方式是1-IOU)。
                    
                    (7)将(6)中得到的所有的代价矩阵作为匈牙利算法的输入,得到线性的匹配的结果,这时候我们得到的结果有三种,第一种是Tracks失配(Unmatched Tracks),我们直接将失配的Tracks(因为这个Tracks是不确定态了,如果是确定态的话则要连续达到一定的次数(默认30次)才可以删除)删除;
                    第二种是Detections失配(Unmatched Detections),我们将这样的Detections初始化为一个新的Tracks(new Tracks);第三种是检测框和预测的框框配对成功,这说明我们前一帧和后一帧追踪成功,将其对应的Detections通过卡尔曼滤波更新其对应的Tracks变量。
                    
                    (8)反复循环(5)-(7)步骤,直到视频帧结束。

        """)


