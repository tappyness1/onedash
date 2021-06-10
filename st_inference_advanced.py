import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import cv2
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import albumentations as A
import plotly.graph_objects as go
import plotly.express as px

# from torchvision import transforms
from albumentations.pytorch import ToTensorV2, ToTensor
from plotly.subplots import make_subplots
from src.datapipeline.inference_post import Dataset_Stats
# from src.datapipeline.clean_metrics import clean_metric_csv
# from src.models.model import Model

from PIL import Image
from ast import literal_eval
# from configparser import ConfigParser, ExtendedInterpolation
from annotated_text import annotated_text
# from src.models.eval.confusion_matrix import ConfusionMatrix

st.set_page_config(layout="wide")
pd.options.mode.chained_assignment = None  # default='warn'

def get_img_tensor(img_file_path):
    """[Retrieve the tensor information of a given image using cv2]

    Parameters
    ----------
    img_file_path : [str]
        [path of the image]

    Returns
    -------
    img : [tensor]
        [image expressed as tensor using cv2, resized to 608 by 608]
    """

    # Get img tensor
    img = cv2.imread(filename = img_file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(608,608),
        ToTensor()
        ],
        bbox_params=A.BboxParams(format='pascal_voc',
        label_fields=['class_labels']),
        )

    augmented = transform(image=img, bboxes = [], class_labels = [])
    # img comes out as int, need to change to float.
    img = augmented['image'].float()
    
    return img

@st.cache(suppress_st_warning=True)
def plot_all_boxes(img_as_tensor, gt_boxes, gt_labels, gt_status, predicted_boxes, predicted_labels, predicted_status, 
keep_matched=True, keep_mismatched=True, keep_empty=True):

    # draw all bounding boxes
    columns = ['x_min', 'y_min', 'x_max', 'y_max', 'label']
    df = pd.DataFrame(columns = columns)
    img = img_as_tensor.cpu()
    img = img.numpy().transpose(1, 2, 0)

    for i, gt_status in enumerate(gt_status):    
        x_min = gt_boxes[i][0] 
        x_max = gt_boxes[i][2] 
        y_min = gt_boxes[i][1] 
        y_max = gt_boxes[i][3] 
        label = gt_labels[i]
        to_add = {'x_min': int(x_min), 'y_min': int(y_max), 'x_max': int(x_max), 'y_max': int(y_min), 'label': int(label)}
        df = df.append(to_add, ignore_index = True)
        pts1 = (int(x_min), int(y_max))
        pts2 = (int(x_max), int(y_min))

        if keep_matched and gt_status == "MATCH":            
            cv2.rectangle(img, pts1, pts2, color=(0,255,0), thickness=1)
            cv2.putText(
                img,
                str(int(label)),
                (int(x_min),int(y_min)),
                cv2.FONT_HERSHEY_COMPLEX, 
                fontScale = 0.5, 
                color = (0,255,0)
                )
        elif keep_mismatched and gt_status == "MISMATCH":            
            cv2.rectangle(img, pts1, pts2, color=(0,255,0), thickness=1)
            cv2.putText(
                img,
                str(int(label)),
                (int(x_min),int(y_min)),
                cv2.FONT_HERSHEY_COMPLEX, 
                fontScale = 0.5, 
                color = (0,255,0)
                )
        elif keep_empty and gt_status == "NO":          
            cv2.rectangle(img, pts1, pts2, color=(0,255,0), thickness=1)           
            cv2.putText(
                img,
                str(int(label)),
                (int(x_min),int(y_min)),
                cv2.FONT_HERSHEY_COMPLEX, 
                fontScale = 0.5, 
                color = (0,255,0)
                )
    
    for i, predicted_status in enumerate(predicted_status):
        x_min = predicted_boxes[i][0] 
        x_max = predicted_boxes[i][2] 
        y_min = predicted_boxes[i][1] 
        y_max = predicted_boxes[i][3] 
        label = predicted_labels[i]
        to_add = {'x_min': int(x_min), 'y_min': int(y_max), 'x_max': int(x_max), 'y_max': int(y_min), 'label': int(label)}
        df = df.append(to_add, ignore_index = True)
        pts1 = (int(x_min), int(y_max))
        pts2 = (int(x_max), int(y_min))

        if keep_matched and predicted_status == "MATCH":            
            cv2.rectangle(img, pts1, pts2, color=(0,0,255), thickness=1)
            cv2.putText(
                img,
                str(int(label)),
                (int(x_max),int(y_min)),
                cv2.FONT_HERSHEY_COMPLEX, 
                fontScale = 0.5, 
                color = (0,0,255)
                )
        elif keep_mismatched and predicted_status == "MISMATCH":            
            cv2.rectangle(img, pts1, pts2, color=(255,0,0), thickness=1)
            cv2.putText(
                img,
                str(int(label)),
                (int(x_max),int(y_min)),
                cv2.FONT_HERSHEY_COMPLEX, 
                fontScale = 0.5, 
                color = (255,0,0)
                )
        elif keep_empty and predicted_status == "NO":          
            cv2.rectangle(img, pts1, pts2, color=(255,0,0), thickness=1)           
            cv2.putText(
                img,
                str(int(label)),
                (int(x_max),int(y_min)),
                cv2.FONT_HERSHEY_COMPLEX, 
                fontScale = 0.5, 
                color = (255,0,0)
                )
    
    return img

@st.cache(suppress_st_warning=True)
def get_df_mismatch(inference_csv):
    data = pd.read_csv(inference_csv)
    data_mismatch = data[~data['num_mismatch'].isin(['[]', '0'])]
    data_mismatch['gt_labels'] = data_mismatch['gt_labels'].apply(literal_eval)
    data_mismatch['gt_match_idx_list'] = data_mismatch['gt_match_idx_list'].apply(literal_eval)
    data_mismatch['gt_area_list'] = data_mismatch['gt_area_list'].apply(literal_eval)
    data_mismatch['gt_analysis'] = data_mismatch['gt_analysis'].apply(literal_eval)
    data_mismatch['gt_boxes_int'] = data_mismatch['gt_boxes_int'].apply(literal_eval)
    # data_mismatch['gt_match_idx_bb_list'] = data_mismatch['gt_match_idx_bb_list'].apply(literal_eval)

    # df_mismatches = pd.DataFrame(columns = ['Match_Type', 'GT_Class', 'Pred_Class', 'GT_Area', 'Pred_Area'])
    df_mismatches = pd.DataFrame(columns = ['Match_Type', 'GT_Class', 'Pred_Class', 'GT_Area'])
    for ind, row in data_mismatch.iterrows():
        for idx, elem in enumerate(row['gt_labels']):
            if row['gt_match_idx_list'][idx] =='-':
                pass
            else:
                # bbox = row['gt_match_idx_bb_list'][idx]
                # pred_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                gt_bbox = row['gt_boxes_int'][idx]
                gt_width = gt_bbox[2] - gt_bbox[0]
                gt_height = gt_bbox[3] - gt_bbox[1]
                elem_dict = {
                            'Match_Type': row['gt_analysis'][idx],
                            'GT_Class': row['gt_labels'][idx],
                            'Pred_Class': row['gt_match_idx_list'][idx],
                            'GT_Area': row['gt_area_list'][idx],
                            # 'Pred_Area': pred_area,
                            'GT_width' : gt_width,
                            'GT_height': gt_height}
                df_mismatches = df_mismatches.append(elem_dict, ignore_index = True)
    # df_mismatches['Match_Type'] = df_mismatches['Match_Type'].astype(int)
    df_mismatches['GT_Class'] = df_mismatches['GT_Class'].astype(int)
    df_mismatches['Pred_Class'] = df_mismatches['Pred_Class'].astype(int)
    df_mismatches['GT_Class'] = df_mismatches['GT_Class'].astype('category')
    df_mismatches['Pred_Class'] = df_mismatches['Pred_Class'].astype('category')

    new = df_mismatches["Pred_Class"].astype(str).copy() 
    df_mismatches['Class_GT_Pred'] = df_mismatches["GT_Class"].astype(str).str.cat(new, sep ="-")  
    df_mismatches['Height_Width_Ratio'] = df_mismatches['GT_height'] / df_mismatches['GT_width']
    df_mismatches['Height_Width_Ratio'] = df_mismatches['Height_Width_Ratio'].round(2)

    return df_mismatches

# def plot_boxplots(df_mismatches, column_name, filter = None):
      # seaborn boxplots  
#     if filter is not None:
#         df_mismatches_filtered = df_mismatches[df_mismatches['GT_Area'] < filter]
#     else: 
#         df_mismatches_filtered = df_mismatches.copy()
#     title_name = ['buoy', 'large military', 'large non-military','sailboat','small military', 'small non-military']
#     for i in range(6):
#         df_gt_filtered = df_mismatches_filtered[df_mismatches_filtered['GT_Class'] == (i+1)].copy()
#         fig, axes = plt.subplots(1, 2, figsize = (15,5) )
#         sns.boxplot(ax = axes [0], data = df_gt_filtered, y = column_name)
#         axes[0].set(xlabel = f'GT_Class - {(i+1)}')
#         sns.boxplot(ax = axes [1], data = df_gt_filtered, x = 'Class_GT_Pred', y = column_name)
#         fig.suptitle(f'Mismatch Analysis - {title_name[i]}', fontsize=16)
#         st.pyplot(fig)

def plot_boxplots(df_mismatches, column_name, class_labels= None, filter = None):
    """Plotly graphs for boxplot mismatches

    Args:
        df_mismatches ([type]): [description]
        column_name ([type]): [description]
        filter ([type], optional): [description]. Defaults to None.
    """
    if filter is not None:
        df_mismatches_filtered = df_mismatches[df_mismatches['GT_Area'] < filter]
    else: 
        df_mismatches_filtered = df_mismatches.copy()
    
    for i in range(len(class_labels)):
        df_gt_filtered = df_mismatches_filtered[df_mismatches_filtered['GT_Class'] == (i+1)].copy()
        fig = make_subplots(rows=1, cols=2, horizontal_spacing = 0.04)
        # trace0 = (go.Box(y = df_gt_filtered[column_name], name=f"All {title_name[i]} GT", boxmean = True))
        tplist = [str(x)+'-'+str(x) for x in range(1, len(class_labels) + 1)]
        left_df = df_mismatches_filtered[df_mismatches_filtered['Class_GT_Pred'].isin(tplist)]
        trace0 = (go.Box( x= left_df['Class_GT_Pred'], y = left_df[column_name], 
                  name=f"All Classes' GT", boxmean = True))
        trace1 = (go.Box(x = df_gt_filtered['Class_GT_Pred'], y = df_gt_filtered[column_name], name = f"(Mis)Matches for {class_labels[i]}", 
                  boxmean = True))
        fig.append_trace(trace0, 1, 1)
        fig.append_trace(trace1, 1, 2)
        fig.update_layout(title_text=f'Mismatch Analysis - {class_labels[i]}', height=700, legend=dict(
                            y=-0.05,
                            x=0.35,
                            orientation = 'h'
                        ))
        fig.update_traces(boxpoints='all', jitter = 0.3, pointpos = -1.5)
        yaxes_title = column_name.replace('_', ' ')
        yaxes_max = max([left_df[column_name].quantile(.95)*1.10, df_mismatches_filtered[column_name].quantile(.95)*1.10])
        fig.update_yaxes(range=[0, yaxes_max*1.05])
        fig.update_xaxes(categoryorder='category ascending')
        fig.update_yaxes(title_text=yaxes_title, row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)

def main(inference_csv, img_folder, names_file):    

    # Select Plot Option
    st.sidebar.markdown("## Select Mode of Analysis")
    option_view = st.sidebar.radio(
        "Select Presentation View",
        ('Dataset Level', 'Image Level'))    

    Dataset = Dataset_Stats(inference_csv, names_file)
    class_labels = Dataset.get_names_list()

    # Create a legend table with class names
    cls_legend_str = "Class Label Name:\n"
    for idx, cls_name in enumerate(class_labels, start=1):
        cls_legend_str = cls_legend_str + str(idx) + " - " + cls_name + "\n"        

    if option_view == 'Image Level':
        # Widgets Control
        st.sidebar.text("-------------------------")
        st.sidebar.markdown("## Select Bounding Boxes to show")
        keep_matched = st.sidebar.checkbox('Keep Matched', value=True)
        keep_mismatched = st.sidebar.checkbox('Keep Mismatched', value=True)
        keep_empty = st.sidebar.checkbox('Keep Empty', value=True)
        st.sidebar.text("-------------------------")
        st.sidebar.markdown("## Select Image")
        filter_only_mismatch = st.sidebar.checkbox('Only Images with Mislabels', value=False)

        # Read CSV
        df_inference = pd.read_csv(inference_csv)

        if filter_only_mismatch:
            df_inference = df_inference[df_inference['number_of_predictions'] > 0]
            df_inference = df_inference[df_inference['num_mismatch'] != "0"]
            df_inference = df_inference.reset_index()

        ##### Select Image for Inference
        choice_image = df_inference.image_file_name
               
        selected_image = st.sidebar.selectbox("Select Image to Plot", (choice_image))  
        st.sidebar.text("-------------------------")
        
        
        st.sidebar.markdown("## References")
        st.sidebar.text((cls_legend_str))

        st.sidebar.text("""
        Legend for Bounding Boxes: 
        Green - Ground Truth
        Blue - Prediction (Matched)
        Red - Prediction (No Match)
        """)

        # Get Path of Image
        img_file_path =  img_folder + selected_image

        st.title("Analysis - Image Level")
        st.subheader("Image Inference")
        # st.write(f"Perfoming Inference on Image: <{selected_image}>")
        
        # Filter out associated information of the image from the pre-loaded csv dataframe
        df_selected_image = df_inference[df_inference['image_file_name'] == selected_image]

        # Apply literal eval of columns containing information on Ground Truth
        df_selected_image.gt_labels = df_selected_image.gt_labels.apply(literal_eval)
        df_selected_image.gt_boxes = df_selected_image.gt_boxes.apply(literal_eval)
        df_selected_image.gt_boxes_int = df_selected_image.gt_boxes_int.apply(literal_eval)
        df_selected_image.gt_area_type = df_selected_image.gt_area_type.apply(literal_eval)
        df_selected_image.gt_match = df_selected_image.gt_match.apply(literal_eval)
        df_selected_image.gt_analysis = df_selected_image.gt_analysis.apply(literal_eval)
        df_selected_image.truncated_list = df_selected_image.truncated_list.apply(literal_eval)
        df_selected_image.occluded_list = df_selected_image.occluded_list.apply(literal_eval)

        # Apply literal eval of columns containing information on Predictions
        df_selected_image.predicted_boxes = df_selected_image.predicted_boxes.apply(literal_eval)
        # df_selected_image.predicted_boxes_conf = df_selected_image.predicted_boxes_conf.apply(literal_eval)
        df_selected_image.predicted_boxes_int = df_selected_image.predicted_boxes_int.apply(literal_eval)
        df_selected_image.predicted_labels = df_selected_image.predicted_labels.apply(literal_eval)
        df_selected_image.predicted_confidence = df_selected_image.predicted_confidence.apply(literal_eval)
        df_selected_image.predicted_labels_conf = df_selected_image.predicted_labels_conf.apply(literal_eval)
        df_selected_image.predicted_confidence_conf = df_selected_image.predicted_confidence_conf.apply(literal_eval)
        df_selected_image.prediction_match = df_selected_image.prediction_match.apply(literal_eval)
        df_selected_image.prediction_analysis = df_selected_image.prediction_analysis.apply(literal_eval)

        # Retreive cell values in the form of variables
        for _,rows in df_selected_image.iterrows():
            # image_file_path = rows["image_file_path"]
            number_of_gt = rows["number_of_gt"]
            gt_boxes = rows["gt_boxes"]
            # gt_boxes_int = rows["gt_boxes_int"]
            gt_labels = rows["gt_labels"]
            gt_area_type = rows["gt_area_type"]
            gt_match = rows["gt_match"]
            gt_analysis = rows["gt_analysis"]
            truncated_list = rows["truncated_list"]
            occluded_list = rows["occluded_list"]
            number_of_predictions = rows["number_of_predictions"]
            predicted_boxes = rows["predicted_boxes"]
            # predicted_boxes_conf = rows["predicted_boxes_conf"]
            # predicted_boxes_int = rows["predicted_boxes_int"]
            predicted_labels = rows["predicted_labels"]
            # predicted_confidence = rows["predicted_confidence"] 
            predicted_labels_conf = rows["predicted_labels_conf"]
            predicted_confidence_conf = rows["predicted_confidence_conf"] 
            prediction_match = rows["prediction_match"]
            prediction_analysis = rows["prediction_analysis"]

            truncated_list_bool = list(map(bool, truncated_list))
            truncated_list_bool = list(map(str, truncated_list_bool))
            occluded_list_bool = list(map(bool, occluded_list))     
            occluded_list_bool = list(map(str, occluded_list_bool))    
            predicted_confidence_conf_round = [round(confidence, 4) for confidence in predicted_confidence_conf]
        
        df_gt = pd.DataFrame({
            "GT_L": gt_labels,
            "T": truncated_list_bool,
            "O": occluded_list_bool,
            "GT_Status": gt_analysis,
            "Size" :gt_area_type,
            "Matches": gt_match
        })

        df_gt.insert(0, 'GT_IDX', df_gt.index)

        df_prediction = pd.DataFrame({
            "P_L": predicted_labels_conf,
            "Conf": predicted_confidence_conf_round,
            "P_Status": prediction_analysis,
            "Matches": prediction_match
        })

        df_prediction.insert(0, 'P_IDX', df_prediction.index)    
        
        df_gt_prediction = pd.merge(df_gt, df_prediction, how='outer', on="Matches")
        df_gt_prediction = df_gt_prediction.sort_values("GT_Status")
        df_gt_prediction.loc[:, 'IOU'] = df_gt_prediction.Matches.map(lambda x: x[2])
        df_gt_prediction.fillna('-', inplace=True)
        df_gt_prediction = df_gt_prediction.reset_index(drop=True)

        df_gt_prediction_show = df_gt_prediction.reindex(['GT_L','GT_Status','P_L','P_Status','Conf','IOU',"Size",'T','O'], axis=1)        

        img_tensor = get_img_tensor(img_file_path)

        img_final = plot_all_boxes(
            img_tensor, 
            gt_boxes=gt_boxes, 
            gt_labels=gt_labels,
            gt_status=gt_analysis,
            predicted_boxes=predicted_boxes,
            predicted_labels=predicted_labels_conf,
            predicted_status=prediction_analysis, 
            keep_matched=keep_matched,
            keep_mismatched=keep_mismatched,
            keep_empty=keep_empty)     

        st.image(img_final, caption=img_file_path, width=500, clamp=True, use_column_width=False)

        st.dataframe(df_gt_prediction_show)

        st.subheader("Meaning of Labels:")
        st.text("-------------------------")
                
        annotated_text(
            ("GT_L", "Ground Truth Label", "#afa"),
            ("GT_Status", "Ground Truth Status", "#afa"),
            ("P_L", "Prediction Label", "#faa"),
            ("P_Status", "Prediction Status", "#faa"),
            ("P_Conf", "Prediction Confidence", "#faa"),
            ("IOU", "Intersection over Union of Prediction and Ground Truth", "#8ef"),
            ("Size", "Ground Truth Area Size", "#afa"),
            ("T", "Truncated Ground Truth", "#afa"),
            ("O", "Occluded Ground Truth", "#afa")
        )

        # st.title("Prediction Details")
        # st.write(f"Number of Ground Truth: {number_of_gt}")
        # st.write(f"Number of Prediction: {number_of_predictions}")
        # st.write(f"Number of Prediction >= Confidence Threshold: {number_of_predictions_conf}") 
        # st.write(f"Number of Prediction < Confidence Threshold: {number_of_predictions - number_of_predictions_conf}") 
        # st.text("-------------------------")
        # st.write(f"Number of Matches: {len(match_correct_list)} / {number_of_gt}")
        # st.write(f"Number of Wrong Matches: {len(match_wrong_list)} / {number_of_gt}")
        # st.write(f"Number of Ground Truth not matched: {number_of_gt - len(match_correct_list) - len(match_wrong_list)}/{number_of_gt}")
        # st.write(f"Number of Prediction hitting a Ground Truth: {len(match_correct_list) + len(match_wrong_list)}/{number_of_predictions_conf}")
        # st.write(f"Number of Prediction not hitting any Ground Truth: {number_of_predictions_conf - len(match_correct_list) - len(match_wrong_list)}/{number_of_predictions_conf}")
        
        # st.title("Confusion Matrix of Image")
        # CM.process_full_matrix()
        # cm_df = CM.return_as_df()

        # st.dataframe(cm_df)

    else:
        st.sidebar.text("------------------------------")
        st.sidebar.markdown("## Select Analysis to show")
        keep_dataset = st.sidebar.checkbox('Counts of Ground Truth and Prediction Overview', value=True)
        keep_tpfpfn = st.sidebar.checkbox('TP/FP/FN Overview', value=True)
        keep_mislabels = st.sidebar.checkbox('Mislabels Overview', value=True)
        # keep_AP = st.sidebar.checkbox('Show AP Metrics', value=False)
        # keep_CM = st.sidebar.checkbox('Show Full Confusion Matrix', value=False)
        keep_mismatch_analysis = st.sidebar.checkbox('Mislabel Area Analysis', value=True)
        keep_mismatch_height = st.sidebar.checkbox('Mislabel Height Analysis', value=False)
        keep_mismatch_width = st.sidebar.checkbox('Mislabel Width Analysis', value=False)
        keep_mismatch_height_width_ratio = st.sidebar.checkbox('Mislabel Height to Width Ratio Analysis', value=False)
        st.sidebar.text("-------------------------")
     
        st.sidebar.markdown("## References")
        st.sidebar.text((cls_legend_str))
        
        tp_list, fp_list, fn_list = Dataset.get_tpfpfn_list()
        gt_list, prediction_list = Dataset.get_gtprediction_list()
        df_mismatches = get_df_mismatch(inference_csv)

        if keep_dataset:            
            st.title("Analysis - Dataset Level")
            nms_thresh, conf_thresh, iou_thresh = Dataset.get_threshold()
            
            st.write(f"NMS Threshold: {nms_thresh}")
            st.write(f"Confidence Threshold: {conf_thresh}")
            st.write(f"IOU Threshold: {iou_thresh}")

            # Plot Ground Truth vs Predictions            
            st.header("Number of Ground Truth and Predictions")
            st.write(f"Total Number of Ground Truth: {sum(gt_list)}")
            st.write(f"Total Number of Predictions Generated: {sum(prediction_list)}")
            fig = go.Figure(data=[
                go.Bar(
                    name='Ground Truth', x=class_labels, y=gt_list, 
                    text=gt_list, textposition='auto'),
                go.Bar(
                    name='Predictions', x=class_labels, y=prediction_list, 
                    text=prediction_list, textposition='auto'),
            ])

            fig.update_layout(barmode='group', title_text='Ground Truth vs Predictions')
            st.plotly_chart(fig, use_container_width=True)
            

        if keep_tpfpfn:

            # Plot TP/FP/FN
            st.header("Number of True Positive (TP), False Positive (FP), False Negative (FN)")
            st.write(f"Total Number of TP: {sum(tp_list)}")
            st.write(f"Total Number of FP: {sum(fp_list)}")
            st.write(f"Total Number of FN: {sum(fn_list)}")
            st.write(f"Precision: {round(sum(tp_list)/(sum(fp_list)+sum(tp_list)),4)}")
            st.write(f"Recall: {round(sum(tp_list)/(sum(fn_list)+sum(tp_list)),4)}")

            fig = go.Figure(data=[
                go.Bar(
                    name='True Positive', x=class_labels, y=tp_list, 
                    text=tp_list, textposition='auto'),
                go.Bar(
                    name='False Positive', x=class_labels, y=fp_list, 
                    text=fp_list, textposition='auto'),
                go.Bar(
                    name='False Negative', x=class_labels, y=fn_list, 
                    text=fn_list, textposition='auto'),
            ])

            fig.update_layout(barmode='group', title_text='True Positive vs False Positive vs False Negative')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Meaning of Labels:")
            st.text("-------------------------")
            st.write("##### True Positive (TP):  Matching Ground Truth and Prediction with the Correct Class Label and Accepted IOU Threshold")
            st.write("##### False Positive (FP): Prediction that did not get matched with a Ground Truth")
            st.write("##### False Negative (FN): Ground Truth that is not detected")
            st.write("##### Precision:           Precision is the ability of a classifier to identify only relevant objects")
            st.write("##### Recall:              Recall is a metric which measures the ability of a classifier to find all the ground truths")

        if keep_mislabels:
            mislabels_dict, mislabel_total = Dataset.get_mislabels_dict()

            st.header("Mislabels Overview")
            st.write(f"Total Number of Mislabels: {mislabel_total}")            
            sort_mislabel = st.checkbox('Sort by Mislabels Name', value=False)

            df_mislabels = pd.DataFrame.from_dict(mislabels_dict, orient='index', columns=["Counts"])
            df_mislabels = df_mislabels.sort_values("Counts", ascending=False)
            df_mislabels.index.name = 'Mislabel'
            df_mislabels.reset_index(inplace=True)
            # st.dataframe(df_mislabels)

            if sort_mislabel:
                df_mislabels = df_mislabels.sort_values("Mislabel")
            
            fig = px.bar(df_mislabels, x='Mislabel', y='Counts', text="Counts")
            fig.update_layout(title_text='Mislabel Counts = [Actual, Predicted]')
            st.plotly_chart(fig, use_container_width=True)

        if keep_mismatch_analysis:
            st.header("Mislabel Analysis - Ground Truth Area")
            st.write("""
            - This looks at the area of Ground Truth (GT) for those which were mislabelled grouped by class
            - By analysing the median of the mislabelled GT Area and comparing them to the TP GT Area, 
              you would be able to better determine if the model did not perform well due to the area of the 
              bounding box             
            """)
            plot_boxplots(df_mismatches, 'GT_Area', class_labels = class_labels, filter = 1000)

        if keep_mismatch_height_width_ratio:
            st.header("Mislabel Analysis - Height to Width Ratio")
            st.write("""
            - This looks at the Height-to-Width (HtW) ratio of Ground Truth (GT) bounding boxes for those which were mislabelled grouped by class
            - If the HtW ratio is more than 1, it would mean that the bounding box is taller than it is wide and vice versa if the ratio is less than 1
            - By comparing the HtW ratio of the mislabelled GTs and the TP GTs, we can see if the nature of the bounding boxes plays a part in confusing the model 
            when identifying certain objects
            - eg - the model would often confuse an object like a tall ship for sailboats when the GT is actually a large non-military ship               
            """)
            plot_boxplots(df_mismatches, 'Height_Width_Ratio', class_labels = class_labels)

        if keep_mismatch_height:
            st.header("Mislabel Analysis - Ground Truth Height")
            st.write("""
            - This looks at the Height of Ground Truth (GT) bounding boxes for those which were mislabelled grouped by class
            - By comparing the Height of mislabelled GTs and the TP GTs, we can see if the nature of the bounding boxes plays a part in confusing the model 
            when identifying certain objects
            - eg - the model would often confuse an object like a tall ship for sailboats when the GT is actually a large non-military ship      
            - conversely, if a tall ship is confused for a normally short one like small non-military, then it means that there are other features that is 
            confusing the model         
            """)
            plot_boxplots(df_mismatches, 'GT_height', class_labels = class_labels)
        
        if keep_mismatch_width: 
            st.header('Mislabel Analysis - Ground Truth Width')
            st.write("""
            - This looks at the Width of Ground Truth (GT) bounding boxes for those which were mislabelled grouped by class
            - By comparing the Width of mislabelled GTs and the TP GTs, we can see if the nature of the bounding boxes plays a part in confusing the model 
            when identifying certain objects
            - eg - the model would often confuse an object like a long ship for large non-military vessel when the GT is actually a small non-military ship      
            """)
            plot_boxplots(df_mismatches, 'GT_width', class_labels = class_labels)

        # if keep_AP:
        #     st.title("AP Metrics")
        #     hello = clean_metric_csv("csv/metrics.csv")
        #     # hello = clean_metric_csv(hello)
        #     st.dataframe(hello)

        # if keep_CM:
        #     st.title(f"Confusion Matrix")
        #     a = np.loadtxt(open("csv/confusion_matrix.csv", "rb"), delimiter=",", skiprows=1)
        #     st.write(a)
        
if __name__ == "__main__":
    main(
        inference_csv="data/outputs/model_analysis/df_inference_details_nms_0.4_conf_0.5_iou_0.4.csv",
        img_folder="data/data_batch123/",
        names_file="csv/obj.names"
    )