import pickle
import cv2
import argparse
import csv
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import pandas as pd
import math
import PySimpleGUI as sg


feature_names = [
    "back_nose_x",
    "back_nose_y",
    "back_nose_z",
    "back_left_eye_x",
    "back_left_eye_y",
    "back_left_eye_z",
    "back_right_eye_x",
    "back_right_eye_y",
    "back_right_eye_z",
    "back_left_ear_x",
    "back_left_ear_y",
    "back_left_ear_z",
    "back_right_ear_x",
    "back_right_ear_y",
    "back_right_ear_z",
    "back_left_shoulder_x",
    "back_left_shoulder_y",
    "back_left_shoulder_z",
    "back_right_shoulder_x",
    "back_right_shoulder_y",
    "back_right_shoulder_z",
    "back_left_elbow_x",
    "back_left_elbow_y",
    "back_left_elbow_z",
    "back_right_elbow_x",
    "back_right_elbow_y",
    "back_right_elbow_z",
    "back_left_wrist_x",
    "back_left_wrist_y",
    "back_left_wrist_z",
    "back_right_wrist_x",
    "back_right_wrist_y",
    "back_right_wrist_z",
    "back_left_hip_x",
    "back_left_hip_y",
    "back_left_hip_z",
    "back_right_hip_x",
    "back_right_hip_y",
    "back_right_hip_z",
    "back_left_knee_x",
    "back_left_knee_y",
    "back_left_knee_z",
    "back_right_knee_x",
    "back_right_knee_y",
    "back_right_knee_z",
    "back_left_ankle_x",
    "back_left_ankle_y",
    "back_left_ankle_z",
    "back_right_ankle_x",
    "back_right_ankle_y",
    "back_right_ankle_z",
    "side_nose_x",
    "side_nose_y",
    "side_nose_z",
    "side_left_eye_x",
    "side_left_eye_y",
    "side_left_eye_z",
    "side_right_eye_x",
    "side_right_eye_y",
    "side_right_eye_z",
    "side_left_ear_x",
    "side_left_ear_y",
    "side_left_ear_z",
    "side_right_ear_x",
    "side_right_ear_y",
    "side_right_ear_z",
    "side_left_shoulder_x",
    "side_left_shoulder_y",
    "side_left_shoulder_z",
    "side_right_shoulder_x",
    "side_right_shoulder_y",
    "side_right_shoulder_z",
    "side_left_elbow_x",
    "side_left_elbow_y",
    "side_left_elbow_z",
    "side_right_elbow_x",
    "side_right_elbow_y",
    "side_right_elbow_z",
    "side_left_wrist_x",
    "side_left_wrist_y",
    "side_left_wrist_z",
    "side_right_wrist_x",
    "side_right_wrist_y",
    "side_right_wrist_z",
    "side_left_hip_x",
    "side_left_hip_y",
    "side_left_hip_z",
    "side_right_hip_x",
    "side_right_hip_y",
    "side_right_hip_z",
    "side_left_knee_x",
    "side_left_knee_y",
    "side_left_knee_z",
    "side_right_knee_x",
    "side_right_knee_y",
    "side_right_knee_z",
    "side_left_ankle_x",
    "side_left_ankle_y",
    "side_left_ankle_z",
    "side_right_ankle_x",
    "side_right_ankle_y",
    "side_right_ankle_z",
]

def calcAngle(x1,y1,x2,y2):
  angles = []
  for i in range(len(x1)):
    angles.append(math.atan2(y2[i] - y1[i], x2[i] - x1[i]) * 180/math.pi)
  return angles

class FeatureEngineeringTransformer():
  def __init__(self):
    self.angles_joints = {
        'left_torso': ('left_hip','left_shoulder'),
        'right_torso': ('right_hip','right_shoulder'),
        'left_ulna': ('left_wrist','left_elbow'),
        'right_ulna': ('right_wrist','right_elbow'),
        'left_los': ('left_eye','nose'),#line of sight
        'right_los': ('right_eye','nose')
    }

  def fit(self,x, y=None):
    return self

  def transform(self,x):
    for angle in self.angles_joints:
      joint1, joint2 = self.angles_joints[angle]
      x[f'back_{angle}_angle'] = calcAngle(
          x1 = x[f'back_{joint1}_x'].values,
          y1 = x[f'back_{joint1}_y'].values,
          x2 = x[f'back_{joint2}_x'].values,
          y2 = x[f'back_{joint2}_y'].values
      )
      x[f'side_{angle}_angle'] = calcAngle(
          x1 = x[f'side_{joint1}_x'].values,
          y1 = x[f'side_{joint1}_y'].values,
          x2 = x[f'side_{joint2}_x'].values,
          y2 = x[f'side_{joint2}_y'].values
      )
    return x

class FeatureSelectionTransformer():
  def __init__(self, mode):
    self.mode = mode
  
  def fit(self,x, y=None):
    return self
  
  def transform(self, x, y=None):
    if self.mode == "back":
      cols_to_drop = [c for c in x.columns if "side_" in c]
    elif self.mode == "side":
      cols_to_drop = [c for c in x.columns if "back_" in c]
    elif self.mode == "combined":
      cols_to_drop = []
    
    x = x[[c for c in x.columns if c not in cols_to_drop]]
    return x




class VideoLabeller():
    def __init__(self):
        self.input_size = 192
        self.module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        # Dictionary that maps from joint names to keypoint indices.
        self.KEYPOINT_DICT = {
                'nose': 0,
                'left_eye': 1,
                'right_eye': 2,
                'left_ear': 3,
                'right_ear': 4,
                'left_shoulder': 5,
                'right_shoulder': 6,
                'left_elbow': 7,
                'right_elbow': 8,
                'left_wrist': 9,
                'right_wrist': 10,
                'left_hip': 11,
                'right_hip': 12,
                'left_knee': 13,
                'right_knee': 14,
                'left_ankle': 15,
                'right_ankle': 16
        }
        # Maps bones to a matplotlib color name.
        self.KEYPOINT_EDGE_INDS_TO_COLOR = {
                (0, 1): 'm',
                (0, 2): 'c',
                (1, 3): 'm',
                (2, 4): 'c',
                (0, 5): 'm',
                (0, 6): 'c',
                (5, 7): 'm', 
                (7, 9): 'm',
                (6, 8): 'c',
                (8, 10): 'c',
                (5, 6): 'y',
                (5, 11): 'm',
                (6, 12): 'c',
                (11, 12): 'y',
                (11, 13): 'm',
                (13, 15): 'm',
                (12, 14): 'c',
                (14, 16): 'c'
        }

    def _keypoints_and_edges_for_display(self, keypoints_with_scores,
                                                                            height,
                                                                            width,
                                                                            keypoint_threshold=0.11):
        """Returns high confidence keypoints and edges for visualization.

        Args:
            keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
                the keypoint coordinates and scores returned from the MoveNet model.
            height: height of the image in pixels.
            width: width of the image in pixels.
            keypoint_threshold: minimum confidence score for a keypoint to be
                visualized.

        Returns:
            A (keypoints_xy, edges_xy, edge_colors) containing:
                * the coordinates of all keypoints of all detected entities;
                * the coordinates of all skeleton edges of all detected entities;
                * the colors in which the edges should be plotted.
        """
        keypoints_all = []
        keypoint_edges_all = []
        edge_colors = []
        num_instances, _, _, _ = keypoints_with_scores.shape
        for idx in range(num_instances):
            kpts_x = keypoints_with_scores[0, idx, :, 1]
            kpts_y = keypoints_with_scores[0, idx, :, 0]
            kpts_scores = keypoints_with_scores[0, idx, :, 2]
            kpts_absolute_xy = np.stack(
                    [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
            kpts_above_thresh_absolute = kpts_absolute_xy[
                    kpts_scores > keypoint_threshold, :]
            keypoints_all.append(kpts_above_thresh_absolute)

            for edge_pair, color in self.KEYPOINT_EDGE_INDS_TO_COLOR.items():
                if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                        kpts_scores[edge_pair[1]] > keypoint_threshold):
                    x_start = kpts_absolute_xy[edge_pair[0], 0]
                    y_start = kpts_absolute_xy[edge_pair[0], 1]
                    x_end = kpts_absolute_xy[edge_pair[1], 0]
                    y_end = kpts_absolute_xy[edge_pair[1], 1]
                    line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                    keypoint_edges_all.append(line_seg)
                    edge_colors.append(color)
        if keypoints_all:
            keypoints_xy = np.concatenate(keypoints_all, axis=0)
        else:
            keypoints_xy = np.zeros((0, 17, 2))

        if keypoint_edges_all:
            edges_xy = np.stack(keypoint_edges_all, axis=0)
        else:
            edges_xy = np.zeros((0, 2, 2))
        return keypoints_xy, edges_xy, edge_colors

    def draw_prediction_on_image(self,
            image, keypoints_with_scores, crop_region=None, close_figure=False,
            output_image_height=None):
        """Draws the keypoint predictions on image.

        Args:
            image: A numpy array with shape [height, width, channel] representing the
                pixel values of the input image.
            keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
                the keypoint coordinates and scores returned from the MoveNet model.
            crop_region: A dictionary that defines the coordinates of the bounding box
                of the crop region in normalized coordinates (see the init_crop_region
                function below for more detail). If provided, this function will also
                draw the bounding box on the image.
            output_image_height: An integer indicating the height of the output image.
                Note that the image aspect ratio will be the same as the input image.

        Returns:
            A numpy array with shape [out_height, out_width, channel] representing the
            image overlaid with keypoint predictions.
        """
        height, width, channel = image.shape
        aspect_ratio = float(width) / height
        fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
        # To remove the huge white borders
        fig.tight_layout(pad=0)
        ax.margins(0)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.axis('off')

        im = ax.imshow(image)
        line_segments = LineCollection([], linewidths=(4), linestyle='solid')
        ax.add_collection(line_segments)
        # Turn off tick labels
        scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

        (keypoint_locs, keypoint_edges,
        edge_colors) = self._keypoints_and_edges_for_display(
                keypoints_with_scores, height, width)

        line_segments.set_segments(keypoint_edges)
        line_segments.set_color(edge_colors)
        if keypoint_edges.shape[0]:
            line_segments.set_segments(keypoint_edges)
            line_segments.set_color(edge_colors)
        if keypoint_locs.shape[0]:
            scat.set_offsets(keypoint_locs)

        if crop_region is not None:
            xmin = max(crop_region['x_min'] * width, 0.0)
            ymin = max(crop_region['y_min'] * height, 0.0)
            rec_width = min(crop_region['x_max'], 0.99) * width - xmin
            rec_height = min(crop_region['y_max'], 0.99) * height - ymin
            rect = patches.Rectangle(
                    (xmin,ymin),rec_width,rec_height,
                    linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect)

        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
                fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        if output_image_height is not None:
            output_image_width = int(output_image_height / height * width)
            image_from_plot = cv2.resize(
                    image_from_plot, dsize=(output_image_width, output_image_height),
                        interpolation=cv2.INTER_CUBIC)
        return image_from_plot

    def movenet(self,input_image):
        """Runs detection on an input image.

        Args:
            input_image: A [1, height, width, 3] tensor represents the input image
                pixels. Note that the height/width should already be resized and match the
                expected input resolution of the model before passing into this function.

        Returns:
            A [1, 1, 17, 3] float numpy array representing the predicted keypoint
            coordinates and scores.
        """
        model = self.module.signatures['serving_default']

        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores

    def get_keypoints(self,frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        input_image = tf.expand_dims(image_rgb, axis=0)
        input_image = tf.image.resize_with_pad(input_image, self.input_size, self.input_size)
        keypoints_with_scores = self.movenet(input_image)
        return keypoints_with_scores
        
    def draw_keypoints(self,frame, keypoints):
        display_image = tf.expand_dims(frame, axis=0)
        display_image = tf.cast(tf.image.resize_with_pad(
            display_image, 1280, 1280), dtype=tf.int32)
        output_overlay = self.draw_prediction_on_image(
            np.squeeze(display_image.numpy(), axis=0), keypoints)
        return output_overlay

    def engineer_features_mock(self,keypoints):
        return list(keypoints)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--back', type=str,
                        help='path of the back view video')
    parser.add_argument('--side', type=str,
                        help='path of the side view video')
    args = parser.parse_args()
    
    with open('./pipelines/side_engFalse_LR.pkl', 'rb') as file:
        model_side = pickle.load(file)
    
    with open('./pipelines/back_engFalse_LR.pkl', 'rb') as file:
        model_back = pickle.load(file)
    
    with open('./pipelines/combined_engFalse_LR.pkl', 'rb') as file:
        model_combined = pickle.load(file)

    lab = VideoLabeller()
    print("\n\n\n\n Finished Initialization \n\n\n\n")
    vidcap1 = cv2.VideoCapture(args.back)
    success1,image1 = vidcap1.read()
    len1 = int(vidcap1.get(cv2.CAP_PROP_FRAME_COUNT))
    vidcap2 = cv2.VideoCapture(args.side)
    success2,image2 = vidcap2.read()
    len2 = int(vidcap2.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = np.min([len1,len2])

    # first loop - manually label 1 out of every 6 frames
    count = 0
    while success1 and success2:

        back_keypoints = lab.get_keypoints(image1)
        back_image = lab.draw_keypoints(image1, back_keypoints)

        side_keypoints = lab.get_keypoints(image2)
        side_image = lab.draw_keypoints(image2, side_keypoints)

        image_with_keypoints = cv2.hconcat([back_image, side_image])

        features = list(back_keypoints.flatten()) + list(side_keypoints.flatten())
        features = lab.engineer_features_mock(features)

        pipeline_input = pd.Series(features).to_frame().transpose()
        pipeline_input.columns = feature_names

        pred_side = model_side.predict(pipeline_input)[0]
        pred_side_str = "side dataset: good posture" if pred_side==1 else "side dataset: bad posture"
        cv2.putText(
            image_with_keypoints, #numpy array on which text is written
            pred_side_str, #text
            (1300,100), #position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX, #font family
            1.5, #font size
            (0, 0, 255, 255), #font color
            2) #font stroke

        pred_back = model_back.predict(pipeline_input)[0]
        pred_back_str = "back dataset: good posture" if pred_back==1 else "back dataset: bad posture"
        cv2.putText(
            image_with_keypoints, #numpy array on which text is written
            pred_back_str, #text
            (100,100), #position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX, #font family
            1.5, #font size
            (0, 0, 255, 255), #font color
            2), #font stroke
            

        pred_combined = model_combined.predict(pipeline_input)[0]
        pred_combined_str = "combined dataset: good posture" if pred_combined==1 else "combined dataset: bad posture"
        cv2.putText(
            image_with_keypoints, #numpy array on which text is written
            pred_combined_str, #text
            (800,1000), #position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX, #font family
            1.5, #font size
            (0, 0, 255, 255), #font color
            2) #font stroke

        cv2.imshow('jpg', image_with_keypoints)
        cv2.waitKey(1)

        success1,image1 = vidcap1.read()
        success2,image2 = vidcap2.read()
        count += 1



    cv2.destroyAllWindows()

