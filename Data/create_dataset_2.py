"""
This script to extract skeleton joints position and score.

- This 'annot_folder' is a action class and bounding box for each frames that came with dataset.
    Should be in format of [frame_idx, action_cls, xmin, ymin, xmax, ymax]
        Use for crop a person to use in pose estimation model.
- If have no annotation file you can leave annot_folder = '' for use Detector model to get the
    bounding box.
"""

import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from DetectorLoader import TinyYOLOv3_onecls
from fn import vis_frame_fast
from PoseEstimateLoader import SPPE_FastPose

save_path = "Data/output/UR_falls-pose+score.csv"

# annot_file = "Data-UR/urfall-cam0-falls.csv"  # from create_dataset_1.py
annot_file = "UR-DATA/urfall-cam0-falls.csv"  # from create_dataset_1.py
data_folder = "UR-DATA/Fall_sequences"
# video_folder = "ADL_sequences"
annot_folder = "Data-UR/falldata/urfall/Annotation_files"  # bounding box annotation for each frame.  --> 目前沒用到

W, H = 244, 244  # shape of new images (resize

# DETECTION MODEL.
detector = TinyYOLOv3_onecls()

# POSE MODEL.
# inp_h = 320
# inp_w = 256
# pose_estimator = SPPE_FastPose(inp_h, inp_w)

inp_h = 256
inp_w = 192
pose_estimator = SPPE_FastPose("resnet50", inp_h, inp_w)
# class_names = [
#     "Standing",  # 0
#     "Walking",  # 1
#     "Sitting",  # 2
#     "Lying Down",  # 3
#     "Stand up",  # 4
#     "Sit down",  # 5
#     "Fall Down",  # 6
#     # "No Person",  # 7
# ]  # label.

class_names = [
    "Temporary pose",
    "Lying Down",
    "Standing",
]  # label.

# with score.
columns = [
    "video",
    "frame",
    "Nose_x",
    "Nose_y",
    "Nose_s",
    "LShoulder_x",
    "LShoulder_y",
    "LShoulder_s",
    "RShoulder_x",
    "RShoulder_y",
    "RShoulder_s",
    "LElbow_x",
    "LElbow_y",
    "LElbow_s",
    "RElbow_x",
    "RElbow_y",
    "RElbow_s",
    "LWrist_x",
    "LWrist_y",
    "LWrist_s",
    "RWrist_x",
    "RWrist_y",
    "RWrist_s",
    "LHip_x",
    "LHip_y",
    "LHip_s",
    "RHip_x",
    "RHip_y",
    "RHip_s",
    "LKnee_x",
    "LKnee_y",
    "LKnee_s",
    "RKnee_x",
    "RKnee_y",
    "RKnee_s",
    "LAnkle_x",
    "LAnkle_y",
    "LAnkle_s",
    "RAnkle_x",
    "RAnkle_y",
    "RAnkle_s",
    "label",
]


def normalize_points_with_size(points_xy, width, height, flip=False):
    points_xy[:, 0] /= width
    points_xy[:, 1] /= height
    if flip:
        points_xy[:, 0] = 1 - points_xy[:, 0]
    return points_xy


# 讀取整份 annotation csv
annot_all = pd.read_csv(
    annot_file,
    # names=["video", "frame", "label"],
    usecols=[0, 1, 2],
)
print(annot_all.head(3))
events = [
    f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))
]
events.sort()
for (
    nb_event,
    event,
) in enumerate(events):
    number = event if len(event) == 2 else f"0{event}"
    path_to_images = os.path.join(data_folder, event, f"fall-{number}-cam0-rgb")
    print(path_to_images)
    # ---> ex: path_to_images: UR-DATA/Fall_sequences/9/fall-09-cam0-rgb
    df = pd.DataFrame(columns=columns)
    cur_row = 0

    # Pose Labels.
    frames_label = annot_all[annot_all["video"] == f"fall-{number}"].reset_index(
        drop=True
    )
    # print(frames_label)

    frames_count = len(frames_label)
    frame_size = (W, H)

    # Bounding Boxs Labels.
    annot_file = os.path.join(annot_folder, event + ".txt")
    annot = None
    if os.path.exists(annot_file):
        annot = pd.read_csv(
            annot_file,
            header=None,
            names=["frame_idx", "class", "xmin", "ymin", "xmax", "ymax"],
        )
        annot = annot.dropna().reset_index(drop=True)

        assert frames_count == len(annot), "frame count not equal! {} and {}".format(
            frames_count, len(annot)
        )

    for index, row in frames_label.iterrows():
        img_path = os.path.join(
            path_to_images, f'fall-{number}-cam0-rgb-{str(row["frame"]).zfill(3)}.png'
        )
        print(img_path)

        frame = cv2.imread(img_path)
        frame = cv2.resize(frame, (W, H))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if row["label"] == "-1":
            cls_idx = len(class_names) - 1
        else:
            cls_idx = int(row["label"])
        # cls_idx = int(
        #     frames_label[frames_label["frame"] == row["frame"]].iloc[0]["label"]
        # )
        if annot:
            bb = np.array(annot.iloc[index - 1, 2:].astype(int))
        else:
            bb = detector.detect(frame)
            # 這邊要防呆一下
            if bb is None:
                continue
            bb = detector.detect(frame)[0, :4].numpy().astype(int)
        bb[:2] = np.maximum(0, bb[:2] - 5)
        bb[2:] = np.minimum(frame_size, bb[2:] + 5) if bb[2:].any() != 0 else bb[2:]

        result = []
        if bb.any() != 0:
            result = pose_estimator.predict(
                frame, torch.tensor(bb[None, ...]), torch.tensor([[1.0]])
            )

        if len(result) > 0:
            pt_norm = normalize_points_with_size(
                result[0]["keypoints"].numpy().copy(), frame_size[0], frame_size[1]
            )
            pt_norm = np.concatenate((pt_norm, result[0]["kp_score"]), axis=1)

            # idx = result[0]['kp_score'] <= 0.05
            # pt_norm[idx.squeeze()] = np.nan
            row = [event, index, *pt_norm.flatten().tolist(), cls_idx]
            scr = result[0]["kp_score"].mean()
        else:
            row = [event, index, *[np.nan] * (13 * 3), cls_idx]
            scr = 0.0

        df.loc[cur_row] = row
        cur_row += 1

        # VISUALIZE.
        frame = vis_frame_fast(frame, result)
        frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        frame = cv2.putText(
            frame,
            "Frame: {}, Pose: {}, Score: {:.4f}".format(index, cls_idx, scr),
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        frame = frame[:, :, ::-1]

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

    if os.path.exists(save_path):
        df.to_csv(save_path, mode="a", header=False, index=False)
    else:
        df.to_csv(save_path, mode="w", index=False)
