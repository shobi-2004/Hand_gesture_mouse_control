#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

def draw_circle(image, x, y, roudness, color):
    """draw circle"""
    cv2.circle(image, (int(x), int(y)), roudness, color,
               thickness=5, lineType=cv2.LINE_8, shift=0)

def calculate_distance(lankdmark1, landmark2):
    """
    Calculate Euclidean distance

    Parameters
    ----------
    landmark1 : list
    landmark2 : list

    Returns
    -------
    distance : float
    """
    v = np.array([lankdmark1[0], lankdmark1[1]]) - \
        np.array([landmark2[0], landmark2[1]])
    distance = np.linalg.norm(v)
    return distance

def calculate_moving_average(landmark, ran, LiT):   # (座標、いくつ分の平均か、移動平均を格納するリスト)
    """
    Calculate moving averages

    Parameters
    ----------
    landmark : list
        landmark
    ran : int
        range
    LiT : list
        list in time

    Returns
    -------
    moving average : float
    """

    while len(LiT) < ran:               # ran個分のデータをLiTに追加（最初だけ）
        LiT.append(landmark)
    LiT.append(landmark)                # LiTの更新（最後に追加）
    if len(LiT) > ran:                  # LiTの更新（最初を削除）
        LiT.pop(0)
    return sum(LiT)/ran

def load_gestures(gestures_folder):
    """
    Load gestures from the gestures folder

    Parameters
    ----------
    gestures_folder : str
        Path to the gestures folder

    Returns
    -------
    gestures : dict
        Dictionary of gestures with filenames as keys and images as values
    """
    gestures = {}
    for filename in os.listdir(gestures_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            gesture_name = os.path.splitext(filename)[0]
            image_path = os.path.join(gestures_folder, filename)
            gestures[gesture_name] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return gestures
