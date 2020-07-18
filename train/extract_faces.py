import os
import cv2
import numpy as np
from retinaface.FaceDetector import FaceDetector
from retinaface.alignment import extract_aligned_faces

photos_dir = "./data/images/bing_images/"
celebrities_file = "./data/celebrities.txt"
faces_dir = "./data/images/faces/"

celebs = []
with open(celebrities_file) as f:
    for line in f:
        celebs.append(line.strip())

detector = FaceDetector("models/retinafaceweights.npy", False, 0.4)

for celeb in celebs:
    celeb_photos_dir = os.path.join(photos_dir, celeb.lower().replace(" ", "_"))
    celeb_faces_dir = os.path.join(faces_dir, celeb.lower().replace(" ", "_"))
    if not os.path.isdir(celeb_photos_dir):
        print(f"could not find photos of {celeb}, skipping")
        continue
    if not os.path.isdir(celeb_faces_dir):
        os.mkdir(celeb_faces_dir)
    aligned_faces = []
    face_vectors = []
    count_faces = 0
    for file in os.listdir(celeb_photos_dir):
        try:
            img = cv2.imread(os.path.join(celeb_photos_dir, file))
            faces, landmarks = detector.detect(img, 0.9)
            for i in range(len(faces)):
                landmarks_xs = landmarks[i][:,0]
                landmarks_ys = landmarks[i][:,1]
                points = np.concatenate([landmarks_xs, landmarks_ys], axis = 0).reshape(1,10)
                aligned_face = extract_aligned_faces(img, points)[0]
                cv2.imwrite(os.path.join(celeb_faces_dir, str(count_faces) + ".jpg"), aligned_face)
                count_faces += 1
        except Exception as e:
            print(e)
            continue