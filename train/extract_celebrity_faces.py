import os
import cv2
import numpy as np
from retinaface.retinaface import RetinaFace
from retinaface.alignment import extract_aligned_faces
from recognition.face_embedder_network import FaceEmbedder

photos_dir = "./data/images/bing_images/"
celebrities_file = "./data/celebrities.txt"
faces_dir = "./data/images/faces/"

celebs = []
with open(celebrities_file) as f:
    for line in f:
        celebs.append(line.strip())

detector = RetinaFace("models/retinafaceweights.npy", True, 0.4)
embeddor = FaceEmbedder("models/faceEmeddings.npy")

for celeb in celebs:
    celeb_photos_dir = os.path.join(photos_dir, celeb.lower().replace(" ", "_"))
    celeb_faces_dir = os.path.join(faces_dir, celeb.lower().replace(" ", "_"))
    if not os.path.isdir(celeb_photos_dir):
        print(f"could not find photos of {celeb}, skipping")
        continue
    aligned_faces = []
    face_vectors = []
    count_faces = 0
    for file in os.listdir(celeb_photos_dir):
        img = cv2.imread(os.path.join(celeb_photos_dir, file))
        faces, landmarks = detector.detect(img, 0.9)
        for i in range(len(faces)):
            landmarks_xs = landmarks[0][:,0]
            landmarks_ys = landmarks[0][:,1]
            points = np.concatenate([landmarks_xs, landmarks_ys], axis = 0).reshape(1,10)
            aligned_face = extract_aligned_faces(img, points)[0]
            cv2.imwrite(os.path.join(celeb_faces_dir, str(count_faces) + ".jpg"), aligned_face)
            count_faces += 1
            
