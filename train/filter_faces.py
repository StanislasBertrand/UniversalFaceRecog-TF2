import os
import cv2
import numpy as np
from recognition.face_embedder_network import FaceEmbedder
from scipy.spatial.distance import cosine

data_dir = "./data/images/faces/"
celebrities_file = "./data/celebrities.txt"

celebs = []
with open(celebrities_file) as f:
    for line in f:
        celebs.append(line.strip())

embeddor = FaceEmbedder("./models/faceEmbeddings.npy")


for celeb in celebs:
    celeb_faces_dir = os.path.join(data_dir, celeb.lower().replace(" ", "_"))
    celeb_filtered_faces_dir = os.path.join("./data/images/faces_filtered/", celeb.lower().replace(" ", "_"))
    if not os.path.isdir(celeb_faces_dir):
        print(f"could not find photos of {celeb}, skipping")
        continue
    if not os.path.isdir(celeb_filtered_faces_dir):
        os.mkdir(celeb_filtered_faces_dir)
    embeddings = {}
    mean_embedding = np.zeros([1,512])
    for file in os.listdir(celeb_faces_dir):
        img = cv2.imread(os.path.join(celeb_faces_dir, file))
        embedding_vector = embeddor.model.predict(img[np.newaxis,:,:,:])
        embeddings[file] = embedding_vector
        mean_embedding += embedding_vector
    mean_embedding /= len(embeddings)
    distances = []

    for key in list(embeddings.keys()):
        dist = cosine(mean_embedding, embeddings[key])
        if dist<0.5:
            img = cv2.imread(celeb_faces_dir + "/" + key)
            cv2.imwrite(celeb_filtered_faces_dir + "/" + key, img)

