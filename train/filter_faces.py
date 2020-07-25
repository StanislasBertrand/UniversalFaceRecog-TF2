import os
import cv2
import numpy as np
import sys
sys.path.insert(1, '../recognition/face_embedder_network')
from recognition.face_embedder_network import FaceEmbedder
from scipy.spatial.distance import cosine


def filter_faces(celebs_file, faces_dir, faces_filtered_dir, weights_path):
    celebs = []
    with open(celebs_file) as f:
        for line in f:
            celebs.append(line.strip())

    embeddor = FaceEmbedder(weights_path)


    for celeb in celebs:
        celeb_faces_dir = os.path.join(faces_dir, celeb.lower().replace(" ", "_"))
        celeb_filtered_faces_dir = os.path.join(faces_filtered_dir, celeb.lower().replace(" ", "_"))
        if not os.path.isdir(celeb_faces_dir):
            print(f"could not find photos of {celeb}, skipping")
            continue
        if not os.path.isdir(celeb_filtered_faces_dir):
            os.mkdir(celeb_filtered_faces_dir)
        else:
            print(f"filtered faces photos of {celeb} laready exist at {celeb_filtered_faces_dir}, skipping")
            continue
        embeddings = {}
        mean_embedding = np.zeros([1,512])
        for file in os.listdir(celeb_faces_dir):
            img = cv2.imread(os.path.join(celeb_faces_dir, file))
            embedding_vector = embeddor.model.predict(img[np.newaxis,:,:,:])
            embeddings[file] = embedding_vector
            mean_embedding += embedding_vector
        mean_embedding /= len(embeddings)

        for key in list(embeddings.keys()):
            dist = cosine(mean_embedding, embeddings[key])
            if dist<0.5:
                img = cv2.imread(celeb_faces_dir + "/" + key)
                cv2.imwrite(celeb_filtered_faces_dir + "/" + key, img)

