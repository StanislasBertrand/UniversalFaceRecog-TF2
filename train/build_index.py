import os
import cv2
import numpy as np
import nmslib
import json
from recognition.face_embedder_network import FaceEmbedder

faces_dir = "./data/images/faces_filtered/"
celebrities_file = "./data/celebrities.txt"
embeddings_dir = "./data/embeddings/"
models_dir = "./models/"
celebs = []
with open(celebrities_file) as f:
    for line in f:
        celebs.append(line.strip())
embeddor = FaceEmbedder("./models/faceEmbeddings.npy")
index = nmslib.init(method='hnsw', space='cosinesimil')
mapping = {}

count_celebs = 0
for celeb in celebs:
    celeb_faces_dir = os.path.join(faces_dir, celeb.lower().replace(" ", "_"))
    if not os.path.isdir(celeb_faces_dir):
        print(f"could not find photos of {celeb}, skipping")
        continue
    mean_embedding = np.zeros([1,512])
    count_faces = 0
    for file in os.listdir(celeb_faces_dir):
        try:
            img = cv2.imread(os.path.join(celeb_faces_dir, file))
            embedding = embeddor.model.predict(img[np.newaxis,:,:,:])
            mean_embedding += embedding
            count_faces += 1
        except Exception as e:
            print(e)
            continue
    mean_embedding /= count_faces
    index.addDataPoint(id = count_celebs, data = mean_embedding)
    mapping[count_celebs] = celeb
    np.save(os.path.join(embeddings_dir, celeb.lower().replace(" ", "_") + ".npy"), mean_embedding)
    count_celebs += 1

index.createIndex(print_progress=True)

with open(os.path.join(models_dir, "celebrities_mapping.json"), "w+") as out:
    json.dump(mapping, out)
index.saveIndex(filename = os.path.join(models_dir, "index.nms"))
