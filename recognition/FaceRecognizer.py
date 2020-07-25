import json
import nmslib
import numpy as np
from recognition.face_embedder_network import FaceEmbedder

class FaceRecognizer:
    def __init__(self, index_path, mapping_path, embedder_weights_path):
        with open(mapping_path) as f:
            self.mapping = json.load(f)
        self.index = nmslib.init(method='hnsw', space='cosinesimil')
        self.index.loadIndex(index_path)
        self.embedder = FaceEmbedder(embedder_weights_path)

    def run(self, face, threshold):
        embedding = self.embedder.model.predict(face[np.newaxis,:,:,:])
        indexes, distances = self.index.knnQuery(embedding, 1)
        if distances[0] < threshold:
            print(self.mapping[str(indexes[0])])
            print(distances[0])
            return self.mapping[str(indexes[0])]
        else:
            return None
