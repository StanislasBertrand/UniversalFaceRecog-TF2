import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import cv2
import numpy as np
from retinaface.FaceDetector import FaceDetector
from retinaface.alignment import extract_aligned_faces
from recognition.FaceRecognizer import FaceRecognizer
from absl import app, flags
from absl.flags import FLAGS

flags.DEFINE_string('detector_weights_path', './models/retinafaceweights.npy', 'face detector weights path')
flags.DEFINE_string('embedder_weights_path', './models/faceEmbeddings.npy', 'embedder weights path')
flags.DEFINE_string('recognizer_index_path', './models/index.nms', 'recognizer index path')
flags.DEFINE_string('recognizer_mapping_path', './models/celebrities_mapping.json', 'recognizer mapping path')
flags.DEFINE_string('sample_img', './sample-images/leaders.jpg', 'image to test on')
flags.DEFINE_string('save_destination', 'example_output.jpg', "destination image")
flags.DEFINE_float('det_thresh', 0.9, "detection threshold")
flags.DEFINE_float('recog_thresh', 0.7, "recognition threshold")

def _main(_argv):
    detector = FaceDetector(FLAGS.detector_weights_path, False, 0.4)
    recognizer = FaceRecognizer(FLAGS.recognizer_index_path, FLAGS.recognizer_mapping_path, FLAGS.embedder_weights_path)

    img = cv2.imread(FLAGS.sample_img)
    faces, landmarks = detector.detect(img, FLAGS.det_thresh)
    for n, face in enumerate(faces):
        img = cv2.rectangle(img, (int(face[0]), int(face[1])), (int(face[2]), int(face[3])), (0,0,255), 2)
        landmarks_xs = landmarks[n][:, 0]
        landmarks_ys = landmarks[n][:, 1]
        points = np.concatenate([landmarks_xs, landmarks_ys], axis=0).reshape(1, 10)
        aligned_face = extract_aligned_faces(img, points)[0]
        found_person = recognizer.run(aligned_face, FLAGS.recog_thresh)
        if found_person:
            img = cv2.rectangle(img, (int(face[0]), int(face[1])), (int(face[2]), int(face[3])), (255, 0, 0), 2)
            img = cv2.putText(img=img, text=found_person, fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, org=(int(face[0]), int(face[1])-5), color=(255, 0, 0), thickness=2)
        else:
            img = cv2.rectangle(img, (int(face[0]), int(face[1])), (int(face[2]), int(face[3])), (0, 0, 255), 2)
    cv2.imwrite(FLAGS.save_destination, img)

if __name__ == '__main__':
    try:
        app.run(_main)
    except SystemExit:
        pass
