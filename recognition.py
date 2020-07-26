import cv2
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
flags.DEFINE_float('recog_thresh', 0.6, "recognition threshold")

def _main(_argv):
    detector = FaceDetector(FLAGS.detector_weights_path, False, 0.4)
    recognizer = FaceRecognizer(FLAGS.recognizer_index_path, FLAGS.recognizer_mapping_path, FLAGS.embedder_weights_path)

    img = cv2.imread(FLAGS.sample_img)
    faces, landmarks = detector.detect(img, FLAGS.det_thresh)
    faces_aligned = extract_aligned_faces(img, landmarks)
    for n, aligned_face in enumerate(faces_aligned):
        found_person = recognizer.run(aligned_face, FLAGS.recog_thresh)
        if found_person:
            img = cv2.rectangle(img, (int(faces[n][0]), int(faces[n][1])), (int(faces[n][2]), int(faces[n][3])), (255, 0, 0), 1)
            img = cv2.putText(img=img, text=found_person, fontFace= cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, org=(int(faces[n][0]), int(faces[n][1])-5), color=(255, 0, 0), thickness=2)
        else:
            img = cv2.rectangle(img, (int(faces[n][0]), int(faces[n][1])), (int(faces[n][2]), int(faces[n][3])), (0, 0, 255), 1)
    cv2.imwrite(FLAGS.save_destination, img)

if __name__ == '__main__':
    try:
        app.run(_main)
    except SystemExit:
        pass
