import os
import cv2
from retinaface.FaceDetector import FaceDetector
from retinaface.alignment import extract_aligned_faces


def extract_faces(celebs_file, photos_dir, faces_dir, weights_path):
    celebs = []
    with open(celebs_file) as f:
        for line in f:
            celebs.append(line.strip())

    detector = FaceDetector(weights_path, False, 0.4)

    for celeb in celebs:
        celeb_photos_dir = os.path.join(photos_dir, celeb.lower().replace(" ", "_"))
        celeb_faces_dir = os.path.join(faces_dir, celeb.lower().replace(" ", "_"))
        if not os.path.isdir(celeb_photos_dir):
            print(f"could not find photos of {celeb}, skipping")
            continue
        if not os.path.isdir(celeb_faces_dir):
            os.mkdir(celeb_faces_dir)
        else:
            print(f"faces photos of {celeb} laready exist at {celeb_faces_dir}, skipping")
            continue
        count_faces = 0
        for file in os.listdir(celeb_photos_dir):
            try:
                img = cv2.imread(os.path.join(celeb_photos_dir, file))
                faces, landmarks = detector.detect(img, 0.9)
                aligned_faces = extract_aligned_faces(img, landmarks)
                for aligned_face in aligned_faces:
                    cv2.imwrite(os.path.join(celeb_faces_dir, str(count_faces) + ".jpg"), aligned_face)
                    count_faces += 1
            except Exception as e:
                print(e)
                continue