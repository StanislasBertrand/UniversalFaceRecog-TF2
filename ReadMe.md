# [WIP] Tensorflow based universal face recognition framework

This repository aims at providing a tensorflow2 implementation of a face detection and recognition framework pretrained to detect the faces of many celebrities along with a pipeline to easily add new people to be detected by it.  

### Credit
As of now, it is inspired by the works of [Insightface](https://github.com/deepinsight/insightface#512-d-feature-embedding), [mtcnn](https://github.com/ipazc/mtcnn), and [MMdnn](https://github.com/microsoft/MMdnn)

### TODO
* ~~add alignment to mtcnn~~
* ~~convert embedding extractor to tf2~~
* convert RetinaFace to tf2 for better face detection
* add nmslib knn search for face rec
* add test code for face detection + recognition 
* create first index of people face embeddings
* .....

### Models
Download from [DropBox](https://www.dropbox.com/sh/34vd1zmtsdch8ln/AABfP5l3ITZo5jzgvZaiZZ3ja?dl=0) and place in models folder