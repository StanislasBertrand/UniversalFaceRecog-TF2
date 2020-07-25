# Tensorflow2 universal face recognition framework

Face recognition framework with 80 pretrained celebrities to recognize.  
Easily add any person or celebrity to be recognized !

example output :
![testing on a leaders_photo](sample-results/leaders_output.jpg)   
Check out more examples at the bottom of the page !

*****
### Installation
run
```angular2
python setup.py install
```
If you dont have a GPU, change the line tensoflow-gpu==2.0.1 to tensorflow==2.0.1 from the setup.py file.

*****
### Run recognition
First, download the pretrained models and list of available celebrities from [DropBox](https://www.dropbox.com/sh/34vd1zmtsdch8ln/AABfP5l3ITZo5jzgvZaiZZ3ja?dl=0),
and place the models in the ./models folder

Run : 
```angular2
python recognition.py --sample_img="./sample-images/leaders.jpg" --save_destination="./sample-results/leaders_output.jpg"
```

*****
### Add your own celebrities
In order to add more people to the recognition database, follow the [tutorial](train/tutorial.md)

*****
### Structure
**retinaface**   
This folder contains the code for face detection and alignment. It is based on [this original paper](https://arxiv.org/pdf/1905.00641.pdf), and [this implementation](https://github.com/StanislasBertrand/RetinaFace-tf2).   

**recognition**  
This folder contains the code that takes as input a face and recognizes a person, in two steps:
* Deep learning based face embedding extraction. This neural network take as input a face, returns a feature vector of size 512.
* nearest neighbor search to find a matching embedding of a person in our database. Takes a input an embedding vector, and tries to look for similar embedding vectors in our people database. If a vector in our database is close enough, then we have found that person.

**train**  
This folder contains scripts to add any person to the face recognition database

*****
### Example outputs
![testing on heat](sample-results/heat_output.jpg)
![testing on emmas](sample-results/emmas_output.jpg)
![testing on tarantino](sample-results/hollywood_output.jpg)
![testing on pulp](sample-results/pulp_output.jpg)
![testing on batman](sample-results/batman_output.jpg)

*****
### TODO
* ~~add alignment to mtcnn~~
* ~~convert embedding extractor to tf2~~
* ~~convert RetinaFace to tf2 for better face detection~~
* ~~add nmslib knn search for face rec~~
* ~~add test code for face detection + recognition~~ 
* ~~create first index of people face embeddings~~
* ~~automate pipeline of adding new persons to recognition database~~
* Add more people to the original pretrained celebrities dataset
* Increase accuracy on low res images by augmenting train images
* Finish airflow pipeline for database creation

*****
### Acknowledgements
Most of this work is based on the work of [Insightface](https://github.com/deepinsight/insightface#512-d-feature-embedding) and [MMdnn](https://github.com/microsoft/MMdnn)
If you use this repo, please reference the original face detection work :

```  
@inproceedings{Deng2020CVPR,
title = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
author = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
booktitle = {CVPR},
year = {2020}
}
```

