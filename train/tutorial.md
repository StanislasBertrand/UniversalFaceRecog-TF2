# How to add any person to the face recognition database

### Step 1 : gather images
in order to gather images of the person you wish to detect you can either : 
* Get them from a personnal source, and place the in a data/images/raw_images folder
* For a famous enough peron, use a webcrawler. The download_images.py script does this job.  
First put the list of people you want to add in a celebrities.txt file (see data/celebrities_pretrinaed.txt for an example). Then, in a python shell, run something like :
```angular2
download_images("data/celebrities.txt", "data/images/raw_images/")
```
You should en up in the raw_images dir with 1 subdir for each person you want to add. 

### Step 2 : extract faces from these raw images
In this step, we extract all faces from all the raw photos and save them in a separate dir.   
From the extract_faces.py script, run the method : 
```angular2
extract_faces("data/celebrities.txt", "data/images/raw_images/", "data/images/faces/", "models/retinafaceweights.npy")
```
Once again, in the end the faces dir should contain one subdir for each person, containing faces associated with this person.

### Step 3 : filter out faces that arent the right ones
At this point, each faces/celebrity subdir contains mostly faces or ouf celebrity, but also some faces of other people, or some very blurry faces.   
In this step, we remove these noisy faces we dont want.   
In order to do this, for each face/celebrity subdir, and for each face in the subdir, we calculate the face embedding with our embedding extractor. We then calculate the mean embedding of all the faces of a subdir, and only keep faces that are close enough (according to cosine similarity) to our mean embedding.
If there are not too many noisy images in a subdir, this only keeps the faces of our celebrity.   
From the filter_faces.py script, run the method : 
```angular2
filter_faces("data/celebrities.txt", "data/images/faces/", "data/images/faces_filtered/", "models/faceEmbeddings.npy", 0.5)
```
If after this step you still have noisy images in the faces_filtered dir, try changing the cosine similarity threshold

### Step 4 : Build final nearest neighbor index
In this last step, we build an index that contains, for each celebrity, a the mean embedding of all this celebrities face embeddings.  
From the build_index.py script, run : 
```angular2
build_index("data/celebrities.txt", "data/images/faces_filtered/", "data/embeddings/", "models/")
```
We now have a database of celebrities mean face embeddings !  
At inference, when we detect a face on an image, we can calculate the embedding of this face and search our database with nearest neighbor for similar embeddings. If we find an embedding similar enough... then this face belongs to that person !