import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta
from airflow.utils.dates import days_ago
from download_images import download_images
from extract_faces import extract_faces
from filter_faces import filter_faces
from build_index import build_index


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(0),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG('build_new_recognizer_database', default_args=default_args)
repo_dir = "/home/bertrans/projects/personnal/UniversalFaceRecog-TF2/"
celebs_file = os.path.join(repo_dir, "data/celebrities.txt")
raw_images_dir = os.path.join(repo_dir, "data/images/bing_images/")
faces_dir = os.path.join(repo_dir, "data/images/faces/")
filtered_faces_dir = os.path.join(repo_dir, "data/images/faces_filtered/")
embeddings_dir = os.path.join(repo_dir, "data/embeddings/")
models_dir = os.path.join(repo_dir, "models/")

t1 = PythonOperator(dag=dag,
               task_id='download_images',
               provide_context=False,
               python_callable=download_images,
               op_kwargs={'celebs_file': celebs_file, 'save_dir': raw_images_dir})

t2 = PythonOperator(dag=dag,
               task_id='extract_faces',
               provide_context=False,
               python_callable=extract_faces,
               op_kwargs={'celebs_file': celebs_file, 'photos_dir': raw_images_dir, 'faces_dir': faces_dir, 'weights_path': os.path.join(models_dir, "retinafaceweights.npy")})

t3 = PythonOperator(dag=dag,
               task_id='filter_faces',
               provide_context=False,
               python_callable=filter_faces,
               op_kwargs={'celebs_file': celebs_file, 'faces_dir': faces_dir, 'faces_filtered_dir': filtered_faces_dir,'weights_path': os.path.join(models_dir, "faceEmbeddings.npy")})

t4 = PythonOperator(dag=dag,
               task_id='build_index',
               provide_context=False,
               python_callable=build_index,
               op_kwargs={'celebs_file': celebs_file, 'faces_dir': filtered_faces_dir, 'embeddings_dir': embeddings_dir, 'models_dir': models_dir})

t1 >> t2 >> t3 >> t4
