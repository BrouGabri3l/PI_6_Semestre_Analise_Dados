import os
import ast
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import FeatureHasher
from utils import WEIGHTS
data_path = 'cleaned.csv'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    df['genres_list']     = df['genres'].apply(ast.literal_eval)
    df['categories_list'] = df['categories'].apply(ast.literal_eval)
    df['tags_list'] = df['tags_list'].apply(lambda t:list(ast.literal_eval(t)))
    df['publisher_list']  = df['publishers'].apply(ast.literal_eval)

    print(df['publisher_list'])
    return df

df = load_data(data_path)

mlb_genres     = MultiLabelBinarizer()
G = mlb_genres.fit_transform(df['genres_list']) * WEIGHTS['genres']

mlb_categories = MultiLabelBinarizer()
C = mlb_categories.fit_transform(df['categories_list']) * WEIGHTS['categories']

mlb_tags       = MultiLabelBinarizer()
T_binary = mlb_tags.fit_transform(df['tags_list'])

tfidf = TfidfTransformer()
T = tfidf.fit_transform(T_binary).toarray() * WEIGHTS['tags']
P = df[['windows','linux','mac']].astype(int).values * WEIGHTS['platforms']

HASher   = FeatureHasher(n_features=128, input_type='string')
Pub_hash = HASher.transform(df['publisher_list']).toarray()  # gera (n_samples,128)
Pub_hash = Pub_hash
features = np.hstack([G, C, T, P, Pub_hash])
scaler = StandardScaler().fit(features)
X_scaled = scaler.transform(features)

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
knn.fit(X_pca)


def evaluate_loocv(X, categories_list):
    knn_eval = NearestNeighbors(n_neighbors=2, metric='cosine', algorithm='brute')
    knn_eval.fit(X)
    neigh = knn_eval.kneighbors(X, return_distance=False)
    hits = 0
    for i, nbrs in enumerate(neigh):
        if set(categories_list[i]).intersection(categories_list[nbrs[1]]):
            hits += 1
    return hits / len(neigh)

score = evaluate_loocv(X_pca, df['categories_list'])
print(f'LOO-CV category match rate (PCA+Tags): {score:.3f}')

pickle.dump(mlb_genres,     open(os.path.join(model_dir,'mlb_genres.pkl'),'wb'))
pickle.dump(mlb_categories, open(os.path.join(model_dir,'mlb_categories.pkl'),'wb'))
pickle.dump(mlb_tags,       open(os.path.join(model_dir,'mlb_tags.pkl'),'wb'))
pickle.dump(scaler,         open(os.path.join(model_dir,'scaler.pkl'),'wb'))
pickle.dump(tfidf,          open(os.path.join(model_dir,'tfidf_tags.pkl'),'wb'))
pickle.dump(pca,            open(os.path.join(model_dir,'pca.pkl'),'wb'))
pickle.dump(knn,            open(os.path.join(model_dir,'knn_pca_tags.pkl'),'wb'))
pickle.dump(HASher ,  open(os.path.join(model_dir,'HASher .pkl'),'wb'))
from dotenv import load_dotenv
import boto3
load_dotenv() 

AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_KEY,
    aws_secret_access_key=AWS_SECRET,
    region_name=AWS_REGION
)
s3_client.upload_file(os.path.join(model_dir,'mlb_genres.pkl'), "pi-6-iplay",  os.path.join(model_dir,'mlb_genres.pkl'))
s3_client.upload_file(os.path.join(model_dir,'mlb_categories.pkl'), "pi-6-iplay",  os.path.join(model_dir,'mlb_categories.pkl'))
s3_client.upload_file(os.path.join(model_dir,'mlb_tags.pkl'), "pi-6-iplay",  os.path.join(model_dir,'mlb_tags.pkl'))
s3_client.upload_file(os.path.join(model_dir,'scaler.pkl'), "pi-6-iplay",  os.path.join(model_dir,'scaler.pkl'))
s3_client.upload_file(os.path.join(model_dir,'tfidf_tags.pkl'), "pi-6-iplay",  os.path.join(model_dir,'tfidf_tags.pkl'))
s3_client.upload_file(os.path.join(model_dir,'pca.pkl'), "pi-6-iplay",  os.path.join(model_dir,'pca.pkl'))
s3_client.upload_file(os.path.join(model_dir,'knn_pca_tags.pkl'), "pi-6-iplay",  os.path.join(model_dir,'knn_pca_tags.pkl'))
s3_client.upload_file(os.path.join(model_dir,'HASher .pkl'), "pi-6-iplay", os.path.join(model_dir,'HASher.pkl'))

print('Modelos avaliados e salvos em', model_dir)
