import os
import ast
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer

data_path = 'cleaned.csv'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

WEIGHTS = {
    'genres': 1.0,
    'categories': 1.5,
    'tags': 2.0,
    'platforms': 0.8
}

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['genres','categories','tags','windows','linux','mac']).reset_index(drop=True)

    df['genres_list']     = df['genres'].apply(ast.literal_eval)
    df['categories_list'] = df['categories'].apply(ast.literal_eval)
    df['tags_list']       = df['tags'].apply(lambda t: list(ast.literal_eval(t).keys()))
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

features = np.hstack([G, C, T, P])
scaler = StandardScaler().fit(features)
X_scaled = scaler.transform(features)

pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_scaled)

knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
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
print('Modelos avaliados e salvos em', model_dir)
