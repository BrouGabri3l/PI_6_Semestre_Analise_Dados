import os
import ast
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_distances
from build_recommender import WEIGHTS

class Recommender:
    def __init__(self, data_path='cleaned.csv', model_dir='models', top_k=5):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['genres','categories','tags','windows','linux','mac']).reset_index(drop=True)
        df['genres_list']     = df['genres'].apply(ast.literal_eval)
        df['categories_list'] = df['categories'].apply(ast.literal_eval)
        df['tags_list']       = df['tags'].apply(lambda t: list(ast.literal_eval(t).keys()))
        self.df = df
        self.top_k = top_k

        self.mlb_genres     = pickle.load(open(os.path.join(model_dir,'mlb_genres.pkl'),'rb'))
        self.mlb_categories = pickle.load(open(os.path.join(model_dir,'mlb_categories.pkl'),'rb'))
        self.mlb_tags       = pickle.load(open(os.path.join(model_dir,'mlb_tags.pkl'),'rb'))
        self.tfidf          = pickle.load(open(os.path.join(model_dir,'tfidf_tags.pkl'),'rb'))
        self.scaler         = pickle.load(open(os.path.join(model_dir,'scaler.pkl'),'rb'))
        self.pca            = pickle.load(open(os.path.join(model_dir,'pca.pkl'),'rb'))
        self.knn            = pickle.load(open(os.path.join(model_dir,'knn_pca_tags.pkl'),'rb'))

        G = self.mlb_genres.transform(df['genres_list']) * WEIGHTS['genres']
        C = self.mlb_categories.transform(df['categories_list']) * WEIGHTS['categories']
        Tb = self.mlb_tags.transform(df['tags_list'])
        T = self.tfidf.transform(Tb).toarray() * WEIGHTS['tags']
        P = df[['windows','linux','mac']].astype(int).values * WEIGHTS['platforms']
    
        features = np.hstack([G,
                               C,
                               T,
                               P])
        X_scaled = self.scaler.transform(features)
        self.X_pca = self.pca.transform(X_scaled)

    def recommend(self, user_genres, user_categories, user_played_ids, user_platforms):
        ug = self.mlb_genres.transform([user_genres]) * WEIGHTS['genres']
        uc = self.mlb_categories.transform([user_categories]) * WEIGHTS['categories']
        
        played_tags = []
        for pid in user_played_ids:
            row = self.df[self.df['appid'] == pid]
            if not row.empty:
                played_tags.extend(row.iloc[0]['tags_list'])
        unique_tags = list(set(played_tags)) if played_tags else []
        Tb_u = self.mlb_tags.transform([unique_tags])
        T_u = self.tfidf.transform(Tb_u).toarray() * WEIGHTS['tags']

        P_u = np.array([[int(p.lower() in [u.lower() for u in user_platforms])
                         for p in ['windows','linux','mac']]]) * WEIGHTS['platforms']

        vec = np.hstack([ug, uc, T_u, P_u])
        vec_scaled = self.scaler.transform(vec)
        vec_pca = self.pca.transform(vec_scaled)

        dists, idxs = self.knn.kneighbors(vec_pca, n_neighbors=self.top_k + len(user_played_ids) + 5)
        recs = []
        for i in idxs.flatten():
            aid = int(self.df.loc[i,'appid'])
            if aid not in user_played_ids:
                recs.append(aid)
                if len(recs) >= self.top_k:
                    break
        return recs