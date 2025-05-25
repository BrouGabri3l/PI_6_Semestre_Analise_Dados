import os
import ast
import pandas as pd
import numpy as np
import pickle
import io
import boto3
from sklearn.metrics.pairwise import cosine_distances
from utils import WEIGHTS
from dotenv import load_dotenv
from sqlalchemy import create_engine

class Recommender:
    def __init__(self, data_path='cleaned.csv', model_dir='models', top_k=5):
        load_dotenv() 

        # Montar a string de conexão
        self.db_url = (
            f"postgresql+psycopg2://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}"
            f"@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
        )

        # Criar engine
        self.engine = create_engine(self.db_url)
        # Conectar ao banco e puxar os dados direto pro DataFrame
       
        # chunks = pd.read_sql('SELECT * FROM public."Games"', engine, chunksize=1000
        #                             #   converters={"genres_list": ast.literal_eval,"categories_list": ast.literal_eval,"tags_list": ast.literal_eval, "publisher_list": ast.literal_eval}
        #                                 )
        # dfs = []
        # row_count = pd.read_sql('SELECT COUNT(*) AS total FROM public."Games"', engine)
        # total_rows = row_count.iloc[0]['total']
        # step = 10000
        # for offset in range(0, total_rows, step):
        #     sql = f'SELECT * FROM public."Games" ORDER BY game_id LIMIT {step} OFFSET {offset} '
        #     chunk = pd.read_sql(sql, engine)
        #     dfs.append(chunk)

        # database_df = pd.concat(dfs, ignore_index=True)
        # print(database_df.head())
        mapeamento = {
            "mlb_genres.pkl":        "mlb_genres",
            "mlb_categories.pkl":    "mlb_categories",
            "mlb_tags.pkl":          "mlb_tags",
            "tfidf_tags.pkl":        "tfidf",
            "scaler.pkl":            "scaler",
            "pca.pkl":               "pca",
            "knn_pca_tags.pkl":      "knn",
            "HASher.pkl":           "publisher_hasher"
        }
        # df = database_df.drop(columns=["header_image", "short_description"])
        # self.df = df
        # self.database_df = database_df
        self.top_k = top_k


        # Carregamento de arquivos local
        # self.mlb_genres     = pickle.load(open(os.path.join(model_dir,'mlb_genres.pkl'),'rb'))
        # self.mlb_categories = pickle.load(open(os.path.join(model_dir,'mlb_categories.pkl'),'rb'))
        # self.mlb_tags       = pickle.load(open(os.path.join(model_dir,'mlb_tags.pkl'),'rb'))
        # self.tfidf          = pickle.load(open(os.path.join(model_dir,'tfidf_tags.pkl'),'rb'))
        # self.scaler         = pickle.load(open(os.path.join(model_dir,'scaler.pkl'),'rb'))
        # self.pca            = pickle.load(open(os.path.join(model_dir,'pca.pkl'),'rb'))
        # self.knn            = pickle.load(open(os.path.join(model_dir,'knn_pca_tags.pkl'),'rb'))
        # self.publisher_hasher = pickle.load(open(os.path.join(model_dir,'HASher .pkl'),'rb'))

        # Carregamento de arquivos produção

        # AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
        # AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
        # AWS_REGION = os.getenv("AWS_REGION")
        s3_client = boto3.client(
            's3'
            # aws_access_key_id=AWS_KEY,
            # aws_secret_access_key=AWS_SECRET,
            # region_name=AWS_REGION
        )
        resposta = s3_client.list_objects_v2(Bucket="pi-6-iplay", Prefix="models/")
        arquivos_s3 = {obj['Key'] for obj in resposta.get('Contents', [])}

        for nome_arquivo, nome_atributo in mapeamento.items():
            key_completa = f"{'models/'}{nome_arquivo}" if not nome_arquivo.startswith("models/") else nome_arquivo
            if key_completa not in arquivos_s3:
                print(f"[AVISO] Arquivo '{key_completa}' não encontrado no S3.")
                setattr(self, nome_atributo, None)
                continue

            buffer = io.BytesIO()
            s3_client.download_fileobj("pi-6-iplay", key_completa, buffer)
            buffer.seek(0)
            obj = pickle.load(buffer)
            setattr(self, nome_atributo, obj)
            print(f"Atributo '{nome_atributo}' carregado do arquivo '{key_completa}'.")

        # Binarização
        # G = self.mlb_genres.transform(df['genres_list']) * WEIGHTS['genres']
        # C = self.mlb_categories.transform(df['categories_list']) * WEIGHTS['categories']
        # Tb = self.mlb_tags.transform(df['tags_list'])
        # T = self.tfidf.transform(Tb).toarray() * WEIGHTS['tags']
        # P = df[['windows','linux','mac']].astype(int).values * WEIGHTS['platforms']
        # Pub = self.publisher_hasher.transform(df['publisher_list']).toarray()
        # features = np.hstack([G,
        #                        C,
        #                        T,
        #                        P,
        #                        Pub])
        # X_scaled = self.scaler.transform(features)
        # self.X_pca = self.pca.transform(X_scaled)

    def recommend(self, user_genres, user_categories, user_played_ids, user_platforms, played_tags, user_publishers):
        ug = self.mlb_genres.transform([user_genres]) * WEIGHTS['genres']
        uc = self.mlb_categories.transform([user_categories]) * WEIGHTS['categories']
        up = self.publisher_hasher.transform([user_publishers]).toarray()
        for pid in user_played_ids:
            row = pd.read_sql(f'SELECT game_id AS total FROM public."Games" WHERE game_id = {pid}', self.engine)
            if not row.empty:
                played_tags.extend(row.iloc[0]['tags_list'])
        unique_tags = list(set(played_tags)) if played_tags else []
        Tb_u = self.mlb_tags.transform([unique_tags])
        T_u = self.tfidf.transform(Tb_u).toarray() * WEIGHTS['tags']

        P_u = np.array([[int(p.lower() in [u.lower() for u in user_platforms])
                         for p in ['windows','linux','mac']]]) * WEIGHTS['platforms']
        
        vec = np.hstack([ug, uc, T_u, P_u, up])
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