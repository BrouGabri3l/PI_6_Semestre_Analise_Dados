import os
import ast
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import average_precision_score, ndcg_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction import FeatureHasher
from utils import WEIGHTS
import matplotlib.pyplot as plt
data_path = 'cleaned.csv'
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    df['genres_list']     = df['genres'].apply(ast.literal_eval)
    df['categories_list'] = df['categories'].apply(ast.literal_eval)
    df['tags'] = df['tags'].apply(lambda t:list(ast.literal_eval(t)))
    df['publisher_list']  = df['publishers'].apply(ast.literal_eval)

    print(df['publisher_list'])
    return df

df = load_data(data_path)

mlb_genres     = MultiLabelBinarizer()
G = mlb_genres.fit_transform(df['genres_list']) * WEIGHTS['genres']

mlb_categories = MultiLabelBinarizer()
C = mlb_categories.fit_transform(df['categories_list']) * WEIGHTS['categories']

mlb_tags       = MultiLabelBinarizer()
T_binary = mlb_tags.fit_transform(df['tags'])


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


def precision_at_k(knn, X, categories_list, k=10):
    neigh = knn.kneighbors(X, n_neighbors=k, return_distance=False)
    precisions = []

    for i, nbrs in enumerate(neigh):
        relevant = 0
        for n in nbrs[1:]:
            if set(categories_list[i]).intersection(categories_list[n]):
                relevant += 1
        precisions.append(relevant / (k - 1))

    return np.mean(precisions)

def diversity(knn, X, k=10):
    neigh = knn.kneighbors(X, n_neighbors=k, return_distance=False)
    diversity_scores = []

    for nbrs in neigh:
        distances = []
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                dist = np.linalg.norm(X[nbrs[i]] - X[nbrs[j]])
                distances.append(dist)
        diversity_scores.append(np.mean(distances))

    return np.mean(diversity_scores)

def coverage(knn, X, k=10):
    neigh = knn.kneighbors(X, n_neighbors=k, return_distance=False)
    recommended = set(neigh.flatten())
    return len(recommended) / X.shape[0]

def evaluate_loocv(X, labels_list):
    knn_eval = NearestNeighbors(n_neighbors=2, metric='cosine', algorithm='brute')
    knn_eval.fit(X)
    neigh = knn_eval.kneighbors(X, return_distance=False)
    hits = 0
    for i, nbrs in enumerate(neigh):
        if set(labels_list[i]).intersection(labels_list[nbrs[1]]):
            hits += 1
    return hits / len(neigh)

def compute_ndcg(knn, X, labels_list, k=10):
    neigh = knn.kneighbors(X, n_neighbors=k, return_distance=False)
    ndcgs = []

    for i, nbrs in enumerate(neigh):
        y_true = np.array([1 if set(labels_list[i]).intersection(labels_list[n]) else 0 for n in nbrs])
        y_score = np.linspace(1, 0, num=k)
        ndcg = ndcg_score([y_true], [y_score])
        ndcgs.append(ndcg)

    return np.mean(ndcgs)

def compute_map(knn, X, labels_list, k=10):
    neigh = knn.kneighbors(X, n_neighbors=k, return_distance=False)
    maps = []

    for i, nbrs in enumerate(neigh):
        y_true = np.array([1 if set(labels_list[i]).intersection(labels_list[n]) else 0 for n in nbrs])
        y_score = np.linspace(1, 0, num=k)
        ap = average_precision_score(y_true, y_score)
        maps.append(ap)

    return np.mean(maps)

def has_positive(label_row):
    return bool(label_row) and len(label_row) > 0

def run_all_tests(knn, X, df, k=10):
    results = {}

    for label_name in ['categories_list', 'genres_list', 'publisher_list', 'tags']:
        print(f"\nTesting with: {label_name}")
        labels_list = df[label_name]
        if not any(has_positive(row) for row in labels_list):
            print(f"Atenção: Nenhum positivo em {label_name}! Pulando métricas.")
            results[label_name] = None
            continue
        precision = precision_at_k(knn, X, labels_list, k)
        print(f'Precision@{k}: {precision:.3f}')
        div = diversity(knn, X, k)
        print(f'Diversity: {div:.3f}')
        cov = coverage(knn, X, k)
        print(f'Coverage: {cov:.3f}')
        loocv = evaluate_loocv(X, labels_list)
        print(f'LOO-CV Match Rate: {loocv:.3f}')
        ndcg = compute_ndcg(knn, X, labels_list, k)
        print(f'NDCG@{k}: {ndcg:.3f}')
        map_score = compute_map(knn, X, labels_list, k)
        print(f'MAP@{k}: {map_score:.3f}')

        results[label_name] = {
            'Precision': precision,
            'Diversity': div,
            'Coverage': cov,
            'LOO-CV': loocv,
            'NDCG': ndcg,
            'MAP': map_score
        }

    return results

def plot_radar(results, k=10):
    labels = list(results.keys())
    metrics = ['Precision', 'Diversity', 'Coverage', 'LOO-CV', 'NDCG', 'MAP']

    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for label_name in labels:
        values = [results[label_name][metric] for metric in metrics]
        values += values[:1]
        ax.plot(angles, values, label=label_name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)

    plt.title(f'KNN Evaluation Radar (k={k})')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.show()

print(df.columns)
results = run_all_tests(knn, X_pca, df, k=10)

# Exibir resultados
for label, metrics in results.items():
    print(f"\nResults for {label}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.3f}")

# Gerar gráfico radar comparativo
test_plot = plot_radar(results, k=10)

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
