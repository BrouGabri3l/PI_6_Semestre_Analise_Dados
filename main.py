
from flask import Flask, request, jsonify
from flasgger import Swagger
from load_recommender import load_recommender

reco = load_recommender()
app = Flask(__name__)
Swagger(app)

@app.route('/recommend', methods=['POST'])
def recommend_route():
    """
    Recomendação de jogos por IDs e detalhes via distâncias ponderadas.
    ---
    tags:
      - recommendations
    parameters:
      - in: body
        name: body
        schema:
          type: object
          properties:
            genres:
              type: array
              items:
                type: string
            categories:
              type: array
              items:
                type: string
            played:
              type: array
              items:
                type: integer
            platforms:
              type: array
              items:
                type: string
          required: [genres, categories, played, platforms]
    responses:
      200:
        description: Lista de appids e detalhes dos jogos recomendados
    """
    data = request.get_json(force=True)
    # Obter IDs recomendados
    rec_ids = reco.recommend(
        user_genres=data.get('genres', []),
        user_categories=data.get('categories', []),
        user_played_ids=data.get('played', []),
        user_platforms=data.get('platforms', [])
    )
    # Filtrar detalhes diretamente do dataset
    rec_rows = reco.df[reco.df['appid'].isin(rec_ids)]
    # Construir lista de dicionários com campos relevantes
    details = []
    for _, row in rec_rows.iterrows():
        details.append({
            'appid': int(row.appid),
            'name': row["name"],
            'genres': row.genres_list,
            'categories': row.categories_list,
            'windows': bool(row.windows),
            'linux': bool(row.linux),
            'mac': bool(row.mac)
        })
    return jsonify({
        'recommendation_ids': rec_ids,
        'recommendations': details
    })


@app.route('/game/<int:game_id>', methods=['GET'])
def game_route(game_id):
    """
    Consulta de jogo por ID.
    ---
    tags:
      - games
    parameters:
      - in: path
        name: game_id
        type: integer
        required: true
    responses:
      200:
        description: Detalhes do jogo
      404:
        description: Não encontrado
    """
    df = reco.df
    row = df[df['appid'] == game_id]

    if row.empty:
        return jsonify({'error': 'Jogo não encontrado'}), 404
    g = row.iloc[0]
  
    return jsonify({
        'appid': int(g["appid"]),
        'name': g["name"],
        'genres': g["genres"],
        'categories': g["categories"],
        **{col: bool(g[col]) for col in ['windows','linux','mac'] if col in g}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)