
from flask import Flask, request, jsonify
from flasgger import Swagger
from load_recommender import load_recommender

reco = load_recommender()
app = Flask(__name__)
Swagger(app)

@app.route('/ping', methods=['GET'])
def ping():
  return "Pong"

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
            suggested_games_ids:
              type: array
              items:
                type: integer
            played_games_ids:
              type: array
              items:
                type: integer
            played_games:
              type: array
              items:
                type: string
            operational_systems:
              type: array
              items:
                type: string
            game_modes:
              type: array
              items:
                type: string
            publishers:
              type: array
              items:
                type: string
            game_styles:
              type: array
              items:
                type: string 
            camera_perspective:
              type: array
              items:
                type: string
          required: [genres, categories, played_games_ids, operational_systems]
    responses:
      200:
        description: Lista de appids e detalhes dos jogos recomendados
    """
    data = request.get_json(force=True)
    # Obter IDs recomendados
    rec_ids = reco.recommend(
        user_genres=data.get('genres', []),
        user_categories=data.get('categories', []),
        user_played_ids=data.get('played_games_ids', []),
        # favorite_games=data.get('favorite_games')
        user_platforms=data.get('operational_systems', []),
        played_tags=data.get('game_styles', []) + data.get('camera_perspective', []),
        user_publishers=data.get("publishers", [])
        # user_game_modes=data.get('game_modes', []),
        # user_game_styles=data.get('game_styles', []),
        # user_camera_perspective=data.get('camera_perspective', [])

    )
    # Filtrar detalhes diretamente do dataset
    rec_rows = reco.database_df[reco.database_df['appid'].isin(rec_ids)]
    # Construir lista de dicionários com campos relevantes
    details = []
    for _, row in rec_rows.iterrows():
        details.append({
            'appid': int(row.appid),
            'name': row["name"],
            'genres': row.genres_list,
            'categories': row.categories_list,
            'operational_systems':{ **{col: bool(row[col]) for col in ['windows','linux','mac'] if col in row}}
        }
)
    return jsonify({
        'recommendation_ids': rec_ids,
        'recommendations': details
    })

@app.route('/games', methods=['GET'])
def list_games():
    """
    Lista todos os jogos paginados.
    ---
    tags:
      - games
    parameters:
      - in: query
        name: page
        type: integer
        required: false
        default: 1
        description: Página a ser retornada (1-indexed)
      - in: query
        name: per_page
        type: integer
        required: false
        default: 20
        description: Número de itens por página
    responses:
      200:
        description: Lista paginada de jogos
        schema:
          type: object
          properties:
            page:
              type: integer
            per_page:
              type: integer
            total:
              type: integer
            total_pages:
              type: integer
            games:
              type: array
              items:
                type: object
                properties:
                  appid:
                    type: integer
                  name:
                    type: string
                  release_date:
                    type: string
                  short_description:
                    type: string
                  header_image:
                    type: string
                  publishers:
                    type: array
                    items: { type: string }
                  supported_languages:
                    type: array
                    items: { type: string }
                  genres:
                    type: array
                    items: { type: string }
                  categories:
                    type: array
                    items: { type: string }
                  operational_systems:
                    type: array
                    items: { type: string }
    """

    page     = max(int(request.args.get('page', 1)), 1)
    per_page = max(int(request.args.get('per_page', 20)), 1)

    df = reco.database_df
    total = len(df)
    total_pages = (total + per_page - 1) // per_page

    start = (page - 1) * per_page
    end   = start + per_page

    subset = df.iloc[start:end]
    games = []

    for _, row in subset.iterrows():
        games.append({
            'appid':      int(row.appid),
            'name':       row["name"],
            'short_description': row.short_description,
            'header_image': row.header_image,
            'publishers': row.publishers,
            'supported_languages': row.supported_languages,
            'genres':     row.genres_list,
            'categories': row.categories_list,
             'operational_systems':{ **{col: bool(row[col]) for col in ['windows','linux','mac'] if col in row}}
        })

    return jsonify({
        'page':        page,
        'per_page':    per_page,
        'total':       total,
        'total_pages': total_pages,
        'games':       games
    })

@app.route('/games/<int:game_id>', methods=['GET'])
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
    df = reco.database_df
    row = df[df['appid'] == game_id]

    if row.empty:
        return jsonify({'error': 'Jogo não encontrado'}), 404
    g = row.iloc[0]
    
    return jsonify({
        'appid': int(g["appid"]),
        'name': g["name"],
        'short_description': g['short_description'],
        'header_image': g['header_image'],
        'publishers': g['publishers'],
        'supported_languages': g['supported_languages'],
        'genres': g["genres_list"],
        'categories': g["categories_list"],
        'operational_systems': { **{col: bool(g[col]) for col in ['windows','linux','mac'] if col in g}}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)