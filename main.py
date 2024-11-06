from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

peliculas = pd.read_parquet("./data/dataset_movies.parquet")
app = FastAPI()

@app.get("/mes")
def cantidad_filmaciones_mes(mes: str):
    meses = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
    'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
    'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    numMes = meses[mes.lower()]
    cant = peliculas[peliculas['release_date'].dt.month == numMes]
    return f"{len(cant)} cantidad de peliculas fueron estrenadas en el mes de {mes}"

@app.get("/dia")
def cantidad_filmaciones_dia(dia: str):
    dias = {
    'lunes': 0, 'martes': 1, 'miercoles': 2, 'jueves': 3, 'viernes': 4, 'sabado': 5, 'domingo': 6
    }
    numDia = dias[dia.lower()]
    cant = peliculas[peliculas['release_date'].dt.weekday == numDia]
    return f"{len(cant)} cantidad de peliculas fueron estrenadas en los dias {dia}"

@app.get("/score")
def score_titulo(titulo_de_la_filmacion: str):
    row = peliculas[peliculas['title'] == titulo_de_la_filmacion]
    if row.empty:
        return {"error": "Película no encontrada"}
    year = row['release_year'].values[0]
    score = row['popularity'].values[0]
    return f"la pelicula {titulo_de_la_filmacion} fue estrenada en el año {year} con un score/popularidad de {round(score,2)}"

@app.get("/votos")
def votos_titulo(titulo_de_la_filmacion: str):
    row = peliculas[peliculas['title'] == titulo_de_la_filmacion]
    if row.empty:
        return {"error": "Película no encontrada"}
    year = row['release_year'].values[0]
    votos = row['vote_count'].values[0]
    average = row['vote_average'].values[0]
    if votos >= 2000:
        return f"la pelicula {titulo_de_la_filmacion} fue estrenada en el año {year}. La misma cuenta con un total de {int(votos)} valoraciones, con un promedio de {round(average,2)}"
    else:
        return f"la pelicula {titulo_de_la_filmacion} no cuenta con al menos 2000 valoraciones"

# @app.get("/actor")
# def get_actor(nombreActor: str):
#     return f"El actor {nombreActor} ha participado de xxxx cantidad de filmaciones, el mismo ha conseguido un retorno de xxxx con un promedio de xxxx por filmacion"

# @app.get("/director")
# def get_director(nombeDirector: str):
#     return f"{nombeDirector}"

vectorizer = TfidfVectorizer(stop_words='english')
overviews_tokenizados = vectorizer.fit_transform(peliculas['overview']) #revisar si tengo que recortar el size del dataset
@app.get("/recomendacion")
def recomendacion(titulo: str):
    pelicula = peliculas[peliculas['title'] == titulo]
    if pelicula.empty:
        return {"error": "Película no encontrada"}
    similitudes = cosine_similarity(overviews_tokenizados[peliculas.index == pelicula.index[0]], overviews_tokenizados)
    #Se obtienen las 5 peliculas mas similares 
    similares = similitudes[0].argsort()[-6:-1][::-1] 
    return peliculas.iloc[similares]['title'].tolist()