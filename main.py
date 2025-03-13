import pandas as pd
import ao_core as ao
import ao_arch as ar
import embedding_bucketing.embedding_model_test as em
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import openai_key

# Initialize embedding bucketing
em.config(openai_key)
cache_file = "cache_genre.json"
start_Genre = ["Drama", "Comedy", "Action", "romance", "documentary"]
max_distance = 0.55
cache, Genre = em.init(cache_file, start_Genre)

movies = pd.read_csv("data/movies.csv")
movie_tags = pd.read_csv("data/tags.csv")
ratings = pd.read_csv("data/ratings.csv")

# Merge ratings and movies
train_data = pd.merge(ratings, movies, on="movieId")
print(train_data)

Arch = ar.Arch(arch_i=[6, 15], arch_z=[3], arch_c=[])

agent = ao.Agent(Arch, notes="")

train_data_sample = train_data.sample(n=5000, random_state=42)
test_data_sample = train_data.drop(train_data_sample.index).sample(n=40, random_state=42)

train_data_list = []
ratings_train = []
test_data_list = []
ratings_test = []

def encode_genre(genre):
    binary = [0,0,0,0,0, 0]
    genres = ["Drama", "Comedy", "Action", "romance", "documentary", "Thriller"]
    try:
        indx = genres.index(genre)
        binary[indx] = 1
    except Exception as e:
        pass
    print("encoding: ", genre, "as", binary)
    return binary

for idx, row in train_data_sample.iterrows():
    genres_list = row['genres'].split('|')
    primary_genre = genres_list[0]

    genre_encoding = encode_genre(primary_genre)


    rating = [int(bit) for bit in format(int(row['rating']), "03b")]

    uid = row["userId"]
    userId = list(format(int(uid), "015b")) 
    
    train_row = userId + list(genre_encoding)
    train_data_list.append(train_row)
    ratings_train.append(rating)

agent.next_state_batch(INPUT=train_data_list, LABEL=ratings_train)


for idx, row in test_data_sample.iterrows():
    genres_list = row['genres'].split('|')
    primary_genre = genres_list[0]
    
    genre_encoding = encode_genre(primary_genre)

    uid = row["userId"]
    userId = list(format(int(uid), "015b")) 
    
    test_row = userId + list(genre_encoding)
    test_data_list.append(test_row)
    ratings_test.append(row['rating'])

predictions = []

for test_row in test_data_list:
    predicted_binary = agent.next_state(test_row) 
    print("input : ", test_row, "output: ", predicted_binary)
    predicted_rating = int("".join(map(str, predicted_binary)), 2)
    predictions.append(predicted_rating)

# Evaluate model performance
mse = mean_squared_error(ratings_test, predictions)
mae = mean_absolute_error(ratings_test, predictions)
r2 = r2_score(ratings_test, predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")
