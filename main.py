import pandas as pd
import ao_core as ao
import ao_arch as ar
import embedding_bucketing.embedding_model_test as em
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import openai_key

# Initialize embedding bucketing
em.config(openai_key)
cache_file = "cache_genre.json"
start_Genre = ["Drama", "Comedy", "Action", "romance", "documentary", "Thriller"]

movies = pd.read_csv("data/movies.csv")
movie_tags = pd.read_csv("data/tags.csv")
ratings = pd.read_csv("data/ratings.csv")

# Merge ratings and movies
train_data = pd.merge(ratings, movies, on="movieId")
print(train_data)

Arch = ar.Arch(arch_i=[6, 15], arch_z=[3], arch_c=[])

agent = ao.Agent(Arch, save_meta = True)
agent.full_conn_compress= True

train_data_sample = train_data.sample(n=100000, random_state=42)
test_data_sample = train_data.drop(train_data_sample.index).sample(n=40, random_state=42)

train_data_list = []
ratings_train = []
test_data_list = []
ratings_test = []

def encode_genre(genres_list):
    genre_encoding = [0] * len(start_Genre)  
    for genre in genres_list:
        if genre in start_Genre:
            genre_encoding[start_Genre.index(genre)] = 1
    return genre_encoding


def encode_rating(rating):
    return [int(bit) for bit in format(int(rating), "03b")]

for idx, row in train_data_sample.iterrows():
    genres_list = row['genres'].split('|')


    genre_encoding = encode_genre(genres_list)


    rating = encode_rating(row['rating'])

    uid = row["userId"]
    userId = list(format(int(uid), "015b")) 
    
    train_row = userId + list(genre_encoding)
    train_data_list.append(train_row)
    ratings_train.append(rating)

agent.next_state_batch(INPUT=train_data_list, LABEL=ratings_train)


for idx, row in test_data_sample.iterrows():
    genres_list = row['genres'].split('|')
    
    genre_encoding = encode_genre(genres_list)

    uid = row["userId"]
    userId = list(format(int(uid), "015b")) 
    
    test_row = userId + list(genre_encoding)
    test_data_list.append(test_row)
    ratings_test.append(row['rating'])

predictions = []
correct_predictions = 0

for test_row, actual_rating in zip(test_data_list, ratings_test):
    predicted_binary = agent.next_state(test_row, print_result = True)
    predicted_rating_int = int("".join(map(str, predicted_binary)), 2)
    predicted_rating = predicted_rating_int
    print("predicted rating: ", predicted_rating)
    predictions.append(predicted_rating)
    
    if predicted_rating == actual_rating:
        correct_predictions += 1

# Evaluate model performance
mse = mean_squared_error(ratings_test, predictions)
mae = mean_absolute_error(ratings_test, predictions)
r2 = r2_score(ratings_test, predictions)

# Calculate percentage of correct predictions
accuracy = (correct_predictions / len(ratings_test)) * 100

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")
print(f"Accuracy (% correct predictions): {accuracy:.2f}%")