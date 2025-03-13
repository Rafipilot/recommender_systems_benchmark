import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Merge ratings and movies
merged_data = pd.merge(ratings, movies, on="movieId")

# Feature engineering: Create features for the model
# We will use userId, movieId, and genre for simplicity
merged_data['genres_list'] = merged_data['genres'].apply(lambda x: x.split('|'))
merged_data['num_genres'] = merged_data['genres_list'].apply(len)

# One-hot encoding for genres
genre_dummies = merged_data['genres'].str.get_dummies('|')
merged_data = pd.concat([merged_data, genre_dummies], axis=1)

# Define features and target variable
X = merged_data[['userId', 'movieId', 'num_genres'] + list(genre_dummies.columns)]
y = merged_data['rating']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Linear Regression Model Performance:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")
