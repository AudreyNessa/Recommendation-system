# Recommendation-system
recommends movies a user might like based on the rating a user might give a movie.
uses the movieLens dataset to find the average rating a user gives for each genre.
also finds the average rating for each movie.
The predicted rating is found by the dot product of two neural networks(users and movies).
The networks contain two hidden layers and a dropout layer.
