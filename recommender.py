import numpy as np
import numpy.ma as ma
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
pd.set_option("display.precision", 1)

NUM_OUTPUTS = 32

def main():
    #load data and set configuration variables
    item_train, user_train, y_train, item_features, user_features, item_vecs, movie_dict = load_data()
    num_user_features = user_train.shape[1] - 3 #remove user_id, rating count and ave rating during training
    num_item_features = item_train.shape[1] - 1 #remove movie id at train time
    uvs = 3 #user genre vector start
    ivs = 3 #item genre vector start
    u_s = 3 #start of columns to use in training, users
    i_s = 1 #start of columns to use in training, items

    #print the training data of item, user and y for visualisation
    print_data(item_train, item_features, max_count = 5)
    print_data(user_train, user_features, max_count = 5)
    print(f"y_train[:5]: {y_train[:5]}") #the movie rating given by the user

    #scale the input and output features
    item_train_unscaled = item_train
    user_train_unscaled = user_train
    y_train_unscaled = y_train
    
    scalarItem = StandardScaler()
    scalarItem.fit(item_train)
    item_train = scalarItem.transform(item_train)
    
    scalarUser = StandardScaler()
    scalarUser.fit(user_train)
    user_train = scalarUser.transform(user_train)

    scalarTarget =MinMaxScaler((-1, 1))
    scalarTarget.fit(y_train.reshape(-1,1))
    y_train = scalarTarget.transform(y_train.reshape(-1,1))

    #split the data and set the random_state to the same value to ensure
    #item, user, and y are shuffled identically
    item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
    user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
    y_train, y_test       = train_test_split(y_train, train_size=0.80, shuffle=True, random_state=1)

    #fit the training data to a compiled model and evaluate the model
    model = get_model(num_user_features, num_item_features)
    tf.random.set_seed(1)
    model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)

    model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)

    #create a new user and have the model suggest movies for that user
    new_user_id = 5000; new_rating_ave = 0.0; new_action = 4.0;
    new_adventure = 0.0; new_animation = 3.0; new_children = 3.5;
    new_comedy = 5.0; new_crime = 0.0; new_documentary = 0.0; new_drama = 0.0;
    new_fantasy = 0.0; new_film_noir = 0.0; new_horror = 0.0; new_imax = 0.0;
    new_musical = 0.0; new_mystery = 0.0; new_romance = 0.0; new_scifi = 0.0;
    new_thriller = 0.0; new_war = 0.0; new_western = 0.0; new_rating_count = 3;

    user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave, new_action,
                          new_adventure, new_animation, new_children, new_comedy,
                          new_crime, new_documentary, new_drama, new_fantasy, new_film_noir,
                          new_horror,  new_imax, new_musical, new_mystery, new_romance,
                          new_scifi, new_thriller, new_war, new_western
                          ]])

    #generate and replicate the user vector to match the number of movies in the data set
    user_vecs = gen_user_vecs(user_vec, len(item_vecs))

    #scale over the user and item vectors
    suser_vecs = scalarUser.transform(user_vecs)
    sitem_vecs = scalarItem.transform(item_vecs)

    #make a prediction
    y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

    #unscale the prediction
    y_pu = scalarTarget.inverse_transform(y_p)

    #sort the results, highest prediction first and print out the movies
    sorted_index = np.argsort(-y_pu, axis=0).reshape(-1).tolist()
    sorted_ypu = y_pu[sorted_index]
    sorted_items = item_vecs[sorted_index] #using unscaled vectors for display
                        
    print_pred_movies(sorted_ypu, sorted_items, movie_dict, max_count=10)



def load_data():
    """
    Load the MovieLens data set from the files stored in a directory to
    lists. The directory contains four csv files. The movies.csv file contains
    the movie id, title with the year in brackets, and genres, while
    the ratings.csv contains the user id,movie id, and rating. The links.csv and
    tags.csv will not be used to train.

    The function will return a tuple comprising item_train(the movie id, year,
    average rating, and what genre using one-hot encoding), user_train(user id,
    rating count, average rating, and average rating for each genre), y_train(y_train[i]
    is the rating of user[i] for movie[i]), item_features(a list of the column headers in
    item_train), user_features(a list of the column headers in user_train),
    item_vecs(all the movies with their features), dfMovies(the dataframe of movies.csv)
    """

    #read movie.csv and rating.csv, and store them in pandas dataframes
    dfMovies = pd.read_csv("ml-latest-small/movies.csv")
    dfRatings = pd.read_csv("ml-latest-small/ratings.csv")

    #create item and user train dataframs to store the modified dataset
    item_train = pd.DataFrame()
    user_train = pd.DataFrame()

    #add movie id and year to item_train
    item_train["movieId"] = dfRatings["movieId"]
    movies = dfMovies.groupby(["movieId"]).first() #group movies by their ids
    item_train["year"] = movies["title"][item_train["movieId"]].str.extract(r"\((\d{4})\)")[0].tolist()
    item_train["year"] = item_train["year"].fillna('1950') #set year of movies without a year to 1950
    
    #add the average rating to item_train
    averageRatings = dfRatings.groupby(["movieId"])["rating"].mean()
    item_train["ave rating"] = averageRatings[item_train["movieId"]].tolist()

    #add the user id and add the number of movies a user has rated to user_train
    user_train["userId"] = dfRatings["userId"]
    userRatedCount = dfRatings.groupby(["userId"])["movieId"].count()
    user_train["rating count"] = userRatedCount[user_train["userId"]].tolist()

    #add the average rating of the users
    averageRatings = dfRatings.groupby(["userId"])["rating"].mean()
    user_train["ave rating"] = averageRatings[user_train["userId"]].tolist()

    #split and store all the possible genres in a list
    genres = set()
                                                   
    for genre in  dfMovies["genres"]:
        if genre != "(no genres listed)":
            genres.update(set(genre.split('|')))

    #sort the genres in ascending order
    genres = sorted(genres)

    #add the genre of each movie in item_train using one-hot encoding
    for genre in genres:
        item_train[genre] = np.where(
            movies["genres"][item_train["movieId"]].str.find(genre) != -1 ,
            1, 0
            )
    
    #add the average rating by the users for each genre
    for genre in genres:
        #store the user that have rated a movie with this genre
        usersGenre = pd.DataFrame(
            data = user_train[(item_train[genre] == 1)],
            columns = ["userId"]
            )
        
        #calculate the average rating for the users of this genre and store
        usersGenreAveRating = dfRatings.groupby([usersGenre["userId"]])["rating"].mean()
        user_train[genre] = 0
        uniqueUsersGenre = usersGenre.drop_duplicates()
        
        for user in uniqueUsersGenre["userId"]:
            user_train.loc[user_train["userId"] == user, genre] = usersGenreAveRating[user]
    
    #add y_train than stores the ratings of a user for a movie
    y_train = np.array(dfRatings["rating"], dtype=float)

    #make an array with the features of all the movies without duplicates
    item_vecs = np.array(item_train.drop_duplicates(subset=["movieId"]))

    #store other relevant information
    item_features = list(item_train.columns)
    user_features = list(user_train.columns)

    #convert user_train and item_train to numpy arrays of numbers to be used for training
    user_train = np.array(user_train, dtype=float)
    item_train = np.array(item_train, dtype=float)

    return item_train, user_train, y_train, item_features, user_features, item_vecs, dfMovies
    
def print_data(data_values, data_features, max_count = 10):
    """
    Converts a numpy array to a pandas dataframe and prints out the first max_count
    number of elements in the data frame
    """
    dfData = pd.DataFrame(data = data_values, columns = data_features)
    print(dfData.head(max_count))

def get_model(num_user_features, num_item_features):
    """
    Returns a compiled neural network model with the input shape being the number of features.
    And the output layer has NUM_OUTPUTS units
    
    """

    #set the output of the training data and tf.random.set_seed() to have consistent results
    tf.random.set_seed(1)

    user_NN = tf.keras.models.Sequential([
        #create two hidden layers with relu activation and an output layer with no activation
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),

        #prevent overfitting
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(NUM_OUTPUTS,)
        ])

    item_NN = tf.keras.models.Sequential([
        #create two hidden layers with relu activation and an output layer with no activation
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),

        #prevent overfitting
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(NUM_OUTPUTS,)
        ])

    #create the user input and point to the base network
    input_user = tf.keras.layers.Input(shape=(num_user_features))
    vu = user_NN(input_user)
    input_item = tf.linalg.l2_normalize(vu, axis=1) 

    #create the item input and point to the base network
    input_item = tf.keras.layers.Input(shape=(num_item_features))
    vm = item_NN(input_item)
    vm = tf.linalg.l2_normalize(vm, axis=1)

    #compute the dot product of the two vectors vu and vm
    output = tf.keras.layers.Dot(axes=1)([vu, vm])

    #specify the inputs and outputs of the model
    model = tf.keras.Model([input_user, input_item], output)

    #compile and return the model
    tf.random.set_seed(1)
    cost_fn = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(learning_rate=0.003)
    model.compile(optimizer=opt, loss=cost_fn)

    return model
    
def gen_user_vecs(user_vec, length):
    """
    repeats the user vector using the specified number of times and returns the vector
    """
    return np.tile(user_vec, (length, 1))

def print_pred_movies(sorted_ypu, sorted_items, movie_dict, max_count=5):
    """
    prints out the max_count predicted movies a user might like based on
    the predicted rating the user will give movies
    """
                        
    print("The predicted movies and ratings are: ")
    #create and add values to a dataframe of top movies
    for i in range(max_count):
        print(
            movie_dict[movie_dict["movieId"] == sorted_items[i][0]]["title"].to_string(index=False),
            sorted_ypu[i][0]
            )
            
    
if __name__ == "__main__":
    main()
