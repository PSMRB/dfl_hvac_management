from src.library.Base import *
from sklearn.model_selection import train_test_split
import time

class Database:
    """
    Database is a class which contain all the sets (whole, training and test) as well as the predictor and response variable
    names.
    self.seed = rnd
    self.predictors
    self.response
    self.X
    self.y_true
    self.X_train
    self.X_test
    self.y_train_true
    self.y_test_true
    """

    def __init__(self, data_df, response_name: list, rnd=None, shuffle: bool = True):
        self.shuffled = shuffle
        self.seed = rnd
        cols = list(data_df.columns)
        self.response = [getcolumnname(r_n, cols) for r_n in response_name]
        self.predictors = cols
        for r in self.response:
            self.predictors.remove(r)
        self.X = data_df[self.predictors]
        self.y_true = data_df[self.response]  # y is a pd serie if len(response_name)==1

        # Divide and shuffle into train and test sets
        self.X_train, self.X_test, self.y_train_true, self.y_test_true = \
            train_test_split(self.X, self.y_true, test_size=0.2, random_state=rnd, shuffle=shuffle)

class Stopwatch:
    def __init__(self):
        """Initialize the stopwatch."""
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def start(self):
        """Start or resume the stopwatch."""
        if not self.running:
            self.start_time = time.time()
            self.running = True
        else:
            print("Stopwatch is already running.")

    def stop(self):
        """Stop the stopwatch and record the elapsed time."""
        if self.running:
            self.elapsed_time += time.time() - self.start_time
            self.running = False
            self.start_time = None
        else:
            print("Stopwatch is not running.")

    def reset(self):
        """Reset the stopwatch to its initial state."""
        self.start_time = None
        self.elapsed_time = 0
        self.running = False

    def get_elapsed_time(self):
        """Get the total elapsed time."""
        if self.running:
            return self.elapsed_time + (time.time() - self.start_time)
        return self.elapsed_time
