
import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Read dataset and split into train and eval
df = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep = ",")
df['num_rooms'] = df['total_rooms'] / df['households']
np.random.seed(seed = 1) #makes split reproducible
msk = np.random.rand(len(df)) < 0.8
traindf = df[msk]
evaldf = df[~msk]

# Train and eval input functions
SCALE = 100000

def train_input_fn(df, batch_size):
  return tf.estimator.inputs.pandas_input_fn(x = traindf[["num_rooms"]],
                                             y = traindf["median_house_value"] / SCALE,  # note the scaling
                                             num_epochs = None,
                                             batch_size = batch_size, # note the batch size
                                             shuffle = True)

def eval_input_fn(df, batch_size):
  return tf.estimator.inputs.pandas_input_fn(x = evaldf[["num_rooms"]],
                                             y = evaldf["median_house_value"] / SCALE,  # note the scaling
                                             num_epochs = 1,
                                             batch_size = batch_size,
                                             shuffle = False)

# Define feature columns
features = [tf.feature_column.numeric_column('num_rooms')]

def train_and_evaluate(args):
  # Compute appropriate number of steps
  num_steps = (len(traindf) / args['batch_size']) / args['learning_rate']  # if learning_rate=0.01, hundred epochs

  # Create custom optimizer
  myopt = tf.train.FtrlOptimizer(learning_rate = args['learning_rate']) # note the learning rate

  # Create rest of the estimator as usual
  estimator = tf.estimator.LinearRegressor(model_dir = args['output_dir'], 
                                           feature_columns = features, 
                                           optimizer = myopt)
  #Add rmse evaluation metric
  def rmse(labels, predictions):
    pred_values = tf.cast(predictions['predictions'], tf.float64)
    return {'rmse': tf.metrics.root_mean_squared_error(labels * SCALE, pred_values * SCALE)}
  estimator = tf.contrib.estimator.add_metrics(estimator, rmse)

  train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn(df = traindf, batch_size = args['batch_size']),
                                      max_steps = num_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn = eval_input_fn(df = evaldf, batch_size = len(evaldf)),
                                    steps = None)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)