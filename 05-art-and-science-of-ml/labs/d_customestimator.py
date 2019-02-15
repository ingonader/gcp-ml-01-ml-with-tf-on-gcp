
# coding: utf-8

# <h1> Time series prediction using RNNs, with TensorFlow and Cloud ML Engine </h1>
# 
# This notebook illustrates:
# <ol>
# <li> Creating a Recurrent Neural Network in TensorFlow
# <li> Creating a Custom Estimator in tf.estimator
# <li> Training on Cloud ML Engine
# </ol>
# 
# <p>
# 
# <h3> Simulate some time-series data </h3>
# 
# Essentially a set of sinusoids with random amplitudes and frequencies.

# In[1]:


import os
PROJECT = 'cloud-training-demos' # REPLACE WITH YOUR PROJECT ID
BUCKET = 'cloud-training-demos-ml' # REPLACE WITH YOUR BUCKET NAME
REGION = 'us-central1' # REPLACE WITH YOUR BUCKET REGION e.g. us-central1
os.environ['TFVERSION'] = '1.8'  # Tensorflow version


# In[2]:


import os
output = os.popen("gcloud config get-value project").readlines()
project_name = output[0][:-1]

# change these to try this notebook out
PROJECT = project_name
BUCKET = project_name
BUCKET = BUCKET.replace("qwiklabs-gcp-", "inna-bckt-")
REGION = 'europe-west1'  ## note: Cloud ML Engine not availabe in europe-west3!

# set environment variables:
os.environ['BUCKET'] = BUCKET
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION

print(PROJECT)
print(BUCKET)
print("gsutil mb -l {0} gs://{1}".format(REGION, BUCKET))

## set config for gcp config: [[?]]
print(os.popen("gcloud config set project $PROJECT").readlines())
print(os.popen("gcloud config set compute/region $REGION").readlines())


# In[3]:


get_ipython().run_cell_magic('bash', '', 'gcloud config set project $PROJECT\ngcloud config set compute/region $REGION')


# In[4]:


import tensorflow as tf
print(tf.__version__)


# In[5]:


import numpy as np
import seaborn as sns
import pandas as pd

SEQ_LEN = 10
def create_time_series():
  freq = (np.random.random() * 0.5) + 0.1  # 0.1 to 0.6
  ampl = np.random.random() + 0.5  # 0.5 to 1.5
  x = np.sin(np.arange(0, SEQ_LEN) * freq) * ampl
  return x

for i in range(0, 5):
  sns.tsplot( create_time_series() );  # 5 series


# In[6]:


def to_csv(filename, N):
  with open(filename, 'w') as ofp:
    for lineno in range(0, N):
      seq = create_time_series()
      line = ",".join(map(str, seq))
      ofp.write(line + '\n')

to_csv('train.csv', 1000)  # 1000 sequences
to_csv('valid.csv',  50)


# In[7]:


get_ipython().system('head -5 train.csv valid.csv')


# <h2> RNN </h2>
# 
# For more info, see:
# <ol>
# <li> http://colah.github.io/posts/2015-08-Understanding-LSTMs/ for the theory
# <li> https://www.tensorflow.org/tutorials/recurrent for explanations
# <li> https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb for sample code
# </ol>
# 
# Here, we are trying to predict from 9 values of a timeseries, the tenth value.
# 
# <p>
# 
# <h3> Imports </h3>
# 
# Several tensorflow packages and shutil

# In[8]:


import tensorflow as tf
import shutil
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.rnn as rnn


# <h3> Input Fn to read CSV </h3>
# 
# Our CSV file structure is quite simple -- a bunch of floating point numbers (note the type of DEFAULTS). We ask for the data to be read BATCH_SIZE sequences at a time.  The Estimator API in tf.contrib.learn wants the features returned as a dict. We'll just call this timeseries column 'rawdata'.
# <p>
# Our CSV file sequences consist of 10 numbers. We'll assume that 9 of them are inputs and we need to predict the last one.

# In[9]:


DEFAULTS = [[0.0] for x in range(0, SEQ_LEN)]
BATCH_SIZE = 20
TIMESERIES_COL = 'rawdata'
# In each sequence, column index 0 to N_INPUTS - 1 are features, and column index N_INPUTS to SEQ_LEN are labels
N_OUTPUTS = 1
N_INPUTS = SEQ_LEN - N_OUTPUTS


# Reading data using the Estimator API in tf.estimator requires an input_fn. This input_fn needs to return a dict of features and the corresponding labels.
# <p>
# So, we read the CSV file.  The Tensor format here will be a scalar -- entire line.  We then decode the CSV. At this point, all_data will contain a list of scalar Tensors. There will be SEQ_LEN of these tensors.
# <p>
# We split this list of SEQ_LEN tensors into a list of N_INPUTS Tensors and a list of N_OUTPUTS Tensors. We stack them along the first dimension to then get a vector Tensor for each.  We then put the inputs into a dict and call it features.  The other is the ground truth, so labels.

# In[10]:


# Read data and convert to needed format
def read_dataset(filename, mode, batch_size = 512):
  def _input_fn():
    # Provide the ability to decode a CSV
    def decode_csv(line):
      # all_data is a list of scalar tensors
      all_data = tf.decode_csv(line, record_defaults = DEFAULTS)
      inputs = all_data[:len(all_data) - N_OUTPUTS]  # first N_INPUTS values
      labels = all_data[len(all_data) - N_OUTPUTS:] # last N_OUTPUTS values

      # Convert each list of rank R tensors to one rank R+1 tensor
      inputs = tf.stack(inputs, axis = 0)   ## stack the features into a matrix
      labels = tf.stack(labels, axis = 0)
      
      # Convert input R+1 tensor into a feature dictionary of one R+1 tensor
      features = {TIMESERIES_COL: inputs}  ## one feature, but which is a matrix

      return features, labels

    # Create list of files that match pattern
    file_list = tf.gfile.Glob(filename)

    # Create dataset from file list
    dataset = tf.data.TextLineDataset(file_list).map(decode_csv)

    if mode == tf.estimator.ModeKeys.TRAIN:
        num_epochs = None # indefinitely
        dataset = dataset.shuffle(buffer_size = 10 * batch_size)
    else:
        num_epochs = 1 # end-of-input after this

    dataset = dataset.repeat(num_epochs).batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels
  return _input_fn


# <h3> Define RNN </h3>
# 
# A recursive neural network consists of possibly stacked LSTM cells.
# <p>
# The RNN has one output per input, so it will have 8 output cells.  We use only the last output cell, but rather use it directly, we do a matrix multiplication of that cell by a set of weights to get the actual predictions. This allows for a degree of scaling between inputs and predictions if necessary (we don't really need it in this problem).
# <p>
#     
# <p> You have two tasks to complete
#     
# <ol>
#     <li>Firstly, define loss, train_op and eval_metric_ops as a function of mode</li>
#     <li>Secondly, use the defined variables to instantiate an EstimatorSpec</li>
#  </ol>

# In[26]:


LSTM_SIZE = 3  # number of hidden layers in each of the LSTM cells

# Create the inference model
def simple_rnn(features, labels, mode):
  # 0. Reformat input shape to become a sequence
  x = tf.split(features[TIMESERIES_COL], N_INPUTS, 1) # Splits a tensor into sub tensors (will create N_INPUTS tensors)
    
  # 1. Configure the RNN
  lstm_cell = rnn.BasicLSTMCell(LSTM_SIZE, forget_bias = 1.0)
  outputs, _ = rnn.static_rnn(lstm_cell, x, dtype = tf.float32)

  # Slice to keep only the last cell of the RNN
  outputs = outputs[-1]
  
  # Output is result of linear activation of last layer of RNN
  weight = tf.Variable(tf.random_normal([LSTM_SIZE, N_OUTPUTS]))
  bias = tf.Variable(tf.random_normal([N_OUTPUTS]))
  predictions = tf.matmul(outputs, weight) + bias
    
  # 2. Loss function, training/eval ops
  # TODO: Implement training/eval ops for training, evaluation and prediction
  if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
    loss = tf.losses.mean_squared_error(labels = labels,
                                      predictions = predictions)
    train_op = tf.contrib.layers.optimize_loss(loss = loss,
                                               global_step = tf.train.get_global_step(),
                                               learning_rate = 0.01,
                                               optimizer = "SGD")
    eval_metric_ops = {
      "rmse" : tf.metrics.root_mean_squared_error(labels, predictions)
    }
  else:  ## if mode other than train or eval.
    loss = None
    train_op = None
    eval_metric_ops = None

  
  # 3. Create predictions
  predictions_dict = {"predicted": predictions}
  
  # 4. Create export outputs
  export_outputs = {"predict_export_outputs": tf.estimator.export.PredictOutput(outputs = predictions)}
  
  # 5. Return an EstimatorSpec
  return tf.estimator.EstimatorSpec( # TODO
    mode = mode,
    predictions = predictions,
    loss = loss,
    train_op = train_op,
    eval_metric_ops = eval_metric_ops,
    export_outputs = export_outputs
  )


# <h3> Estimator </h3>
# 
# Distributed training is launched off using an Estimator.  The key line here is that we use tf.estimator.Estimator rather than, say tf.estimator.DNNRegressor.  This allows us to provide a model_fn, which will be our RNN defined above.  Note also that we specify a serving_input_fn -- this is how we parse the input data provided to us at prediction time.
# 
# You have one task to complete: instantiate an estimator using the model function we defined previously.

# In[27]:


# Create functions to read in respective datasets
def get_train():
  return read_dataset(filename = 'train.csv', mode = tf.estimator.ModeKeys.TRAIN, batch_size = 512)

def get_valid():
  return read_dataset(filename = 'valid.csv', mode = tf.estimator.ModeKeys.EVAL, batch_size = 512)


# In[28]:


# Create serving input function
def serving_input_fn():
  feature_placeholders = {
      TIMESERIES_COL: tf.placeholder(tf.float32, [None, N_INPUTS])
  }
  
  features = {
    key: tf.expand_dims(tensor, -1)
    for key, tensor in feature_placeholders.items()
  }
  features[TIMESERIES_COL] = tf.squeeze(features[TIMESERIES_COL], axis = [2])
    
  return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


# In[29]:


# Create custom estimator's train and evaluate function
def train_and_evaluate(output_dir):
  # TODO: Instantiate an estimator using our model function
  estimator = tf.estimator.Estimator(model_fn = simple_rnn, 
                                    model_dir = output_dir)
  train_spec = tf.estimator.TrainSpec(input_fn = get_train(),
                                    max_steps = 1000)
  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
  eval_spec = tf.estimator.EvalSpec(input_fn = get_valid(),
                                  steps = None,
                                  exporters = exporter)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# In[30]:


# Run the model
shutil.rmtree('outputdir', ignore_errors = True) # start fresh each time
train_and_evaluate('outputdir')


# <h3> Standalone Python module </h3>
# 
# To train this on Cloud ML Engine, we take the code in this notebook and make a standalone Python module.

# In[38]:


get_ipython().run_cell_magic('bash', '', '# Run module as-is\necho $PWD\nrm -rf outputdir\nexport PYTHONPATH=${PYTHONPATH}:${PWD}/simplernn\npython -m trainer.task \\\n  --train_data_paths="${PWD}/train.csv*" \\\n  --eval_data_paths="${PWD}/valid.csv*"  \\\n  --output_dir=outputdir \\\n  --job-dir=./tmp')


# Try out online prediction.  This is how the REST API will work after you train on Cloud ML Engine

# In[31]:


get_ipython().run_line_magic('writefile', 'test.json')
{"rawdata_input": [0,0.214,0.406,0.558,0.655,0.687,0.65,0.549,0.393]}


# In[34]:


# local predict doesn't work with Python 3 yet.
# %%bash
# MODEL_DIR=$(ls ./outputdir/export/exporter/)
# gcloud ml-engine local predict --model-dir=./outputdir/export/exporter/$MODEL_DIR --json-instances=test.json


# <h3> Cloud ML Engine </h3>
# 
# Now to train on Cloud ML Engine.

# In[39]:


get_ipython().run_cell_magic('bash', '', 'echo $BUCKET\necho ${PWD}/simplernn/trainer\n\n# Run module on Cloud ML Engine\nOUTDIR=gs://${BUCKET}/simplernn/model_trained\nJOBNAME=simplernn_$(date -u +%y%m%d_%H%M%S)\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n   --region=$REGION \\\n   --module-name=trainer.task \\\n   --package-path=${PWD}/simplernn/trainer \\\n   --job-dir=$OUTDIR \\\n   --staging-bucket=gs://$BUCKET \\\n   --scale-tier=BASIC \\\n   --runtime-version=1.4 \\\n   -- \\\n   --train_data_paths="gs://${BUCKET}/train.csv*" \\\n   --eval_data_paths="gs://${BUCKET}/valid.csv*"  \\\n   --output_dir=$OUTDIR')


# <h2> Variant: long sequence </h2>
# 
# To create short sequences from a very long sequence.

# In[42]:


import tensorflow as tf
import numpy as np

def breakup(sess, x, lookback_len):
  N = sess.run(tf.size(x))
  windows = [tf.slice(x, [b], [lookback_len]) for b in range(0, N-lookback_len)]
  windows = tf.stack(windows)
  return windows

x = tf.constant(np.arange(1,11, dtype=np.float32))
with tf.Session() as sess:
    print('input=', x.eval())
    seqx = breakup(sess, x, 5)
    print('output=', seqx.eval())


# ## Variant: Keras
# 
# You can also invoke a Keras model from within the Estimator framework by creating an estimator from the compiled Keras model:

# In[40]:


def make_keras_estimator(output_dir):
  ## import keras _from_ tensorflow(!)
  from tensorflow import keras
  model = keras.models.Sequential()
  model.add(keras.layers.Dense(32, input_shape=(N_INPUTS,), name=TIMESERIES_INPUT_LAYER))
  model.add(keras.layers.Activation('relu'))
  model.add(keras.layers.Dense(1))
  model.compile(loss = 'mean_squared_error',
                optimizer = 'adam',
                metrics = ['mae', 'mape']) # mean absolute [percentage] error
  return keras.estimator.model_to_estimator(model)


# In[41]:


get_ipython().run_cell_magic('bash', '', '# Run module as-is\necho $PWD\nrm -rf outputdir\nexport PYTHONPATH=${PYTHONPATH}:${PWD}/simplernn\npython -m trainer.task \\\n  --train_data_paths="${PWD}/train.csv*" \\\n  --eval_data_paths="${PWD}/valid.csv*"  \\\n  --output_dir=${PWD}/outputdir \\\n  --job-dir=./tmp --keras')


# In[43]:


get_ipython().system('cat /content/datalab/gcp-ml-01-ml-with-tf-on-gcp/05-art-and-science-of-ml/labs/simplernn/trainer/task.py')


# In[44]:


get_ipython().system('cat /content/datalab/gcp-ml-01-ml-with-tf-on-gcp/05-art-and-science-of-ml/labs/simplernn/trainer/model.py')


# Copyright 2017 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License