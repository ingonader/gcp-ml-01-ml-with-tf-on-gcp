
# coding: utf-8

# # Hyperparameter tuning with Cloud ML Engine

# **Learning Objectives:**
#   * Improve the accuracy of a model by hyperparameter tuning

# In[1]:


import os
PROJECT = 'cloud-training-demos' # REPLACE WITH YOUR PROJECT ID
BUCKET = 'cloud-training-demos-ml' # REPLACE WITH YOUR BUCKET NAME
REGION = 'us-central1' # REPLACE WITH YOUR BUCKET REGION e.g. us-central1


# In[2]:


# for bash
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

os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = '1.8'  # Tensorflow version


# In[4]:


get_ipython().run_cell_magic('bash', '', 'gcloud config set project $PROJECT\ngcloud config set compute/region $REGION')


# ## Create command-line program
# 
# In order to submit to Cloud ML Engine, we need to create a distributed training program. Let's convert our housing example to fit that paradigm, using the Estimators API.

# In[5]:


get_ipython().run_cell_magic('bash', '', 'rm -rf house_prediction_module\nmkdir house_prediction_module\nmkdir house_prediction_module/trainer\ntouch house_prediction_module/trainer/__init__.py')


# In[11]:


get_ipython().run_cell_magic('writefile', 'house_prediction_module/trainer/task.py', 'import argparse\nimport os\nimport json\nimport shutil\n\nfrom . import model\n    \nif __name__ == \'__main__\' and "get_ipython" not in dir():\n  parser = argparse.ArgumentParser()\n  # TODO: Add learning_rate and batch_size as command line args\n  parser.add_argument(\n    \'--learning_rate\',\n    type = float,\n    help = \'Set the learning rate for the network training\',\n    default = 0.01\n  )\n  parser.add_argument(\n    \'--batch_size\',\n    type = int,\n    default = 30,\n    help = \'Batch size for batch gradient descent\'\n  )\n  parser.add_argument(\n      \'--output_dir\',\n      help = \'GCS location to write checkpoints and export models.\',\n      required = True\n  )\n  parser.add_argument(\n      \'--job-dir\',\n      help = \'this model ignores this field, but it is required by gcloud\',\n      default = \'junk\'\n  )\n  args = parser.parse_args()\n  arguments = args.__dict__\n  \n  # Unused args provided by service\n  arguments.pop(\'job_dir\', None)\n  arguments.pop(\'job-dir\', None)\n  \n  # Append trial_id to path if we are doing hptuning\n  # This code can be removed if you are not using hyperparameter tuning\n  arguments[\'output_dir\'] = os.path.join(\n      arguments[\'output_dir\'],\n      json.loads(\n          os.environ.get(\'TF_CONFIG\', \'{}\')\n      ).get(\'task\', {}).get(\'trial\', \'\')\n  )\n  \n  # Run the training\n  shutil.rmtree(arguments[\'output_dir\'], ignore_errors=True) # start fresh each time\n  \n  # Pass the command line arguments to our model\'s train_and_evaluate function\n  model.train_and_evaluate(arguments)')


# In[12]:


get_ipython().run_cell_magic('writefile', 'house_prediction_module/trainer/model.py', '\nimport numpy as np\nimport pandas as pd\nimport tensorflow as tf\n\ntf.logging.set_verbosity(tf.logging.INFO)\n\n# Read dataset and split into train and eval\ndf = pd.read_csv("https://storage.googleapis.com/ml_universities/california_housing_train.csv", sep = ",")\ndf[\'num_rooms\'] = df[\'total_rooms\'] / df[\'households\']\nnp.random.seed(seed = 1) #makes split reproducible\nmsk = np.random.rand(len(df)) < 0.8\ntraindf = df[msk]\nevaldf = df[~msk]\n\n# Train and eval input functions\nSCALE = 100000\n\ndef train_input_fn(df, batch_size):\n  return tf.estimator.inputs.pandas_input_fn(x = traindf[["num_rooms"]],\n                                             y = traindf["median_house_value"] / SCALE,  # note the scaling\n                                             num_epochs = None,\n                                             batch_size = batch_size, # note the batch size\n                                             shuffle = True)\n\ndef eval_input_fn(df, batch_size):\n  return tf.estimator.inputs.pandas_input_fn(x = evaldf[["num_rooms"]],\n                                             y = evaldf["median_house_value"] / SCALE,  # note the scaling\n                                             num_epochs = 1,\n                                             batch_size = batch_size,\n                                             shuffle = False)\n\n# Define feature columns\nfeatures = [tf.feature_column.numeric_column(\'num_rooms\')]\n\ndef train_and_evaluate(args):\n  # Compute appropriate number of steps\n  num_steps = (len(traindf) / args[\'batch_size\']) / args[\'learning_rate\']  # if learning_rate=0.01, hundred epochs\n\n  # Create custom optimizer\n  myopt = tf.train.FtrlOptimizer(learning_rate = args[\'learning_rate\']) # note the learning rate\n\n  # Create rest of the estimator as usual\n  estimator = tf.estimator.LinearRegressor(model_dir = args[\'output_dir\'], \n                                           feature_columns = features, \n                                           optimizer = myopt)\n  #Add rmse evaluation metric\n  def rmse(labels, predictions):\n    pred_values = tf.cast(predictions[\'predictions\'], tf.float64)\n    return {\'rmse\': tf.metrics.root_mean_squared_error(labels * SCALE, pred_values * SCALE)}\n  estimator = tf.contrib.estimator.add_metrics(estimator, rmse)\n\n  train_spec = tf.estimator.TrainSpec(input_fn = train_input_fn(df = traindf, batch_size = args[\'batch_size\']),\n                                      max_steps = num_steps)\n  eval_spec = tf.estimator.EvalSpec(input_fn = eval_input_fn(df = evaldf, batch_size = len(evaldf)),\n                                    steps = None)\n  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)')


# In[13]:


get_ipython().run_cell_magic('bash', '', 'rm -rf house_trained\nexport PYTHONPATH=${PYTHONPATH}:${PWD}/house_prediction_module\ngcloud ml-engine local train \\\n    --module-name=trainer.task \\\n    --job-dir=house_trained \\\n    --package-path=$(pwd)/trainer \\\n    -- \\\n    --batch_size=30 \\\n    --learning_rate=0.02 \\\n    --output_dir=house_trained')


# # Create hyperparam.yaml

# In[15]:


get_ipython().run_cell_magic('writefile', 'hyperparam.yaml', 'trainingInput:\n  hyperparameters:\n    goal: MINIMIZE\n    maxTrials: 5\n    maxParallelTrials: 1\n    hyperparameterMetricTag: rmse\n    params:\n    - parameterName: batch_size\n      type: INTEGER\n      minValue: 8\n      maxValue: 64\n      scaleType: UNIT_LINEAR_SCALE\n    - parameterName: learning_rate\n      type: DOUBLE\n      minValue: 0.01\n      maxValue: 0.1\n      scaleType: UNIT_LOG_SCALE')


# In[16]:


get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/house_trained   # CHANGE bucket name appropriately\ngsutil rm -rf $OUTDIR\nexport PYTHONPATH=${PYTHONPATH}:${PWD}/house_prediction_module\ngcloud ml-engine jobs submit training house_$(date -u +%y%m%d_%H%M%S) \\\n   --config=hyperparam.yaml \\\n   --module-name=trainer.task \\\n   --package-path=$(pwd)/house_prediction_module/trainer \\\n   --job-dir=$OUTDIR \\\n   --runtime-version=$TFVERSION \\\n   -- \\\n   --output_dir=$OUTDIR')


# In[18]:


get_ipython().system('gcloud ml-engine jobs describe house_190212_080418 # CHANGE jobId appropriately')


# ## Challenge exercise
# Add a few engineered features to the housing model, and use hyperparameter tuning to choose which set of features the model uses.
# 
# <p>
# Copyright 2018 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License