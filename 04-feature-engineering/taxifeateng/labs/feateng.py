
# coding: utf-8

# <h1> Feature Engineering </h1>
# 
# In this notebook, you will learn how to incorporate feature engineering into your pipeline.
# <ul>
# <li> Working with feature columns </li>
# <li> Adding feature crosses in TensorFlow </li>
# <li> Reading data from BigQuery </li>
# <li> Creating datasets using Dataflow </li>
# <li> Using a wide-and-deep model </li>
# </ul>

# Apache Beam only works in Python 2 at the moment, so we're going to switch to the Python 2 kernel. In the above menu, click the dropdown arrow and select `python2`. After that, run the following to ensure we've installed Beam.

# In[1]:


get_ipython().run_cell_magic('bash', '', 'source activate py2env\nconda install -y pytz\npip uninstall -y google-cloud-dataflow\npip install --upgrade apache-beam[gcp]')


# After doing a pip install, you have to ```Reset Session``` so that the new packages are picked up.  Please click on the button in the above menu.

# In[1]:


import tensorflow as tf
import apache_beam as beam
import shutil
print(tf.__version__)


# <h2> 1. Environment variables for project and bucket </h2>
# 
# <li> Your project id is the *unique* string that identifies your project (not the project name). You can find this from the GCP Console dashboard's Home page.  My dashboard reads:  <b>Project ID:</b> cloud-training-demos </li>
# <li> Cloud training often involves saving and restoring model files. Therefore, we should <b>create a single-region bucket</b>. If you don't have a bucket already, I suggest that you create one from the GCP console (because it will dynamically check whether the bucket name you want is available) </li>
# </ol>
# <b>Change the cell below</b> to reflect your Project ID and bucket name.
# 

# In[2]:


import os
REGION = 'us-central1' # Choose an available region for Cloud MLE from https://cloud.google.com/ml-engine/docs/regions.
BUCKET = 'cloud-training-demos-ml' # REPLACE WITH YOUR BUCKET NAME. Use a regional bucket in the region you selected.
PROJECT = 'cloud-training-demos'    # CHANGE THIS


# In[9]:


import os
output = os.popen("gcloud config get-value project").readlines()
project_name = output[0][:-1]

# change these to try this notebook out
PROJECT = project_name
BUCKET = project_name
BUCKET = BUCKET.replace("qwiklabs-", "inna-")
REGION = 'europe-west1'  ## note: Cloud ML Engine not availabe in europe-west3!

print(PROJECT)
print(BUCKET)
print("gsutil mb -l {0} gs://{1}".format(REGION, BUCKET))

# for bash
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = '1.8' 

## ensure we're using python2 env
os.environ['CLOUDSDK_PYTHON'] = 'python2'


# In[10]:


get_ipython().run_cell_magic('bash', '', '## ensure gcloud is up to date\ngcloud components update\n\ngcloud config set project $PROJECT\ngcloud config set compute/region $REGION\n\n## ensure we predict locally with our current Python environment\ngcloud config set ml_engine/local_python `which python`')


# <h2> 2. Specifying query to pull the data </h2>
# 
# Let's pull out a few extra columns from the timestamp.

# In[11]:


def create_query(phase, EVERY_N):
  if EVERY_N == None:
    EVERY_N = 4 #use full dataset
    
  #select and pre-process fields
  base_query = """
SELECT
  (tolls_amount + fare_amount) AS fare_amount,
  DAYOFWEEK(pickup_datetime) AS dayofweek,
  HOUR(pickup_datetime) AS hourofday,
  pickup_longitude AS pickuplon,
  pickup_latitude AS pickuplat,
  dropoff_longitude AS dropofflon,
  dropoff_latitude AS dropofflat,
  passenger_count*1.0 AS passengers,
  CONCAT(STRING(pickup_datetime), STRING(pickup_longitude), STRING(pickup_latitude), STRING(dropoff_latitude), STRING(dropoff_longitude)) AS key
FROM
  [nyc-tlc:yellow.trips]
WHERE
  trip_distance > 0
  AND fare_amount >= 2.5
  AND pickup_longitude > -78
  AND pickup_longitude < -70
  AND dropoff_longitude > -78
  AND dropoff_longitude < -70
  AND pickup_latitude > 37
  AND pickup_latitude < 45
  AND dropoff_latitude > 37
  AND dropoff_latitude < 45
  AND passenger_count > 0
  """
  
  #add subsampling criteria by modding with hashkey
  if phase == 'train': 
    query = "{} AND ABS(HASH(pickup_datetime)) % {} < 2".format(base_query,EVERY_N)
  elif phase == 'valid': 
    query = "{} AND ABS(HASH(pickup_datetime)) % {} == 2".format(base_query,EVERY_N)
  elif phase == 'test':
    query = "{} AND ABS(HASH(pickup_datetime)) % {} == 3".format(base_query,EVERY_N)
  return query
    
print create_query('valid', 100) #example query using 1% of data


# In[12]:


import google.datalab.bigquery as bq

query_test = """
#legacySQL
SELECT
  (tolls_amount + fare_amount) AS fare_amount,
  DAYOFWEEK(pickup_datetime) AS dayofweek,
  HOUR(pickup_datetime) AS hourofday,
  pickup_longitude AS pickuplon,
  pickup_latitude AS pickuplat,
  dropoff_longitude AS dropofflon,
  dropoff_latitude AS dropofflat,
  passenger_count*1.0 AS passengers,
  CONCAT(STRING(pickup_datetime), STRING(pickup_longitude), STRING(pickup_latitude), STRING(dropoff_latitude), STRING(dropoff_longitude)) AS key
FROM
  [nyc-tlc:yellow.trips]
WHERE
  trip_distance > 0
  AND fare_amount >= 2.5
  AND pickup_longitude > -78
  AND pickup_longitude < -70
  AND dropoff_longitude > -78
  AND dropoff_longitude < -70
  AND pickup_latitude > 37
  AND pickup_latitude < 45
  AND dropoff_latitude > 37
  AND dropoff_latitude < 45
  AND passenger_count > 0
  AND ABS(HASH(pickup_datetime)) % 100 == 2
  LIMIT 10
"""

res = bq.Query(query_test).execute().result().to_dataframe()
res.head(n = 10)


# Try the query above in https://bigquery.cloud.google.com/table/nyc-tlc:yellow.trips if you want to see what it does (ADD LIMIT 10 to the query!)

# ```
# Row fare_amount dayofweek hourofday pickuplon pickuplat dropofflon  dropofflat  passengers  key  
# 1 18.0  5 19  -73.973092  40.750065 -74.009415  40.7128 1.0 2013-05-02 19:25:00.000000-73.973140.750140.7128-74.0094   
# 2 39.83 6 17  -73.863462  40.769847 -73.983307  40.735722 3.0 2013-12-06 17:50:00.000000-73.863540.769840.7357-73.9833   
# 3 21.5  2 21  -73.994028  40.751273 -73.977827  40.680942 1.0 2012-11-26 21:45:00.000000-73.99440.751340.6809-73.9778  
# 4 35.3  5 11  -73.97348 40.751112 -73.861642  40.768322 1.0 2012-11-15 11:32:00.000000-73.973540.751140.7683-73.8616   
# 5 25.5  6 12  -73.969537  40.760892 -73.885182  40.77218  1.0 2012-11-16 12:20:00.000000-73.969540.760940.7722-73.8852   
# 6 24.5  6 11  -73.9526  40.772473 -73.861593  40.76822  5.0 2011-04-29 11:30:00.000000-73.952640.772540.7682-73.8616   
# 7 30.3  6 5 -73.945982  40.8038 -73.872017  40.774515 6.0 2013-02-15 05:54:00.000000-73.94640.803840.7745-73.872   
# 8 39.33 7 18  -73.871122  40.773887 -73.986117  40.75057  1.0 2014-10-18 18:17:00.000000-73.871140.773940.7506-73.9861   
# 9 30.5  6 8 -73.994405  40.690012 -73.984307  40.763057 1.0 2014-09-26 08:45:00.000000-73.994440.6940.7631-73.9843   
# 10  18.0  7 1 -74.00346 40.725322 -73.97648 40.78495  2.0 2012-12-01 01:39:00.000000-74.003540.725340.785-73.9765 
# ```

# <h2> 3. Preprocessing Dataflow job from BigQuery </h2>
# 
# This code reads from BigQuery and saves the data as-is on Google Cloud Storage.  We can do additional preprocessing and cleanup inside Dataflow, but then we'll have to remember to repeat that prepreprocessing during inference. It is better to use tf.transform which will do this book-keeping for you, or to do preprocessing within your TensorFlow model. We will look at this in future notebooks. For now, we are simply moving data from BigQuery to CSV using Dataflow.
# 
# While we could read from BQ directly from TensorFlow (See: https://www.tensorflow.org/api_docs/python/tf/contrib/cloud/BigQueryReader), it is quite convenient to export to CSV and do the training off CSV.  Let's use Dataflow to do this at scale.
# 
# Because we are running this on the Cloud, you should go to the GCP Console (https://console.cloud.google.com/dataflow) to look at the status of the job. It will take several minutes for the preprocessing job to launch.

# In[13]:


get_ipython().run_line_magic('bash', '')
gsutil -m rm -rf gs://$BUCKET/taxifare/ch4/taxi_preproc/


# In[14]:


import datetime

####
# Arguments:
#   -rowdict: Dictionary. The beam bigquery reader returns a PCollection in
#     which each row is represented as a python dictionary
# Returns:
#   -rowstring: a comma separated string representation of the record with dayofweek
#     converted from int to string (e.g. 3 --> Tue)
####
def to_csv(rowdict):
  days = ['null', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
  CSV_COLUMNS = 'fare_amount,dayofweek,hourofday,pickuplon,pickuplat,dropofflon,dropofflat,passengers,key'.split(',')
  rowdict['dayofweek'] = days[rowdict['dayofweek']]
  rowstring = ','.join([str(rowdict[k]) for k in CSV_COLUMNS])
  return rowstring


####
# Arguments:
#   -EVERY_N: Integer. Sample one out of every N rows from the full dataset.
#     Larger values will yield smaller sample
#   -RUNNER: 'DirectRunner' or 'DataflowRunner'. Specfy to run the pipeline
#     locally or on Google Cloud respectively. 
# Side-effects:
#   -Creates and executes dataflow pipeline. 
#     See https://beam.apache.org/documentation/programming-guide/#creating-a-pipeline
####
def preprocess(EVERY_N, RUNNER):
  job_name = 'preprocess-taxifeatures' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')
  print 'Launching Dataflow job {} ... hang on'.format(job_name)
  OUTPUT_DIR = 'gs://{0}/taxifare/ch4/taxi_preproc/'.format(BUCKET)

  #dictionary of pipeline options
  options = {
    'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
    'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
    'job_name': 'preprocess-taxifeatures' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S'),
    'project': PROJECT,
    'runner': RUNNER
  }
  #instantiate PipelineOptions object using options dictionary
  opts = beam.pipeline.PipelineOptions(flags=[], **options)
  #instantantiate Pipeline object using PipelineOptions
  with beam.Pipeline(options=opts) as p:
      for phase in ['train', 'valid']:
        query = create_query(phase, EVERY_N) 
        outfile = os.path.join(OUTPUT_DIR, '{}.csv'.format(phase))
        (
          p | 'read_{}'.format(phase) >> beam.io.Read(beam.io.BigQuerySource(query = query)) ##TODO: read from BigQuery
            | 'tocsv_{}'.format(phase) >> beam.Map(to_csv) ##TODO: apply the to_csv function to every row
            | 'write_{}'.format(phase) >> beam.io.Write(beam.io.WriteToText(outfile)) ##TODO: write to outfile
        )
  print("Done")


# Run pipeline locally

# In[15]:


preprocess(EVERY_N = 50 * 10000, RUNNER = 'DirectRunner') 


# In[16]:


print 'gs://{0}/taxifare/ch4/taxi_preproc/'.format(BUCKET)


# In[17]:


get_ipython().system('gsutil ls gs://$BUCKET/taxifare/ch4/taxi_preproc/')


# Run pipleline on cloud on a larger sample size.

# In[18]:


# preprocess(EVERY_N = 50 * 100, RUNNER = 'DataflowRunner')  ## time: 9 min 33 secs in the training environment
# #change first arg to None to preprocess full dataset

## changed to the same as locally:
preprocess(EVERY_N = 50 * 10000, RUNNER = 'DataflowRunner')  ## time: 7 min 8 secs

## changed to 100 times more than locally:
#preprocess(EVERY_N = 50 * 10000 * 100, RUNNER = 'DataflowRunner')  ## time: 6 min 59 sec

## changed to 10000 times more than locally:
#preprocess(EVERY_N = 50 * 10000 * 10000, RUNNER = 'DataflowRunner')  ## time: ?

## but something went wrong: the number of elements processed ('added') in
## dataflow seems way to low...


# Once the job completes, observe the files created in Google Cloud Storage

# In[19]:


get_ipython().run_line_magic('bash', '')
gsutil ls -l gs://$BUCKET/taxifare/ch4/taxi_preproc/


# In[30]:


get_ipython().run_line_magic('bash', '')
#print first 10 lines of first shard of train.csv
gsutil cat "gs://$BUCKET/taxifare/ch4/taxi_preproc/train.csv-00000-of-*" | head


# <h2> 4. Develop model with new inputs </h2>
# 
# Download the first shard of the preprocessed data to enable local development.

# In[34]:


get_ipython().system('ls -l ./sample')


# In[35]:


get_ipython().run_line_magic('bash', '')
mkdir sample
#gsutil cp "gs://$BUCKET/taxifare/ch4/taxi_preproc/train.csv-00000-of-*" sample/train.csv
gsutil cp "gs://$BUCKET/taxifare/ch4/taxi_preproc/train.csv-00000-of-00001" sample/train.csv
#gsutil cp "gs://$BUCKET/taxifare/ch4/taxi_preproc/valid.csv-00000-of-*" sample/valid.csv
gsutil cp "gs://$BUCKET/taxifare/ch4/taxi_preproc/valid.csv-00000-of-00001" sample/valid.csv


# Complete the TODOs in taxifare/trainer/model.py so that the code below works.

# In[30]:


get_ipython().system('grep TODO taxifare/trainer/*.py')


# ```
# 04-feature-engineering/taxifeateng/labs/taxifare/trainer/model.py:    # TODO: Define feature columns for dayofweek, hourofday, pickuplon, pickuplat, dropofflat, d
# ropofflon, passengers
# 04-feature-engineering/taxifeateng/labs/taxifare/trainer/model.py:    # TODO: Add any engineered columns here
# 04-feature-engineering/taxifeateng/labs/taxifare/trainer/model.py:     TODO: Build an estimator starting from INPUT COLUMNS.
# 04-feature-engineering/taxifeateng/labs/taxifare/trainer/model.py:    return None # TODO: Add estimator definition here
# 04-feature-engineering/taxifeateng/labs/taxifare/trainer/model.py:    # TODO: Add any engineered features to the dict
# 04-feature-engineering/taxifeateng/labs/taxifare/trainer/model.py:        # TODO: What features will user provide? What will their types be?
# 04-feature-engineering/taxifeateng/labs/taxifare/trainer/model.py:    # TODO: Add any extra placeholders for inputs you'll generate
# 04-feature-engineering/taxifeateng/labs/taxifare/trainer/model.py:      features, # TODO: Wrap this with a call to add_engineered
# 04-feature-engineering/taxifeateng/labs/taxifare/trainer/model.py:            return features, label # TODO: Wrap this with a call to add_engineered
# ```

# We have two new inputs in the INPUT_ColumNS, three engineered features, and the estimator involves bucketization and feature crosses:
# 

# In[22]:


get_ipython().system('grep -A 20 "INPUT_COLUMNS = " taxifare/trainer/model.py')


# In[23]:


get_ipython().system('grep -A 20 "build_estimator" taxifare/trainer/model.py')


# In[24]:


get_ipython().system('grep -A 20 "add_engineered(" taxifare/trainer/model.py')


# In[36]:


get_ipython().run_line_magic('bash', '')
rm -rf taxifare.tar.gz taxi_trained
export PYTHONPATH=${PYTHONPATH}:${PWD}/taxifare
python -m trainer.task   --train_data_paths=${PWD}/sample/train.csv   --eval_data_paths=${PWD}/sample/valid.csv    --output_dir=${PWD}/taxi_trained   --train_steps=1000   --job-dir=/tmp


# In[37]:


get_ipython().system('ls taxi_trained/export/exporter/')


# In[38]:


get_ipython().run_line_magic('writefile', '/tmp/test.json')
{"dayofweek": "Sun", "hourofday": 17, "pickuplon": -73.885262, "pickuplat": 40.773008, "dropofflon": -73.987232, "dropofflat": 40.732403, "passengers": 2}


# In[39]:


get_ipython().run_line_magic('bash', '')
model_dir=$(ls ${PWD}/taxi_trained/export/exporter)
gcloud ml-engine local predict   --model-dir=${PWD}/taxi_trained/export/exporter/${model_dir}   --json-instances=/tmp/test.json


# In[ ]:


#if gcloud ml-engine local predict fails, might need to update glcoud
#!gcloud --quiet components update


# <h2> 5. Train on cloud </h2>
# 

# In[40]:


get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/taxifare/ch4/taxi_trained\nJOBNAME=lab4a_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n  --region=$REGION \\\n  --module-name=trainer.task \\\n  --package-path=${PWD}/taxifare/trainer \\\n  --job-dir=$OUTDIR \\\n  --staging-bucket=gs://$BUCKET \\\n  --scale-tier=BASIC \\\n  --runtime-version=$TFVERSION \\\n  -- \\\n  --train_data_paths="gs://$BUCKET/taxifare/ch4/taxi_preproc/train*" \\\n  --eval_data_paths="gs://${BUCKET}/taxifare/ch4/taxi_preproc/valid*"  \\\n  --train_steps=5000 \\\n  --output_dir=$OUTDIR')


# <h2> 6. Inspect with TensorBoard </h2>
# 

# In[41]:


from google.datalab.ml import TensorBoard
OUTDIR='gs://{0}/taxifare/ch4/taxi_trained'.format(BUCKET)
print OUTDIR
TensorBoard().start(OUTDIR)


# In[53]:


print(OUTDIR)


# In[50]:


get_ipython().system('ls -l ./taxifare/trainer')


# In[52]:


import taxifare.trainer.model


# What is your RMSE?

# Copyright 2016 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License