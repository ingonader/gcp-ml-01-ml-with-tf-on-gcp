
# coding: utf-8

# <h1> Exploring tf.transform </h1>
# 
# While Pandas is fine for experimenting, for operationalization of your workflow, it is better to do preprocessing in Apache Beam. This will also help if you need to preprocess data in flight, since Apache Beam also allows for streaming.
# 
# Only specific combinations of TensorFlow/Beam are supported by tf.transform. So make sure to get a combo that is.
# 
# * TFT 0.8.0
# * TF 1.8 or higher
# * Apache Beam [GCP] 2.5.0 or higher

# In[1]:


get_ipython().run_cell_magic('bash', '', 'source activate py2env\npip uninstall -y google-cloud-dataflow\nconda install -y pytz==2018.4\nconda install -y pyopenssl\npip install apache-beam[gcp] tensorflow_transform==0.8.0')


# <b>Restart the kernel</b> after you do a pip install (click on the <b>Reset</b> button in Datalab)

# In[1]:


get_ipython().run_line_magic('bash', '')
pip freeze | grep -e 'flow\|beam'


# In[2]:


import tensorflow as tf
import tensorflow_transform as tft
import shutil
print(tf.__version__)


# In[3]:


# change these to try this notebook out
BUCKET = 'cloud-training-demos-ml'
PROJECT = 'cloud-training-demos'
REGION = 'us-central1'


# In[7]:


## in datalabvm cloud datalab notebook:
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


# In[8]:


get_ipython().run_line_magic('bash', '')
gcloud config set project $PROJECT
gcloud config set compute/region $REGION


# In[9]:


get_ipython().run_cell_magic('bash', '', 'if ! gsutil ls | grep -q gs://${BUCKET}/; then\n  gsutil mb -l ${REGION} gs://${BUCKET}\nfi')


# ## Input source: BigQuery
# 
# Get data from BigQuery but defer filtering etc. to Beam.
# Note that the dayofweek column is now strings.

# In[11]:


import google.datalab.bigquery as bq
def create_query(phase, EVERY_N):
  """
  phase: 1=train 2=valid
  """
  base_query = """
WITH daynames AS
  (SELECT ['Sun', 'Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat'] AS daysofweek)
SELECT
  (tolls_amount + fare_amount) AS fare_amount,
  daysofweek[ORDINAL(EXTRACT(DAYOFWEEK FROM pickup_datetime))] AS dayofweek,
  EXTRACT(HOUR FROM pickup_datetime) AS hourofday,
  pickup_longitude AS pickuplon,
  pickup_latitude AS pickuplat,
  dropoff_longitude AS dropofflon,
  dropoff_latitude AS dropofflat,
  passenger_count AS passengers,
  'notneeded' AS key
FROM
  `nyc-tlc.yellow.trips`, daynames
WHERE
  trip_distance > 0 AND fare_amount > 0
  """

  if EVERY_N == None:
    if phase < 2:
      # training
      query = "{0} AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING)),4) < 2".format(base_query)
    else:
      query = "{0} AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING)),4) = {1}".format(base_query, phase)
  else:
      query = "{0} AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))),{1}) = {2}".format(base_query, EVERY_N, phase)
    
  return query

query = create_query(2, 100000)
print(query)


# In[12]:


df_valid = bq.Query(query).execute().result().to_dataframe()
display(df_valid.head())
df_valid.describe()


# ## Create ML dataset using tf.transform and Dataflow
# 
# Let's use Cloud Dataflow to read in the BigQuery data and write it out as CSV files. Along the way, let's use tf.transform to do scaling and transforming. Using tf.transform allows us to save the metadata to ensure that the appropriate transformations get carried out during prediction as well.

# In[13]:


get_ipython().run_line_magic('writefile', 'requirements.txt')
tensorflow-transform==0.8.0


# Test transform_data is type pcollection. test if _ = is neccesary

# In[14]:


import datetime
import tensorflow as tf
import apache_beam as beam
import tensorflow_transform as tft
from tensorflow_transform.beam import impl as beam_impl

def is_valid(inputs):
  try:
    pickup_longitude = inputs['pickuplon']
    dropoff_longitude = inputs['dropofflon']
    pickup_latitude = inputs['pickuplat']
    dropoff_latitude = inputs['dropofflat']
    hourofday = inputs['hourofday']
    dayofweek = inputs['dayofweek']
    passenger_count = inputs['passengers']
    fare_amount = inputs['fare_amount']
    return (fare_amount >= 2.5 and pickup_longitude > -78 and pickup_longitude < -70       and dropoff_longitude > -78 and dropoff_longitude < -70 and pickup_latitude > 37       and pickup_latitude < 45 and dropoff_latitude > 37 and dropoff_latitude < 45       and passenger_count > 0)
  except:
    return False
  
def preprocess_tft(inputs):
      import datetime   
      print inputs
      result = {}
      result['fare_amount'] = tf.identity(inputs['fare_amount'])     
      result['dayofweek'] = tft.string_to_int(inputs['dayofweek']) # builds a vocabulary
      result['hourofday'] = tf.identity(inputs['hourofday']) # pass through
      result['pickuplon'] = (tft.scale_to_0_1(inputs['pickuplon'])) # scaling numeric values
      result['pickuplat'] = (tft.scale_to_0_1(inputs['pickuplat']))
      result['dropofflon'] = (tft.scale_to_0_1(inputs['dropofflon']))
      result['dropofflat'] = (tft.scale_to_0_1(inputs['dropofflat']))
      result['passengers'] = tf.cast(inputs['passengers'], tf.float32) # a cast
      result['key'] = tf.as_string(tf.ones_like(inputs['passengers'])) # arbitrary TF func
      # engineered features
      latdiff = inputs['pickuplat'] - inputs['dropofflat']
      londiff = inputs['pickuplon'] - inputs['dropofflon']
      result['latdiff'] = tft.scale_to_0_1(latdiff)
      result['londiff'] = tft.scale_to_0_1(londiff)
      dist = tf.sqrt(latdiff * latdiff + londiff * londiff)
      result['euclidean'] = dist
      return result

def preprocess(in_test_mode):
  import os
  import os.path
  import tempfile
  from apache_beam.io import tfrecordio
  from tensorflow_transform.coders import example_proto_coder
  from tensorflow_transform.tf_metadata import dataset_metadata
  from tensorflow_transform.tf_metadata import dataset_schema
  from tensorflow_transform.beam import tft_beam_io
  from tensorflow_transform.beam.tft_beam_io import transform_fn_io

  job_name = 'preprocess-taxi-features' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')    
  if in_test_mode:
    import shutil
    print 'Launching local job ... hang on'
    OUTPUT_DIR = './preproc_tft'
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    EVERY_N = 100000
  else:
    print 'Launching Dataflow job {} ... hang on'.format(job_name)
    OUTPUT_DIR = 'gs://{0}/taxifare/preproc_tft/'.format(BUCKET)
    import subprocess
    subprocess.call('gsutil rm -r {}'.format(OUTPUT_DIR).split())
    EVERY_N = 10000
    
  options = {
    'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
    'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
    'job_name': job_name,
    'project': PROJECT,
    'max_num_workers': 24,
    'teardown_policy': 'TEARDOWN_ALWAYS',
    'no_save_main_session': True,
    'requirements_file': 'requirements.txt'
  }
  opts = beam.pipeline.PipelineOptions(flags=[], **options)
  if in_test_mode:
    RUNNER = 'DirectRunner'
  else:
    RUNNER = 'DataflowRunner'

  # set up raw data metadata
  raw_data_schema = {
    colname : dataset_schema.ColumnSchema(tf.string, [], dataset_schema.FixedColumnRepresentation())
                   for colname in 'dayofweek,key'.split(',')
  }
  raw_data_schema.update({
      colname : dataset_schema.ColumnSchema(tf.float32, [], dataset_schema.FixedColumnRepresentation())
                   for colname in 'fare_amount,pickuplon,pickuplat,dropofflon,dropofflat'.split(',')
    })
  raw_data_schema.update({
      colname : dataset_schema.ColumnSchema(tf.int64, [], dataset_schema.FixedColumnRepresentation())
                   for colname in 'hourofday,passengers'.split(',')
    })
  raw_data_metadata = dataset_metadata.DatasetMetadata(dataset_schema.Schema(raw_data_schema))

  # run Beam  
  with beam.Pipeline(RUNNER, options=opts) as p:
    with beam_impl.Context(temp_dir=os.path.join(OUTPUT_DIR, 'tmp')):
      # save the raw data metadata
      raw_data_metadata | 'WriteInputMetadata' >> tft_beam_io.WriteMetadata(
            os.path.join(OUTPUT_DIR, 'metadata/rawdata_metadata'),
            pipeline=p)
      
      # read training data from bigquery and filter rows     
      raw_data = (p 
        | 'train_read' >> beam.io.Read(beam.io.BigQuerySource(query=create_query(1, EVERY_N), use_standard_sql=True))
        | 'train_filter' >> beam.Filter(is_valid))
      raw_dataset = (raw_data, raw_data_metadata)
      
      # analyze and transform training data
      transformed_dataset, transform_fn = (
          raw_dataset | beam_impl.AnalyzeAndTransformDataset(preprocess_tft))
      transformed_data, transformed_metadata = transformed_dataset
      
      # save transformed training data to disk in efficient tfrecord format
      transformed_data | 'WriteTrainData' >> tfrecordio.WriteToTFRecord(
          os.path.join(OUTPUT_DIR, 'train'),
          file_name_suffix='.gz',
          coder=example_proto_coder.ExampleProtoCoder(
              transformed_metadata.schema))
      
      # read eval data from bigquery and filter rows  
      raw_test_data = (p 
        | 'eval_read' >> beam.io.Read(beam.io.BigQuerySource(query=create_query(2, EVERY_N), use_standard_sql=True))
        | 'eval_filter' >> beam.Filter(is_valid))
      raw_test_dataset = (raw_test_data, raw_data_metadata)
      
      # transform eval data
      transformed_test_dataset = (
          (raw_test_dataset, transform_fn) | beam_impl.TransformDataset())
      transformed_test_data, _ = transformed_test_dataset
      
      # save transformed training data to disk in efficient tfrecord format
      transformed_test_data | 'WriteTestData' >> tfrecordio.WriteToTFRecord(
          os.path.join(OUTPUT_DIR, 'eval'),
          file_name_suffix='.gz',
          coder=example_proto_coder.ExampleProtoCoder(
              transformed_metadata.schema))
      
      # save transformation function to disk for use at serving time
      transform_fn | 'WriteTransformFn' >> transform_fn_io.WriteTransformFn(
          os.path.join(OUTPUT_DIR, 'metadata'))

preprocess(in_test_mode=False) # change to True to run locally


# In[15]:


get_ipython().run_line_magic('bash', '')
# ls preproc_tft
gsutil ls gs://${BUCKET}/taxifare/preproc_tft/


# <h2> Train off preprocessed data </h2>

# In[16]:


get_ipython().run_line_magic('bash', '')
rm -rf taxifare_tft.tar.gz taxi_trained
export PYTHONPATH=${PYTHONPATH}:$PWD/taxifare_tft
python -m trainer.task    --train_data_paths="gs://${BUCKET}/taxifare/preproc_tft/train*"    --eval_data_paths="gs://${BUCKET}/taxifare/preproc_tft/eval*"     --output_dir=./taxi_trained    --train_steps=10 --job-dir=/tmp    --metadata_path=gs://${BUCKET}/taxifare/preproc_tft/metadata


# In[17]:


get_ipython().system('ls $PWD/taxi_trained/export/exporter')


# In[18]:


get_ipython().run_line_magic('writefile', '/tmp/test.json')
{"dayofweek":"Thu","hourofday":17,"pickuplon": -73.885262,"pickuplat": 40.773008,"dropofflon": -73.987232,"dropofflat": 40.732403,"passengers": 2}


# In[19]:


get_ipython().run_line_magic('bash', '')
model_dir=$(ls $PWD/taxi_trained/export/exporter/)
gcloud ml-engine local predict     --model-dir=./taxi_trained/export/exporter/${model_dir}     --json-instances=/tmp/test.json


# Copyright 2016-2018 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License