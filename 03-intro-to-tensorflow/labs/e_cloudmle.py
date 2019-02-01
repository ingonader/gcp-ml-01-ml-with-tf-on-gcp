
# coding: utf-8

# <h1> Scaling up ML using Cloud ML Engine </h1>
# 
# In this notebook, we take a previously developed TensorFlow model to predict taxifare rides and package it up so that it can be run in Cloud MLE. For now, we'll run this on a small dataset. The model that was developed is rather simplistic, and therefore, the accuracy of the model is not great either.  However, this notebook illustrates *how* to package up a TensorFlow model to run it within Cloud ML. 
# 
# Later in the course, we will look at ways to make a more effective machine learning model.

# <h2> Environment variables for project and bucket </h2>
# 
# Note that:
# <ol>
# <li> Your project id is the *unique* string that identifies your project (not the project name). You can find this from the GCP Console dashboard's Home page.  My dashboard reads:  <b>Project ID:</b> cloud-training-demos </li>
# <li> Cloud training often involves saving and restoring model files. If you don't have a bucket already, I suggest that you create one from the GCP console (because it will dynamically check whether the bucket name you want is available). A common pattern is to prefix the bucket name by the project id, so that it is unique. Also, for cost reasons, you might want to use a single region bucket. </li>
# </ol>
# <b>Change the cell below</b> to reflect your Project ID and bucket name.
# 

# In[7]:


import os
output = os.popen("gcloud config get-value project").readlines()
project_name = output[0][:-1]

# change these to try this notebook out
PROJECT = project_name
BUCKET = project_name
BUCKET = project_name #BUCKET.replace("qwiklabs-gcp-", "inna-bckt-")
REGION = 'europe-west1'  ## note: Cloud ML Engine not availabe in europe-west3!

print(PROJECT)
print(BUCKET)
print("gsutil mb -l {0} gs://{1}".format(REGION, BUCKET))

## set config for gcp config: [[?]]
print(os.popen("gcloud config set project $PROJECT").readlines())
print(os.popen("gcloud config set compute/region $REGION").readlines())


# In[8]:


# for bash
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = '1.8'  # Tensorflow version


# In[9]:


get_ipython().run_line_magic('bash', '')
gcloud config set project $PROJECT
gcloud config set compute/region $REGION


# Allow the Cloud ML Engine service account to read/write to the bucket containing training data.

# In[10]:


get_ipython().run_line_magic('bash', '')
PROJECT_ID=$PROJECT
AUTH_TOKEN=$(gcloud auth print-access-token)
SVC_ACCOUNT=$(curl -X GET -H "Content-Type: application/json"     -H "Authorization: Bearer $AUTH_TOKEN"     https://ml.googleapis.com/v1/projects/${PROJECT_ID}:getConfig     | python -c "import json; import sys; response = json.load(sys.stdin);     print(response['serviceAccount'])")

echo "Authorizing the Cloud ML Service account $SVC_ACCOUNT to access files in $BUCKET"
gsutil -m defacl ch -u $SVC_ACCOUNT:R gs://$BUCKET
gsutil -m acl ch -u $SVC_ACCOUNT:R -r gs://$BUCKET  # error message (if bucket is empty) can be ignored
gsutil -m acl ch -u $SVC_ACCOUNT:W gs://$BUCKET


# <h2> Packaging up the code </h2>
# 
# Take your code and put into a standard Python package structure.  <a href="taxifare/trainer/model.py">model.py</a> and <a href="taxifare/trainer/task.py">task.py</a> containing the Tensorflow code from earlier (explore the <a href="taxifare/trainer/">directory structure</a>).

# In[10]:


get_ipython().run_line_magic('bash', '')
# TODO: Make sure that model is complete and has no remaining TODOs
grep TODO taxifare/trainer/*.py


# <h2> Find absolute paths to your data </h2>

# Note the absolute paths below. /content is mapped in Datalab to where the home icon takes you

# In[11]:


get_ipython().run_line_magic('bash', '')
echo $PWD
rm -rf $PWD/taxi_trained
head -1 $PWD/taxi-train.csv
head -1 $PWD/taxi-valid.csv


# <h2> Running the Python module from the command-line </h2>

# <h4> Monitor using Tensorboard </h4>

# In[13]:


from google.datalab.ml import TensorBoard
TensorBoard().start('./taxi_trained')


# In[17]:


get_ipython().run_line_magic('bash', '')
rm -rf taxifare.tar.gz taxi_trained
export PYTHONPATH=${PYTHONPATH}:${PWD}/taxifare
python -m trainer.task    --train_data_paths="${PWD}/taxi-train*"    --eval_data_paths=${PWD}/taxi-valid.csv     --output_dir=${PWD}/taxi_trained    --train_steps=100 --job-dir=./tmp


# In[18]:


get_ipython().run_line_magic('bash', '')
ls $PWD/taxi_trained/export/exporter/


# In[19]:


get_ipython().run_line_magic('writefile', './test.json')
{"pickuplon": -73.885262,"pickuplat": 40.773008,"dropofflon": -73.987232,"dropofflat": 40.732403,"passengers": 2}


# In[20]:


get_ipython().run_line_magic('bash', '')
model_dir=$(ls ${PWD}/taxi_trained/export/exporter)
gcloud ml-engine local predict     --model-dir=${PWD}/taxi_trained/export/exporter/${model_dir}     --json-instances=./test.json


# <h2> Running locally using gcloud </h2>

# In[21]:


get_ipython().run_line_magic('bash', '')
rm -rf taxifare.tar.gz taxi_trained
gcloud ml-engine local train    --module-name=trainer.task    --package-path=${PWD}/taxifare/trainer    --    --train_data_paths=${PWD}/taxi-train.csv    --eval_data_paths=${PWD}/taxi-valid.csv     --train_steps=1000    --output_dir=${PWD}/taxi_trained 


# When I ran it (due to random seeds, your results will be different), the ```average_loss``` (Mean Squared Error) on the evaluation dataset was 187, meaning that the RMSE was around 13.

# In[22]:


for pid in TensorBoard.list()['pid']:
  TensorBoard().stop(pid)
  print 'Stopped TensorBoard with pid {}'.format(pid)


# If the above step (to stop TensorBoard) appears stalled, just move on to the next step. You don't need to wait for it to return.

# In[23]:


get_ipython().system('ls $PWD/taxi_trained')


# <h2> Submit training job using gcloud </h2>
# 
# First copy the training data to the cloud.  Then, launch a training job.
# 
# After you submit the job, go to the cloud console (http://console.cloud.google.com) and select <b>Machine Learning | Jobs</b> to monitor progress.  
# 
# <b>Note:</b> Don't be concerned if the notebook stalls (with a blue progress bar) or returns with an error about being unable to refresh auth tokens. This is a long-lived Cloud job and work is going on in the cloud.  Use the Cloud Console link (above) to monitor the job.

# In[24]:


get_ipython().run_line_magic('bash', '')
echo $BUCKET
gsutil -m rm -rf gs://${BUCKET}/taxifare/smallinput/
gsutil -m cp ${PWD}/*.csv gs://${BUCKET}/taxifare/smallinput/


# In[25]:


get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/taxifare/smallinput/taxi_trained\nJOBNAME=lab3a_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n   --region=$REGION \\\n   --module-name=trainer.task \\\n   --package-path=${PWD}/taxifare/trainer \\\n   --job-dir=$OUTDIR \\\n   --staging-bucket=gs://$BUCKET \\\n   --scale-tier=BASIC \\\n   --runtime-version=$TFVERSION \\\n   -- \\\n   --train_data_paths="gs://${BUCKET}/taxifare/smallinput/taxi-train*" \\\n   --eval_data_paths="gs://${BUCKET}/taxifare/smallinput/taxi-valid*"  \\\n   --output_dir=$OUTDIR \\\n   --train_steps=10000')


# Don't be concerned if the notebook appears stalled (with a blue progress bar) or returns with an error about being unable to refresh auth tokens. This is a long-lived Cloud job and work is going on in the cloud. 
# 
# <b>Use the Cloud Console link to monitor the job and do NOT proceed until the job is done.</b>

# In[26]:


get_ipython().system('gcloud ml-engine jobs describe lab3a_190201_094123')


# <h2> Deploy model </h2>
# 
# Find out the actual name of the subdirectory where the model is stored and use it to deploy the model.  Deploying model will take up to <b>5 minutes</b>.

# In[27]:


get_ipython().run_line_magic('bash', '')
gsutil ls gs://${BUCKET}/taxifare/smallinput/taxi_trained/export/exporter


# In[28]:


get_ipython().run_line_magic('bash', '')
MODEL_NAME="taxifare"
MODEL_VERSION="v1"
MODEL_LOCATION=$(gsutil ls gs://${BUCKET}/taxifare/smallinput/taxi_trained/export/exporter | tail -1)
echo "Run these commands one-by-one (the very first time, you'll create a model and then create a version)"
#gcloud ml-engine versions delete ${MODEL_VERSION} --model ${MODEL_NAME}
#gcloud ml-engine models delete ${MODEL_NAME}
gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version $TFVERSION


# <h2> Prediction </h2>

# In[29]:


get_ipython().run_line_magic('bash', '')
gcloud ml-engine predict --model=taxifare --version=v1 --json-instances=./test.json


# In[30]:


from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import json

credentials = GoogleCredentials.get_application_default()
api = discovery.build('ml', 'v1', credentials=credentials,
            discoveryServiceUrl='https://storage.googleapis.com/cloud-ml/discovery/ml_v1_discovery.json')

request_data = {'instances':
  [
      {
        'pickuplon': -73.885262,
        'pickuplat': 40.773008,
        'dropofflon': -73.987232,
        'dropofflat': 40.732403,
        'passengers': 2,
      }
  ]
}

parent = 'projects/%s/models/%s/versions/%s' % (PROJECT, 'taxifare', 'v1')
response = api.projects().predict(body=request_data, name=parent).execute()
print "response={0}".format(response)


# <h2> Train on larger dataset </h2>
# 
# I have already followed the steps below and the files are already available. <b> You don't need to do the steps in this comment. </b> In the next chapter (on feature engineering), we will avoid all this manual processing by using Cloud Dataflow.
# 
# Go to http://bigquery.cloud.google.com/ and type the query:
# <pre>
# SELECT
#   (tolls_amount + fare_amount) AS fare_amount,
#   pickup_longitude AS pickuplon,
#   pickup_latitude AS pickuplat,
#   dropoff_longitude AS dropofflon,
#   dropoff_latitude AS dropofflat,
#   passenger_count*1.0 AS passengers,
#   'nokeyindata' AS key
# FROM
#   [nyc-tlc:yellow.trips]
# WHERE
#   trip_distance > 0
#   AND fare_amount >= 2.5
#   AND pickup_longitude > -78
#   AND pickup_longitude < -70
#   AND dropoff_longitude > -78
#   AND dropoff_longitude < -70
#   AND pickup_latitude > 37
#   AND pickup_latitude < 45
#   AND dropoff_latitude > 37
#   AND dropoff_latitude < 45
#   AND passenger_count > 0
#   AND ABS(HASH(pickup_datetime)) % 1000 == 1
# </pre>
# 
# Note that this is now 1,000,000 rows (i.e. 100x the original dataset).  Export this to CSV using the following steps (Note that <b>I have already done this and made the resulting GCS data publicly available</b>, so you don't need to do it.):
# <ol>
# <li> Click on the "Save As Table" button and note down the name of the dataset and table.
# <li> On the BigQuery console, find the newly exported table in the left-hand-side menu, and click on the name.
# <li> Click on "Export Table"
# <li> Supply your bucket name and give it the name train.csv (for example: gs://cloud-training-demos-ml/taxifare/ch3/train.csv). Note down what this is.  Wait for the job to finish (look at the "Job History" on the left-hand-side menu)
# <li> In the query above, change the final "== 1" to "== 2" and export this to Cloud Storage as valid.csv (e.g.  gs://cloud-training-demos-ml/taxifare/ch3/valid.csv)
# <li> Download the two files, remove the header line and upload it back to GCS.
# </ol>
# 
# <p/>
# <p/>
# 
# <h2> Run Cloud training on 1-million row dataset </h2>
# 
# This took 60 minutes and uses as input 1-million rows.  The model is exactly the same as above. The only changes are to the input (to use the larger dataset) and to the Cloud MLE tier (to use STANDARD_1 instead of BASIC -- STANDARD_1 is approximately 10x more powerful than BASIC).  At the end of the training the loss was 32, but the RMSE (calculated on the validation dataset) was stubbornly at 9.03. So, simply adding more data doesn't help.

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\nXXXXX  this takes 60 minutes. if you are sure you want to run it, then remove this line.\n\nOUTDIR=gs://${BUCKET}/taxifare/ch3/taxi_trained\nJOBNAME=lab3a_$(date -u +%y%m%d_%H%M%S)\nCRS_BUCKET=cloud-training-demos # use the already exported data\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n   --region=$REGION \\\n   --module-name=trainer.task \\\n   --package-path=${PWD}/taxifare/trainer \\\n   --job-dir=$OUTDIR \\\n   --staging-bucket=gs://$BUCKET \\\n   --scale-tier=STANDARD_1 \\\n   --runtime-version=$TFVERSION \\\n   -- \\\n   --train_data_paths="gs://${CRS_BUCKET}/taxifare/ch3/train.csv" \\\n   --eval_data_paths="gs://${CRS_BUCKET}/taxifare/ch3/valid.csv"  \\\n   --output_dir=$OUTDIR \\\n   --train_steps=100000')


# ## Challenge Exercise
# 
# Modify your solution to the challenge exercise in d_trainandevaluate.ipynb appropriately. Make sure that you implement training and deployment. Increase the size of your dataset by 10x since you are running on the cloud. Does your accuracy improve?

# In[31]:


import datalab.bigquery as bq
import tensorflow as tf
import numpy as np
import shutil
print(tf.__version__)

from google.datalab.ml import TensorBoard

import math
import pandas as pd

def gen_data(n):
  r = np.random.uniform(.5, 2, n)
  h = np.random.uniform(.5, 2, n)
  v = r ** 2 * h * math.pi
  dat = pd.DataFrame({
    'r': np.round(r, 1),
    'h': np.round(h, 1),
    'v': np.round(v, 1)
  })
  return dat


dat = gen_data(5000)
dat_eval = gen_data(1000)
dat_test = gen_data(1000)

dat.head(n = 2)


# In[38]:


## write data to file:
dat.to_csv('dat-cyl-train.csv', header = False, index = False)
dat_eval.to_csv('dat-cyl-eval.csv', header = False, index = False)
dat_test.to_csv('dat-cyl-test.csv', header = False, index = False)


# In[40]:


get_ipython().system('ls -l dat-cyl*.csv')


# In[41]:


get_ipython().run_line_magic('bash', '')
echo $PWD
rm -rf $PWD/cyl_trained
head -1 $PWD/dat-cyl-train.csv
head -1 $PWD/dat-cyl-eval.csv


# ## Running locally using Python

# In[42]:


get_ipython().run_line_magic('bash', '')
rm -rf labs/cyl.tar.gz labs/cyl_trained
export PYTHONPATH=${PYTHONPATH}:${PWD}/cyl
python -m trainer.task    --train_data_paths="${PWD}/dat-cyl-train*"    --eval_data_paths=${PWD}/dat-cyl-eval.csv     --output_dir=${PWD}/labs/cyl_trained    --train_steps=1000 --job-dir=./tmp


# In[43]:


get_ipython().run_line_magic('bash', '')
ls $PWD/taxi_trained/export/exporter/


# In[44]:


get_ipython().run_line_magic('writefile', './test.json')
{"h": 1.0,"r": 2.0}


# In[50]:


get_ipython().run_line_magic('bash', '')
model_dir=$(ls ${PWD}/labs/cyl_trained/export/exporter)


# In[52]:


get_ipython().run_line_magic('bash', '')
model_dir=$(ls ${PWD}/labs/cyl_trained/export/exporter)

gcloud ml-engine local predict     --model-dir=${PWD}/labs/cyl_trained/export/exporter/${model_dir}     --json-instances=./test.json


# ## Running locally using gcloud

# In[53]:


get_ipython().run_line_magic('bash', '')
rm -rf labs/cyl.tar.gz labs/cyl_trained
gcloud ml-engine local train    --module-name=trainer.task    --package-path=${PWD}/cyl/trainer    --    --train_data_paths=${PWD}/dat-cyl-train.csv    --eval_data_paths=${PWD}/dat-cyl-eval.csv     --train_steps=1000    --output_dir=${PWD}/labs/cyl_trained 


# ## Submit Training using gcloud

# In[54]:


get_ipython().run_line_magic('bash', '')
echo $BUCKET
gsutil -m rm -rf gs://${BUCKET}/cyl/smallinput/
gsutil -m cp ${PWD}/*.csv gs://${BUCKET}/cyl/smallinput/


# In[55]:


get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/cyl/smallinput/cyl_trained\nJOBNAME=lab3a_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\ngcloud ml-engine jobs submit training $JOBNAME \\\n   --region=$REGION \\\n   --module-name=trainer.task \\\n   --package-path=${PWD}/cyl/trainer \\\n   --job-dir=$OUTDIR \\\n   --staging-bucket=gs://$BUCKET \\\n   --scale-tier=BASIC \\\n   --runtime-version=$TFVERSION \\\n   -- \\\n   --train_data_paths="gs://${BUCKET}/cyl/smallinput/dat-cyl-train*" \\\n   --eval_data_paths="gs://${BUCKET}/cyl/smallinput/dat-cyl-eval*"  \\\n   --output_dir=$OUTDIR \\\n   --train_steps=10000')


# Copyright 2016 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License