
# coding: utf-8

# <h1> Repeatable splitting </h1>
# 
# In this notebook, we will explore the impact of different ways of creating machine learning datasets.
# 
# <p>
# 
# Repeatability is important in machine learning. If you do the same thing now and 5 minutes from now and get different answers, then it makes experimentation difficult. In other words, you will find it difficult to gauge whether a change you made has resulted in an improvement or not.

# In[1]:


import google.datalab.bigquery as bq


# <h3> Create a simple machine learning model </h3>
# 
# The dataset that we will use is <a href="https://bigquery.cloud.google.com/table/bigquery-samples:airline_ontime_data.flights">a BigQuery public dataset</a> of airline arrival data. Click on the link, and look at the column names. Switch to the Details tab to verify that the number of records is 70 million, and then switch to the Preview tab to look at a few rows.
# <p>
# We want to predict the arrival delay of an airline based on the departure delay. The model that we will use is a zero-bias linear model:
# $$ delay_{arrival} = \alpha * delay_{departure} $$
# <p>
# To train the model is to estimate a good value for $\alpha$. 
# <p>
# One approach to estimate alpha is to use this formula:
# $$ \alpha = \frac{\sum delay_{departure} delay_{arrival} }{  \sum delay_{departure}^2 } $$
# Because we'd like to capture the idea that this relationship is different for flights from New York to Los Angeles vs. flights from Austin to Indianapolis (shorter flight, less busy airports), we'd compute a different $alpha$ for each airport-pair.  For simplicity, we'll do this model only for flights between Denver and Los Angeles.

# <h2> Naive random split (not repeatable) </h2>

# In[2]:


compute_alpha = """
#standardSQL
SELECT 
   SAFE_DIVIDE(SUM(arrival_delay * departure_delay), SUM(departure_delay * departure_delay)) AS alpha
FROM
(
   SELECT RAND() AS splitfield,
   arrival_delay,
   departure_delay
FROM
  `bigquery-samples.airline_ontime_data.flights`
WHERE
  departure_airport = 'DEN' AND arrival_airport = 'LAX'
)
WHERE
  splitfield < 0.8
"""


# In[3]:


results = bq.Query(compute_alpha).execute().result().to_dataframe()
alpha = results['alpha'][0]
print(alpha)


# <h3> What is wrong with calculating RMSE on the training and test data as follows? </h3>

# In[6]:


compute_rmse = """
#standardSQL
SELECT
  dataset,
  SQRT(AVG((arrival_delay - ALPHA * departure_delay)*(arrival_delay - ALPHA * departure_delay))) AS rmse,
  COUNT(arrival_delay) AS num_flights
FROM (
  SELECT
    IF (RAND() < 0.8, 'train', 'eval') AS dataset,
    arrival_delay,
    departure_delay
  FROM
    `bigquery-samples.airline_ontime_data.flights`
  WHERE
    departure_airport = 'DEN'
    AND arrival_airport = 'LAX' )
GROUP BY
  dataset
"""
bq.Query(compute_rmse.replace('ALPHA', str(alpha))).execute().result().to_dataframe()


# Hint:
# * Are you really getting the same training data in the compute_rmse query as in the compute_alpha query?
# * Do you get the same answers each time you rerun the compute_alpha and compute_rmse blocks?

# <h3> How do we correctly train and evaluate? </h3>
# <br/>
# Here's the right way to compute the RMSE using the actual training and held-out (evaluation) data. Note how much harder this feels.
# 
# Although the calculations are now correct, the experiment is still not repeatable.
# 
# Try running it several times; do you get the same answer?

# In[10]:


train_and_eval_rand = """
#standardSQL
WITH
  alldata AS (
  SELECT
    IF (RAND() < 0.8,
      'train',
      'eval') AS dataset,
    arrival_delay,
    departure_delay
  FROM
    `bigquery-samples.airline_ontime_data.flights`
  WHERE
    departure_airport = 'DEN'
    AND arrival_airport = 'LAX' ),
  training AS (
  SELECT
    SAFE_DIVIDE( SUM(arrival_delay * departure_delay) , SUM(departure_delay * departure_delay)) AS alpha
  FROM
    alldata
  WHERE
    dataset = 'train' )
SELECT
  MAX(alpha) AS alpha,
  dataset,
  SQRT(AVG((arrival_delay - alpha * departure_delay)*(arrival_delay - alpha * departure_delay))) AS rmse,
  COUNT(arrival_delay) AS num_flights
FROM
  alldata,
  training
GROUP BY
  dataset
ORDER BY dataset
"""


# In[11]:


bq.Query(train_and_eval_rand).execute().result().to_dataframe()


# In[12]:


bq.Query(train_and_eval_rand).execute().result().to_dataframe()


# <h2> Using HASH of date to split the data </h2>
# 
# Let's split by date and train.

# In[18]:


explain_hash = """
#standardSQL
SELECT 
   date,
   FARM_FINGERPRINT(date) as farmfp,
   ABS(FARM_FINGERPRINT(date)) as abs_farmfp,
   MOD(ABS(FARM_FINGERPRINT(date)), 10) as mod_abs_farmfp
FROM
  `bigquery-samples.airline_ontime_data.flights`
WHERE
  departure_airport = 'DEN' AND arrival_airport = 'LAX'
  AND MOD(ABS(FARM_FINGERPRINT(date)), 10) < 8
LIMIT 10
"""


# In[19]:


bq.Query(explain_hash).execute().result().to_dataframe()


# In[13]:


compute_alpha = """
#standardSQL
SELECT 
   SAFE_DIVIDE(SUM(arrival_delay * departure_delay), SUM(departure_delay * departure_delay)) AS alpha
FROM
  `bigquery-samples.airline_ontime_data.flights`
WHERE
  departure_airport = 'DEN' AND arrival_airport = 'LAX'
  AND MOD(ABS(FARM_FINGERPRINT(date)), 10) < 8
"""
results = bq.Query(compute_alpha).execute().result().to_dataframe()
alpha = results['alpha'][0]
print(alpha)


# We can now use the alpha to compute RMSE. Because the alpha value is repeatable, we don't need to worry that the alpha in the compute_rmse will be different from the alpha computed in the compute_alpha.

# In[15]:


compute_rmse = """
#standardSQL
SELECT
  IF(MOD(ABS(FARM_FINGERPRINT(date)), 10) < 8, 'train', 'eval') AS dataset,
  SQRT(AVG((arrival_delay - ALPHA * departure_delay)*(arrival_delay - ALPHA * departure_delay))) AS rmse,
  COUNT(arrival_delay) AS num_flights
FROM
    `bigquery-samples.airline_ontime_data.flights`
WHERE
    departure_airport = 'DEN'
    AND arrival_airport = 'LAX'
GROUP BY
  dataset
"""
bq.Query(compute_rmse.replace('ALPHA', str(alpha))).execute().result().to_dataframe()


# Note also that the RMSE on the evaluation dataset more from the RMSE on the training dataset when we do the split correctly.  This should be expected; in the RAND() case, there was leakage between training and evaluation datasets, because there is high correlation between flights on the same day.
# <p>
# This is one of the biggest dangers with doing machine learning splits the wrong way -- <b> you will develop a false sense of confidence in how good your model is! </b>

# Copyright 2018 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.