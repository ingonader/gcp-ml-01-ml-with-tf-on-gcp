
# coding: utf-8

# <h1> Explore and create ML datasets </h1>
# 
# In this notebook, we will explore data corresponding to taxi rides in New York City to build a Machine Learning model in support of a fare-estimation tool. The idea is to suggest a likely fare to taxi riders so that they are not surprised, and so that they can protest if the charge is much higher than expected.
# 
# <div id="toc"></div>
# 
# Let's start off with the Python imports that we need.

# In[3]:


import google.datalab.bigquery as bq
import seaborn as sns
import pandas as pd
import numpy as np
import shutil


# <h3> Extract sample data from BigQuery </h3>
# 
# The dataset that we will use is <a href="https://bigquery.cloud.google.com/table/nyc-tlc:yellow.trips">a BigQuery public dataset</a>. Click on the link, and look at the column names. Switch to the Details tab to verify that the number of records is one billion, and then switch to the Preview tab to look at a few rows.
# 
# Write a SQL query to pick up the following fields
# <pre>
#   pickup_datetime,
#   pickup_longitude, pickup_latitude, 
#   dropoff_longitude, dropoff_latitude,
#   passenger_count,
#   trip_distance,
#   tolls_amount,
#   fare_amount,
#   total_amount
# </pre>
# from the dataset and explore a random subsample of the data. Sample size should be about 10,000 records. Make sure to pick a repeatable subset of the data so that if someone reruns this notebook, they will get the same results.
# <p>
# <b>Hint (highlight to see)</b>
# <pre style="color: white">
# Set the query string to be:
# SELECT above_fields FROM
#   `nyc-tlc.yellow.trips`
# WHERE
#   MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))),100000) = 1
# Then, use the BQ library:
# trips = bq.Query(query).execute().result().to_dataframe()
# </pre>

# In[4]:


# TODO: write a BigQuery query for the above fields
# Store it into a Pandas dataframe named "trips" that contains about 10,000 records.


# <h3> Exploring data </h3>
# 
# Let's explore this dataset and clean it up as necessary. We'll use the Python Seaborn package to visualize graphs and Pandas to do the slicing and filtering.

# In[8]:


ax = sns.regplot(x = "trip_distance", y = "fare_amount", ci = None, truncate = True, data = trips)


# Hmm ... do you see something wrong with the data that needs addressing?
# 
# It appears that we have a lot of invalid data that is being coded as zero distance and some fare amounts that are definitely illegitimate. Let's remove them from our analysis. We can do this by modifying the BigQuery query to keep only trips longer than zero miles and fare amounts that are at least the minimum cab fare ($2.50).
# 
# What's up with the streaks at \$45 and \$50?  Those are fixed-amount rides from JFK and La Guardia airports into anywhere in Manhattan, i.e. to be expected. Let's list the data to make sure the values look reasonable.
# 
# Let's examine whether the toll amount is captured in the total amount.

# In[9]:


tollrides = trips[trips['tolls_amount'] > 0]
tollrides[tollrides['pickup_datetime'] == '2014-05-20 23:09:00']


# Looking a few samples above, it should be clear that the total amount reflects fare amount, toll and tip somewhat arbitrarily -- this is because when customers pay cash, the tip is not known.  So, we'll use the sum of fare_amount + tolls_amount as what needs to be predicted.  Tips are discretionary and do not have to be included in our fare estimation tool.
# 
# Let's also look at the distribution of values within the columns.

# In[10]:


trips.describe()


# Hmm ... The min, max of longitude look strange.
# 
# Finally, let's actually look at the start and end of a few of the trips.

# In[11]:


def showrides(df, numlines):
  import matplotlib.pyplot as plt
  lats = []
  lons = []
  goodrows = df[df['pickup_longitude'] < -70]
  for iter, row in goodrows[:numlines].iterrows():
    lons.append(row['pickup_longitude'])
    lons.append(row['dropoff_longitude'])
    lons.append(None)
    lats.append(row['pickup_latitude'])
    lats.append(row['dropoff_latitude'])
    lats.append(None)

  sns.set_style("darkgrid")
  plt.plot(lons, lats)

showrides(trips, 10)


# In[12]:


showrides(tollrides, 10)


# As you'd expect, rides that involve a toll are longer than the typical ride.

# <h3> Quality control and other preprocessing </h3>
# 
# We need to do some clean-up of the data:
# <ol>
# <li>New York city longitudes are around -74 and latitudes are around 41.</li>
# <li>We shouldn't have zero passengers.</li>
# <li>Clean up the total_amount column to reflect only fare_amount and tolls_amount, and then remove those two columns.</li>
# <li>Before the ride starts, we'll know the pickup and dropoff locations, but not the trip distance (that depends on the route taken), so remove it from the ML dataset</li>
# <li>Discard the timestamp</li>
# </ol>
# 
# Let's change the BigQuery query appropriately.  In production, we'll have to carry out the same preprocessing on the real-time input data. 

# In[1]:


def sample_between(a, b):
    basequery = """
SELECT
  (tolls_amount + fare_amount) AS fare_amount,
  pickup_longitude AS pickuplon,
  pickup_latitude AS pickuplat,
  dropoff_longitude AS dropofflon,
  dropoff_latitude AS dropofflat,
  passenger_count*1.0 AS passengers
FROM
  `nyc-tlc.yellow.trips`
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
    sampler = "AND MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))), EVERY_N) = 1"
    sampler2 = "AND {0} >= {1}\n AND {0} < {2}".format(
           "MOD(ABS(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING))), EVERY_N * 100)",
           "(EVERY_N * {})".format(a), "(EVERY_N * {})".format(b)
        )
    return "{}\n{}\n{}".format(basequery, sampler, sampler2)

def create_query(phase, EVERY_N):
  """Phase: train (70%) valid (15%) or test (15%)"""
  query = ""
  if phase == 'train':
    # Training
    query = sample_between(0, 70)
  elif phase == 'valid':
    # Validation
    query = sample_between(70, 85)
  else:
    # Test
    query = sample_between(85, 100)
  return query.replace("EVERY_N", str(EVERY_N))


# In[ ]:


# TODO: try out train, test and valid here
print create_query('train', 100000)


# In[39]:


def to_csv(df, filename):
  outdf = df.copy(deep = False)
  outdf.loc[:, 'key'] = np.arange(0, len(outdf)) # rownumber as key
  # Reorder columns so that target is first column
  cols = outdf.columns.tolist()
  cols.remove('fare_amount')
  cols.insert(0, 'fare_amount')
  print cols  # new order of columns
  outdf = outdf[cols]
  outdf.to_csv(filename, header = False, index_label = False, index = False)
  print "Wrote {} to {}".format(len(outdf), filename)

for phase in ['train', 'valid', 'test']:
  query = create_query(phase, 100000)
  df = bq.Query(query).execute().result().to_dataframe()
  to_csv(df, 'taxi-{}.csv'.format(phase))


# <h3> Verify that datasets exist </h3>

# In[40]:


get_ipython().system('ls -l *.csv')


# We have 3 .csv files corresponding to train, valid, test.  The ratio of file-sizes correspond to our split of the data.

# In[41]:


get_ipython().run_line_magic('bash', '')
head taxi-train.csv


# Looks good! We now have our ML datasets and are ready to train ML models, validate them and evaluate them.

# <h3> Benchmark </h3>
# 
# Before we start building complex ML models, it is a good idea to come up with a very simple model and use that as a benchmark.
# 
# My model is going to be to simply divide the mean fare_amount by the mean trip_distance to come up with a rate and use that to predict.  Let's compute the RMSE of such a model.

# In[42]:


import datalab.bigquery as bq
import pandas as pd
import numpy as np
import shutil

def distance_between(lat1, lon1, lat2, lon2):
  # Haversine formula to compute distance "as the crow flies".  Taxis can't fly of course.
  dist = np.degrees(np.arccos(np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1)))) * 60 * 1.515 * 1.609344
  return dist

def estimate_distance(df):
  return distance_between(df['pickuplat'], df['pickuplon'], df['dropofflat'], df['dropofflon'])

def compute_rmse(actual, predicted):
  return np.sqrt(np.mean((actual - predicted)**2))

def print_rmse(df, rate, name):
  print "{1} RMSE = {0}".format(compute_rmse(df['fare_amount'], rate * estimate_distance(df)), name)

FEATURES = ['pickuplon','pickuplat','dropofflon','dropofflat','passengers']
TARGET = 'fare_amount'
columns = list([TARGET])
columns.extend(FEATURES) # in CSV, target is the first column, after the features
columns.append('key')
df_train = pd.read_csv('taxi-train.csv', header = None, names = columns)
df_valid = pd.read_csv('taxi-valid.csv', header = None, names = columns)
df_test = pd.read_csv('taxi-test.csv', header = None, names = columns)
rate = df_train['fare_amount'].mean() / estimate_distance(df_train).mean()
print "Rate = ${0}/km".format(rate)
print_rmse(df_train, rate, 'Train')
print_rmse(df_valid, rate, 'Valid') 
print_rmse(df_test, rate, 'Test') 


# The simple distance-based rule gives us a RMSE of <b>$9.35</b> on the validation dataset.  We have to beat this, of course, but you will find that simple rules of thumb like this can be surprisingly difficult to beat. You don't wnat to set a goal on the test dataset because you want to change the architecture of the network etc. to get the best validation error. Then, you can evaluate ONCE on the test data.

# ## Challenge Exercise
# 
# Let's say that you want to predict whether a Stackoverflow question will be acceptably answered. Using this [public dataset of questions](https://bigquery.cloud.google.com/table/bigquery-public-data:stackoverflow.posts_questions), create a machine learning dataset that you can use for classification.
# <p>
# What is a reasonable benchmark for this problem?
# What features might be useful?
# <p>
# If you got the above easily, try this harder problem: you want to predict whether a question will be acceptably answered within 2 days. How would you create the dataset?
# <p>
# Hint (highlight to see):
# <p style='color:white' linkstyle='color:white'> 
# You will need to do a SQL join with the table of [answers]( https://bigquery.cloud.google.com/table/bigquery-public-data:stackoverflow.posts_answers) to determine whether the answer was within 2 days.
# </p>

# Copyright 2018 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.