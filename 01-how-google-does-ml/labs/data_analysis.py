
# coding: utf-8

# ## Data Analysis using Datalab and BigQuery

# In[44]:


query="""
SELECT
  departure_delay,
  COUNT(1) AS num_flights,
  APPROX_QUANTILES(arrival_delay, 10) AS arrival_delay_deciles
FROM
  `bigquery-samples.airline_ontime_data.flights`
GROUP BY
  departure_delay
HAVING
  num_flights > 100
ORDER BY
  departure_delay ASC
"""

import google.datalab.bigquery as bq
df = bq.Query(query).execute().result().to_dataframe()
df.head()


# In[45]:


import pandas as pd
percentiles = df['arrival_delay_deciles'].apply(pd.Series)
percentiles = percentiles.rename(columns = lambda x : str(x*10) + "%")
df = pd.concat([df['departure_delay'], percentiles], axis=1)
df.head()


# In[47]:


without_extremes = df.drop(['0%', '100%'], 1)
without_extremes.plot(x='departure_delay', xlim=(-30,50), ylim=(-50,50));


# ## Challenge Exercise
# 
# Your favorite college basketball team is playing at home and trailing by 3 points with 4 minutes left to go. Using this public dataset of [NCAA basketball play-by-data](https://bigquery.cloud.google.com/table/bigquery-public-data:ncaa_basketball), calculate the probability that your team will come from behind to win the game.
# <p>
# Hint (highlight to view)
# <p style='color:white'>
# You will need to find games where period=2, game_clock = 4, and (away_pts - home_pts) = 3.  Then, you will need to find the fraction of such games that end with home_pts > away_pts. </p>
# <p>
# If you got this easily, then for a greater challenge, repeat this exercise, but plot a graph of come-from-behind odds by time-remaining and score-margin. See https://medium.com/analyzing-ncaa-college-basketball-with-gcp/so-youre-telling-me-there-s-a-chance-e4ba0ad7f542 for inspiration.

# Copyright 2018 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.