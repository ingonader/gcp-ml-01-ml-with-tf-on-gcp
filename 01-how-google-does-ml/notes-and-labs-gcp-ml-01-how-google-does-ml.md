# End-to-End Machine Learning with TensorFlow on GCP



## to download materials

software to download materials: <https://github.com/coursera-dl/coursera-dl>

link to coursera material: <https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp/lecture/wWBkE/effective-ml>

```bash
coursera-dl -u ingo.nader@gmail.com google-machine-learning --download-quizzes --download-notebooks --about
```



## Notes

* Do labs in Google Chrome (recommended)
* Open Incognito window, go to <http://console.cloud.google.com/>



# Lab 1: Rent-a-VM to process earthquake data

## Overview

*Duration is 1 min*

In this lab you spin up a virtual machine, configure its  security, access it remotely, and then carry out the steps of an  ingest-transform-and-publish data pipeline manually. 

### **What you learn**

In this lab, you:

* Create a Compute Engine instance with the necessary Access and Security
* SSH into the instance
* Install the software package Git (for source code version control)
* Ingest data into a Compute Engine instance
* Transform data on the Compute Engine instance
* Store the transformed data on Cloud Storage
* Publish Cloud Storage data to the web

## Introduction

*Duration is 1 min*

In this lab you spin up a virtual machine, install software on  it, and use it to do scientific data processing.  We do not recommend  that you work with Compute Engine instances at such a low-level, but you  can! 

In this lab, you will use Google Cloud Platform in a manner similar  to the way you likely use clusters today. Spinning up a virtual machine  and running your jobs on it is the closest you can get to working with  the public cloud as simply rented infrastructure. It doesn't take  advantage of the other benefits that Google Cloud Platform provides --  namely the ability to forget about infrastructure and work with your  scientific computation problems simply as software that requires to be  run.

You will ingest real-time earthquake data published by the United  States Geological Survey (USGS) and create maps that look like this:

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86565/original/img/b3b64f0a8d7eedde.png)

## Create Compute Engine instance with the necessary API access

*Duration is 4 min*

To create a Compute Engine instance:

### **Step 1**

Browse to <https://cloud.google.com/> 

### **Step 2**

Click on **Console**.

### **Step 3**

Click on the Menu (three horizontal lines):

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86565/original/img/aab385b15ea9c7f4.png)

### **Step 4**

Select **Compute Engine**.

### **Step 5**

Click **Create** and wait for a form to load. You will need to change some options on the form that comes up.

### **Step 6**

Change Identify and API access for the Compute Engine default service account to **Allow full access to all Cloud APIs**:

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86565/original/img/8ab244f9cffa6198.png)

### **Step 7**

Now, click **Create**

```bash
gcloud beta compute --project=qwiklabs-gcp-7370b56abba0fc9f instances create instance-2 --zone=us-central1-a --machine-type=n1-standard-1 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=305611432890-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/cloud-platform --image=debian-9-stretch-v20181113 --image-project=debian-cloud --boot-disk-size=10GB --boot-disk-type=pd-standard --boot-disk-device-name=instance-2
```





## SSH into the instance

*Duration is 2 min*

You can remotely access your Compute Engine instance using Secure Shell (SSH):

### Step 1

Click on **SSH**:

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86565/original/img/e4d9f3244db5ba38.png)

Note 

SSH keys are automatically transferred, and that you can ssh directly from the browser, with no extra software needed. 

### **Step 2**

To find some information about the Compute Engine instance, type the following into the command-line:

```
cat /proc/cpuinfo
```

```
processor       : 0
vendor_id       : GenuineIntel
cpu family      : 6
model           : 45
model name      : Intel(R) Xeon(R) CPU @ 2.60GHz
stepping        : 7
microcode       : 0x1
cpu MHz         : 2600.000
cache size      : 20480 KB
physical id     : 0
siblings        : 1
core id         : 0
cpu cores       : 1
apicid          : 0
initial apicid  : 0
fpu             : yes
fpu_exception   : yes
cpuid level     : 13
wp              : yes
flags           : fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2
 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology nonstop_tsc pni pclmulqdq ssse3 cx16 sse4_
1 sse4_2 x2apic popcnt aes xsave avx hypervisor lahf_lm ssbd ibrs ibpb stibp kaiser tsc_adjust xsaveopt arat arch_c
apabilities
bugs            : cpu_meltdown spectre_v1 spectre_v2 spec_store_bypass l1tf
bogomips        : 5200.00
clflush size    : 64
cache_alignment : 64
address sizes   : 46 bits physical, 48 bits virtual
power management:

```



## Install software

*Duration is 2 min*

### **Step 1**

Type the following into command-line:

```
sudo apt-get update
sudo apt-get -y -qq install git
```

### **Step 2**

Verify that git is now installed

```
git --version
```

## Ingest USGS data

*Duration is 3 min*

### **Step 1**

On the command-line, type:

```
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
```

This clones the code repo.

### **Step 2**

Navigate to the folder corresponding to this lab:

```
cd training-data-analyst/courses/machine_learning/deepdive/01_googleml/earthquakes
```

### **Step 3**

Examine the ingest code using `less`:

```
less ingest.sh
```

```bash
#!/bin/bash

# remove older copy of file, if it exists
rm -f earthquakes.csv

# download latest data from USGS
wget http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.csv -O earthquakes.csv

# Copyright 2016 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
```



The `less` command allows you to view the file (Press the **spacebar** to scroll down; the letter **b** to **b**ack up a page; the letter **q** to **q**uit). 

The program `ingest.sh` downloads a dataset of earthquakes  in the past 7 days from the US Geological Survey.  Where is this file  downloaded? To disk or to Cloud Storage? __________________________

### **Step 4**

Run the ingest code:

```
bash ingest.sh
```

### **Step 5**

Verify that some data has been downloaded:

```
head earthquakes.csv
```

The `head` command shows you the first few lines of the file.

## Transform the data

*Duration is 3 min*

You will use a Python program to transform the raw data into a map of earthquake activity:

### Step 1

The transformation code is explained in detail in this notebook: 

<https://github.com/GoogleCloudPlatform/datalab-samples/blob/master/basemap/earthquakes.ipynb>  

Feel free to read the narrative to understand what the transformation  code does.  The notebook itself was written in Datalab, a GCP product  that you will use later in this set of labs.

### **Step 2**

First, install the necessary Python packages on the Compute Engine instance:

```
bash install_missing.sh
```

### **Step 3**

Then, run the transformation code:

```
cat ./transform.py
```

```python
#!/usr/bin/env python3

# Copyright 2016 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# See https://github.com/GoogleCloudPlatform/datalab-samples/blob/master/basemap/earthquakes.ipynb for a notebook that illustrates this code

import csv
import requests
import io
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Classes to hold the data
class EarthQuake:
  def __init__(self, row):
    # Parse earthquake data from USGS
    self.timestamp = row[0]
    self.lat = float(row[1])
    self.lon = float(row[2])
    try:
      self.magnitude = float(row[4])
    except ValueError:
      self.magnitude = 0
        
def get_earthquake_data(url):
  # Read CSV earthquake data from USGS
  response = requests.get(url)
  csvio = io.StringIO(response.text)
  reader = csv.reader(csvio)
  header = next(reader)
  quakes = [EarthQuake(row) for row in reader]
  quakes = [q for q in quakes if q.magnitude > 0]
  return quakes


# control marker color and size based on magnitude
def get_marker(magnitude):
    markersize = magnitude * 2.5;
    if magnitude < 1.0:
        return ('bo'), markersize
    if magnitude < 3.0:
        return ('go'), markersize
    elif magnitude < 5.0:
        return ('yo'), markersize
    else:
        return ('ro'), markersize


def create_png(url, outfile): 
  quakes = get_earthquake_data('http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.csv')
  print(quakes[0].__dict__)

  # Set up Basemap
  mpl.rcParams['figure.figsize'] = '16, 12'
  m = Basemap(projection='kav7', lon_0=-90, resolution = 'l', area_thresh = 1000.0)
  m.drawcoastlines()
  m.drawcountries()
  m.drawmapboundary(fill_color='0.3')
  m.drawparallels(np.arange(-90.,99.,30.))
  junk = m.drawmeridians(np.arange(-180.,180.,60.))

  # sort earthquakes by magnitude so that weaker earthquakes
  # are plotted after (i.e. on top of) stronger ones
  # the stronger quakes have bigger circles, so we'll see both
  start_day = quakes[-1].timestamp[:10]
  end_day = quakes[0].timestamp[:10]
  quakes.sort(key=lambda q: q.magnitude, reverse=True)

  # add earthquake info to the plot
  for q in quakes:
    x,y = m(q.lon, q.lat)
    mcolor, msize = get_marker(q.magnitude)
    m.plot(x, y, mcolor, markersize=msize)

  # add a title
  plt.title("Earthquakes {0} to {1}".format(start_day, end_day))
  plt.savefig(outfile)

if __name__ == '__main__':
  url = 'http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.csv'
  outfile = 'earthquakes.png'
  create_png(url, outfile)
```

```bash
./transform.py
```

```
{'timestamp': '2018-11-30T09:17:56.100Z', 'lat': 40.0566673, 'lon': -120.8028336, 'magnitude': 2.32}
/usr/lib/python3/dist-packages/mpl_toolkits/basemap/__init__.py:1623: MatplotlibDeprecationWarning: The get_axis_bgcolor function was deprecated in version 2.0. Use get_facecolor instead.
  fill_color = ax.get_axis_bgcolor()
/usr/lib/python3/dist-packages/mpl_toolkits/basemap/__init__.py:3260: MatplotlibDeprecationWarning: The ishold function was deprecated in version 2.0.
  b = ax.ishold()
/usr/lib/python3/dist-packages/mpl_toolkits/basemap/__init__.py:3269: MatplotlibDeprecationWarning: axes.hold is deprecated.
    See the API Changes document (http://matplotlib.org/api/api_changes.html)
    for more details.
  ax.hold(b)
```



### **Step 4**

You will notice a new image file if you list the contents of the directory:

```
ls -l
```

```
total 664
-rw-r--r-- 1 google1633866_student google1633866_student    637 Nov 30 09:11 commands.sh
-rw-r--r-- 1 google1633866_student google1633866_student 338439 Nov 30 09:09 earthquakes.csv
-rw-r--r-- 1 google1633866_student google1633866_student    751 Nov 30 09:11 earthquakes.htm
-rw-r--r-- 1 google1633866_student google1633866_student 311644 Nov 30 09:26 earthquakes.png
-rwxr-xr-x 1 google1633866_student google1633866_student    759 Nov 30 09:11 ingest.sh
-rwxr-xr-x 1 google1633866_student google1633866_student    707 Nov 30 09:11 install_missing.sh
drwxr-xr-x 2 google1633866_student google1633866_student   4096 Nov 30 09:11 scheduled
-rwxr-xr-x 1 google1633866_student google1633866_student   3058 Nov 30 09:11 transform.py
```



## Create bucket

*Duration is 2 min*

Create a bucket using the GCP console:

### **Step 1**

Browse to the GCP Console by visiting <http://cloud.google.com> and clicking on **Go To Console**

### **Step 2**

Click on the Menu (3 bars) at the top-left and select **Storage**

### **Step 3**

Click on **Create bucket.** 

### **Step 4**

Choose a globally unique bucket name (your project name is unique, so you could use that).  You can leave it as **Multi-Regional,** or improve speed and reduce costs by making it **Regional** . Then, click **Create**.

Note: Please pick a region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1** as your region.

### **Step 5**

Note down the name of your bucket: _______________________________

> inna-ql-gcp-7370b58abba0fc9f



In this and future labs, you will insert this whenever the directions ask for `<YOUR-BUCKET>.`

## Store data

*Duration is 1 min*

To store the original and transformed data in Cloud Storage

### **Step 1**

In the SSH window of the Compute Engine instance, type:

```
#gsutil cp earthquakes.* gs://<YOUR-BUCKET>/earthquakes/
gsutil cp earthquakes.* gs://inna-ql-gcp-7370b58abba0fc9f/earthquakes/
```

to copy the files to Cloud Storage

### **Step 2**

On the GCP console, click on your bucket name, and notice there are three new files present in the earthquakes folder.

## Publish Cloud Storage files to web

*Duration is 2 min*

To publish Cloud Storage files to the web:

### **Step 1**

In the SSH window of the Compute Engine instance, type:

```
gsutil acl ch -u AllUsers:R gs://<YOUR-BUCKET>/earthquakes/*
gsutil acl ch -u AllUsers:R gs://inna-ql-gcp-7370b58abba0fc9f/earthquakes/*
```

### **Step 2**

Click on the **Public link** corresponding to **earthquakes.htm**

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86565/original/img/ce84dc744ff4c872.png)

### **Step 3**

What is the URL of the published Cloud Storage file? How does it relate to your bucket name and content?

______________________________________________________

* https://storage.cloud.google.com/inna-ql-gcp-7370b58abba0fc9f/earthquakes/earthquakes.png?_ga=2.159725237.-92316385.1543568854
* https://storage.cloud.google.com/inna-ql-gcp-7370b58abba0fc9f/earthquakes/earthquakes.htm?_ga=2.197390631.-92316385.1543568854
* https://storage.cloud.google.com/inna-ql-gcp-7370b58abba0fc9f/earthquakes/earthquakes.csv?_ga=2.197390631.-92316385.1543568854

### **Step 4**

What are some advantages of publishing to Cloud Storage? _____________________________________________

## Clean up

*Duration is 2 min*

To delete the Compute Engine instance (since we won't need it any more):

### **Step 1**

On the GCP console, click the Menu (three horizontal bars) and select **Compute Engine**

### **Step 2**

Click on the checkbox corresponding to the instance that you created (the default name was instance-1)

### **Step 3**

Click on the **Delete** button in the top-right corner

### **Step 4**

Does deleting the instance have any impact on the files that you stored on Cloud Storage? ___________________

## Summary

*Duration is 1 min*

In this lab, you used Google Cloud Platform (GCP) as rented  infrastructure. You can spin up a Compute Engine VM, install custom  software on it, and run your processing jobs. However, using GCP in this  way doesn't take advantage of the other benefits that Google Cloud  Platform provides -- namely the ability to forget about infrastructure  and work with your scientific computation problems simply as software  that requires to be run.

©Google, Inc. or its affiliates. All rights reserved. Do not distribute.

Provide Feedback on this Lab



# Lab 2: Analyzing data using Datalab and BigQuery

## Overview

*Duration is 1 min*

In this lab you analyze a large (70 million rows, 8 GB) airline dataset using Google BigQuery and Cloud Datalab.

### **What you learn**

In this lab, you:

* Launch Cloud Datalab
* Invoke a BigQuery query
* Create graphs in Datalab

This lab illustrates how you can carry out data exploration of large  datasets, but continue to use familiar tools like Pandas and Juypter.  The "trick" is to do the first part of your aggregation in BigQuery, get  back a Pandas dataset and then work with the smaller Pandas dataset  locally.  Cloud Datalab provides a managed Jupyter experience, so that  you don't need to run notebook servers yourself.

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost. 

1. Make sure you signed in to Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, 01:30:00) and make sure you can finish in that time block.

There is no pause feature. You can restart if needed, but you have to start at the beginning.

1. When ready, click **Start Lab.** 
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console.![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86553/original/img/eb2f9b843e5c184c.png) 
3. Click **Open Google Console**.
4. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.

If you use other credentials, you'll get errors or **incur charges**.

1. Accept the terms and skip the recovery resource page.

Do not click **End** unless you are finished with the lab or want to restart it. This clears your work and removes the project.

## Launch Cloud Datalab

To launch Cloud Datalab:

### **Step 1**

Open Cloud Shell. In the upper-right, click **Activate Cloud Shell** (![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86553/original/img/d258c43ee8321ba.png)). And then click **Start Cloud Shell**.

### **Step 2**

In Cloud Shell, type:

```
gcloud compute zones list
```

Note: Please pick a zone in a geographically close region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1-c** as your zone.

### **Step 3**

In Cloud Shell, type:

```
datalab create mydatalabvm --zone <ZONE>
```

Replace <ZONE> with a zone name you picked from the previous step.

Note: follow the prompts during this process.

Datalab will take about 5 minutes to start.

### **Step 4**

Look back at Cloud Shell, and follow any prompts. If asked for a ssh passphrase, just hit return (for no passphrase).

### **Step 5**

If necessary, wait for Datalab to finish launching. Datalab is ready when you see a message prompting you to do a "Web Preview".

### **Step 6**

Click on the **Web Preview** icon on the top-right  corner of the Cloud Shell ribbon.  After clicking on "Web preview",  click on "Change port" and change the port number to 8081.  Click  "Change and Preview" to open the Datalab web interface.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86553/original/img/a10fdc06c9fd5db0.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86553/original/img/9ce67b5f42c53a37.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86553/original/img/3f04d163f957a625.png)

Note: If the cloud shell used for running the  datalab command is closed or interrupted, the connection to your Cloud  Datalab VM will terminate. If that happens, you may be able to reconnect  using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. 

## Enable APIs

* Ensure the **Cloud Source Repositories** API is enabled: <https://console.cloud.google.com/apis/library/sourcerepo.googleapis.com/?q=Repositories> 

## Invoke BigQuery

To invoke a BigQuery query:

### **Step 1**

Navigate to the BigQuery console by selecting BigQuery from the top-left-corner ("hamburger") menu.  

**Note**: If you get an error on the BigQuery console that starts with *Error: Access Denied: Project qwiklabs-resources:*, then click on the drop-down menu and switch to your Qwiklabs project.

### **Step 2**

In the BigQuery Console, click on **Compose Query**.  Then, select **Show Options** and ensure that the Legacy SQL menu is NOT checked (we will be using Standard SQL). 

### **Step 3**

In the query textbox, type:

```
#standardSQL
SELECT
  departure_delay,
  COUNT(1) AS num_flights,
  APPROX_QUANTILES(arrival_delay, 5) AS arrival_delay_quantiles
FROM
  `bigquery-samples.airline_ontime_data.flights`
GROUP BY
  departure_delay
HAVING
  num_flights > 100
ORDER BY
  departure_delay ASC
```

What is the median arrival delay for flights left 35 minutes early?  ___________

(Answer: the typical flight that left 35 minutes early arrived 28 minutes early.)

### **Step 4**

Look back at Cloud Shell, and follow any prompts. If asked for a ssh passphrase, just hit return (for no passphrase).

### **Step 5 (Optional)**

Can you write a query to find the airport pair (departure and arrival  airport) that had the maximum number of flights between them? 

**Hint:** you can group by multiple fields.

One possible answer:

```
#standardSQL
SELECT
  departure_airport,
  arrival_airport,
  COUNT(1) AS num_flights
FROM
  `bigquery-samples.airline_ontime_data.flights`
GROUP BY
  departure_airport,
  arrival_airport
ORDER BY
  num_flights DESC
LIMIT
  10
```

## Draw graphs in Cloud Datalab

### **Step 1**

If necessary, wait for Datalab to finish launching. Datalab is ready when you see a message prompting you to do a "Web Preview".

### **Step 2**

In Cloud Datalab home page (browser), navigate into **notebooks**.  You should now be in datalab/notebooks/

**Step 3**

Start a new notebook by clicking on the **+Notebook** icon.  Rename the notebook to be **flights**.

### **Step 4**

In a cell in Datalab, type the following, then click **Run**

```
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
```

Note that we have gotten the results from BigQuery as a Pandas dataframe.

In what Python data structure are the deciles in?

### **Step 5**

In the next cell in Datalab, type the following, then click **Run**

```
import pandas as pd
percentiles = df['arrival_delay_deciles'].apply(pd.Series)
percentiles = percentiles.rename(columns = lambda x : str(x*10) + "%")
df = pd.concat([df['departure_delay'], percentiles], axis=1)
df.head()
```

What has the above code done to the columns in the Pandas DataFrame?

### **Step 6**

In the next cell in Datalab, type the following, then click **Run**

```
without_extremes = df.drop(['0%', '100%'], 1)
without_extremes.plot(x='departure_delay', xlim=(-30,50), ylim=(-50,50));
```

Suppose we were creating a machine learning model to predict the  arrival delay of a flight. Do you think departure delay is a good input  feature? Is this true at all ranges of departure delays?

Hint: Try removing the xlim and ylim from the plotting command.

## Cleanup (Optional)

**Step 1**

You could leave Datalab instance running until your class ends. The  default machine type is relatively inexpensive. However, if you want to  be frugal, you can stop and restart the instance between labs or when  you go home for the day.  To do so, follow the next two steps.

**Step 2**

Click on the person icon in the top-right corner of your Datalab window and click on the button to **STOP the VM.** 

**Step 3**

You are not billed for stopped VMs. Whenever you want to restart Datalab, open Cloud Shell and type in:

```
datalab connect mydatalabvm
```

This will restart the virtual machine and launch the Docker image that runs Datalab. 

## Summary

In this lab, you learned how to carry out data exploration of  large datasets using BigQuery, Pandas, and Juypter. The "trick" is to do  the first part of your aggregation in BigQuery, get back a Pandas  dataset and then work with the smaller Pandas dataset locally.  Cloud  Datalab provides a managed Jupyter experience, so that you don't need to  run notebook servers yourself.

©Google, Inc. or its affiliates. All rights reserved. Do not distribute.



# Lab 3: Invoking Machine Learning APIs

## Overview

*Duration is 1 min*

In this lab you use Machine Learning APIs from within Datalab.

### **What you learn**

In this lab, you learn how to invoke ML APIs from Datalab and use their results.

## Introduction

*Duration is 1 min*

In this lab, you will first 

* clone the code repo within your Cloud Datalab environment

and then invoke ML APIs from Datalab to carry out some representative tasks:

* Vision API to detect text in an image
* Translate API to translate that text into English
* Natural Language API to find the sentiment of some famous quotes
* Speech API to transcribe an audio file

ML APIs are microservices. When we build ML models ourselves, it  should be our goal to make them so easy to use and stand-alone. 

## Enable APIs

* Ensure the **Cloud Source Repositories** API is enabled: <https://console.cloud.google.com/apis/library/sourcerepo.googleapis.com/?q=Repositories> 

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost. 

1. Make sure you signed in to Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, 01:30:00) and make sure you can finish in that time block.

There is no pause feature. You can restart if needed, but you have to start at the beginning.

1. When ready, click **Start Lab.** 
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console.![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86559/original/img/eb2f9b843e5c184c.png) 
3. Click **Open Google Console**.
4. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.

If you use other credentials, you'll get errors or **incur charges**.

1. Accept the terms and skip the recovery resource page.

Do not click **End** unless you are finished with the lab or want to restart it. This clears your work and removes the project.

## Launch Cloud Datalab

To launch Cloud Datalab:

### **Step 1**

Open Cloud Shell. In the upper-right, click **Activate Cloud Shell** (![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86559/original/img/d258c43ee8321ba.png)). And then click **Start Cloud Shell**.

### **Step 2**

In Cloud Shell, type:

```
gcloud compute zones list
```

Note: Please pick a zone in a geographically close region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1-c** as your zone.

### **Step 3**

In Cloud Shell, type:

```
datalab create mydatalabvm --zone <ZONE>
```

Replace <ZONE> with a zone name you picked from the previous step.

Note: follow the prompts during this process.

Datalab will take about 5 minutes to start.

### **Step 4**

Look back at Cloud Shell, and follow any prompts. If asked for a ssh passphrase, just hit return (for no passphrase).

### **Step 5**

If necessary, wait for Datalab to finish launching. Datalab is ready when you see a message prompting you to do a "Web Preview".

### **Step 6**

Click on the **Web Preview** icon on the top-right  corner of the Cloud Shell ribbon.  After clicking on "Web preview",  click on "Change port" and change the port number to 8081.  Click  "Change and Preview" to open the Datalab web interface.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86559/original/img/a10fdc06c9fd5db0.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86559/original/img/9ce67b5f42c53a37.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86559/original/img/3f04d163f957a625.png)

Note: If the cloud shell used for running the  datalab command is closed or interrupted, the connection to your Cloud  Datalab VM will terminate. If that happens, you may be able to reconnect  using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. 

## Clone course repo within your Datalab instance

To clone the course repo in your Datalab instance:

### **Step 1**

In Cloud Datalab home page (browser), navigate into "**notebooks**" and add a new notebook using the icon ![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86559/original/img/5fdee4bbcdee4b9a.png)  on the top left.

### **Step 2**

Rename this notebook as ‘**repocheckout**'.

### **Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86559/original/img/1a3d26cf1e7e1e5f.png)

### **Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory. 

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/86559/original/img/ce070262a667324b.png)

## Enable APIs and Get API key

*Duration is 1 min*

To get an API key:

### **Step 1**

From the GCP console menu, select **APIs and services** and select **Library**

**Step 2**

In the search box, type **vision** to find the **Google Cloud Vision API** and click on the hyperlink.

### **Step 3**

Click **Enable** if necessary

### **Step 4**

Follow the same process to enable **Translate API, Speech API, and Natural Language** APIs.

### **Step 5**

From the GCP console menu, select **APIs and services** and select **Credentials**. 

### **Step 6**

If you do not already have an API key, click the **Create credentials** button and select **API key**. Once created, click close. You will need this API key in the notebook later.

## Invoke ML APIs from Datalab

*Duration is 8 min*

### **Step 1**

In the Datalab browser, navigate to **training-data-analyst > courses > machine_learning > deepdive > 01_googleml > mlapis.ipynb**

Note: If the cloud shell used for running the  datalab command is closed or interrupted, the connection to your Cloud  Datalab VM will terminate. If that happens, you may be able to reconnect  using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

### **Step 2**

Read the commentary, then run the Python snippets (Use Shift+Enter to  run each piece of code) in the cell, step by step. Make sure to insert  your API Key in the first Python cell.

©Google, Inc. or its affiliates. All rights reserved. Do not distribute.

# [[eof]]

