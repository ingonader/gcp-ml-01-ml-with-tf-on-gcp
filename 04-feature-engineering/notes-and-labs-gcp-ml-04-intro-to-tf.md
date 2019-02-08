# Feature Engineering



## to download materials

software to download materials: <https://github.com/coursera-dl/coursera-dl>

link to coursera material: <https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp/lecture/wWBkE/effective-ml>

```bash
coursera-dl -u ingo.nader@gmail.com feature-engineering --download-quizzes --download-notebooks --about
```



## Notes

* Do labs in Google Chrome (recommended)
* Open Incognito window, go to <http://console.cloud.google.com/>



# Lab 1: [ML on GCP C4] Improving model accuracy with new features

## Overview

*Duration is 1 min*

This lab is part of a lab series, where you go from exploring a taxicab dataset to training and deploying a high-accuracy distributed model with Cloud ML Engine.

### **What you learn**

In this lab, you will :

* Improve the accuracy of a model by adding new features with the appropriate representation

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

1. Make sure you signed into Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, ![img/time.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/699409e014fbb8298cf5747a0535e04d13f51dfbb54fbaba57e7276dd12c6e95.png) and make sure you can finish in that time block.

There is no pause feature. You can restart if needed, but you have to start at the beginning.

1. When ready, click ![img/start_lab.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5c4f31eeebd0a24cae6cdc2760a29cfaf04136b325a3988d6620c3cd04370dda.png).
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console. ![img/open_console.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5eaee037e6cedbf49f6a702ab8a9ef820bb8e717332946ff76a0831a6396aafc.png)
3. Click **Open Google Console**.
4. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.

If you use other credentials, you'll get errors or **incur charges**.

1. Accept the terms and skip the recovery resource page.

Do not click **End** unless you are finished with the lab or want to restart it. This clears your work and removes the project.

## Launch Cloud Datalab

To launch Cloud Datalab:

**Step 1**

Open Cloud Shell. The Cloud Shell icon is at the top right of the Google Cloud Platform [web console](https://console.cloud.google.com/).

**Step 2**

In Cloud Shell, type:

```
gcloud compute zones list
```

**Note**: Please pick a zone in a geographically close region from the following: **us-east1, us-central1, asia-east1, europe-west1**. These are the regions that currently support Cloud ML Engine jobs. Please verify [here](https://cloud.google.com/ml-engine/docs/tensorflow/environment-overview#cloud_compute_regions) since this list may have changed after this lab was last updated. For example, if you are in the US, you may choose **us-east1-c** as your zone.

**Step 3**

In Cloud Shell, type:

```
datalab create mydatalabvm --zone <ZONE>
```

Replace with a zone name you picked from the previous step.

**Note**: follow the prompts during this process.

Datalab will take about 5 minutes to start.

**Step 4**

Look back at Cloud Shell and follow any prompts. If asked for an ssh passphrase, hit return (for no passphrase).

**Step 5**

If necessary, wait for Datalab to finishing launching. Datalab is ready when you see a message prompting you to do a **Web Preview**.

**Step 6**

Click on **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. Click **Change Port** and enter the port **8081** and click **Change and Preview**.

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6332967bdbfead14213237528b4e612f00691e996d73e01fe0fec0bd30a8247f.png)

![web-preview](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/083b75223bee4818c3b7c8624dab552a6873de2a5ae58f077a90919940ac134a.png)

![change-port](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d42795d791a43d871c59c8ad8eccb2290e93209db19e069c02672656761ebbe8.png)

**Note**: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command **datalab connect mydatalabvm** in your new Cloud Shell.

## Clone course repo within your Datalab instance

To clone the course repo in your datalab instance:

**Step 1**

In Cloud Datalab home page (browser), navigate into **notebooks** and add a new notebook using the icon ![notebook.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/fef0cc8c36a1856aa4ca73423f2ba59dde635267437c1253c268f366dfe19899.png) on the top left.

**Step 2**

Rename this notebook as **repocheckout**.

**Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![clone.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0a35d9ea37ae5908d89379a143c4fcd6292a6d29819fd34bc097ae17f21bd875.png)

**Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory.

![datalab.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/821c529680587fbc93135a3edde224a523aa07d0c38a07cf7967f13d082b7f0e.png)

## Experiment with features

*Duration is 15 min*

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **notebooks > training-data-analyst > courses > machine_learning > deepdive > 04_features** and open **a_features.ipynb**.

**Note:** If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

In Datalab, click on **Clear | Clear all Cells**. Now read the narrative and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 2: [ML on GCP C4] A simple Dataflow pipeline (Python)

## Overview

*Duration is 1 min*

In this lab, you learn how to write a simple Dataflow pipeline and run it both locally and on the cloud.

### **What you learn**

In this lab, you learn how to:

* Write a simple pipeline in Python
* Execute the query on the local machine
* Execute the query on the cloud

## Introduction

*Duration is 1 min*

The goal of this lab is to become familiar with the structure of a Dataflow project and learn how to execute a Dataflow pipeline.

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

1. Make sure you signed into Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, ![img/time.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/699409e014fbb8298cf5747a0535e04d13f51dfbb54fbaba57e7276dd12c6e95.png) and make sure you can finish in that time block.

There is no pause feature. You can restart if needed, but you have to start at the beginning.

1. When ready, click ![img/start_lab.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5c4f31eeebd0a24cae6cdc2760a29cfaf04136b325a3988d6620c3cd04370dda.png).
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console. ![img/open_console.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5eaee037e6cedbf49f6a702ab8a9ef820bb8e717332946ff76a0831a6396aafc.png)
3. Click **Open Google Console**.
4. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.

If you use other credentials, you'll get errors or **incur charges**.

1. Accept the terms and skip the recovery resource page.

Do not click **End** unless you are finished with the lab or want to restart it. This clears your work and removes the project.

## Open Dataflow project

*Duration is 3 min*

### **Step 1**

Start CloudShell and clone the source repo which has starter scripts for this lab:

```bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
```

Then navigate to the code for this lab:

```bash
cd training-data-analyst/courses/data_analysis/lab2/python
```

### **Step 2**

Install the necessary dependencies for Python dataflow:

```bash
sudo ./install_packages.sh
```

Verify that you have the right version of pip (should be > 8.0):

```bash
pip -V
```

If not, open a new CloudShell tab and it should pick up the updated pip.

## Pipeline filtering

*Duration is 5 min*

### **Step 1**

View the source code for the pipeline using the Cloud Shell file browser:

![f1f5da1fd2c75d3a.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/27c38bcd508c40850b7650de26c1e481496c25ec1a7cc997c99c594a914c15e9.png)

In the file directory, navigate to **/training-data-analyst/courses/data_analysis/lab2/python**.

![499badba3c564a51.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/19728348b0752831b24a75e9737e7be9caf630b94ffc7e9bd9a88473ff7bcf50.png)

Find **grep.py**.

![8c6f80d2b0a9f0d3.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0c1226c1335414a880e63ec62dece43e2da13535904d6e322ed1302cf4ac9eef.png)

Or you can navigate to the directly and view the file using **nano** if you prefer:

```bash
nano grep.py
```

```python
#!/usr/bin/env python

"""
Copyright Google Inc. 2016
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import apache_beam as beam
import sys

def my_grep(line, term):
   if line.startswith(term):
      yield line

if __name__ == '__main__':
   p = beam.Pipeline(argv=sys.argv)
   input = '../javahelp/src/main/java/com/google/cloud/training/dataanalyst/javahelp/*.java'
   output_prefix = '/tmp/output'
   searchTerm = 'import'

   # find all lines that contain the searchTerm
   (p
      | 'GetJava' >> beam.io.ReadFromText(input)
      | 'Grep' >> beam.FlatMap(lambda line: my_grep(line, searchTerm) )
      | 'write' >> beam.io.WriteToText(output_prefix)
   )

   p.run().wait_until_finish()

```



### **Step 2**

What files are being read? _____________________________________________________

What is the search term? ______________________________________________________

Where does the output go? ___________________________________________________

There are three transforms in the pipeline:

1. What does the transform do? _________________________________
2. What does the second transform do? ______________________________

* Where does its input come from? ________________________
* What does it do with this input? __________________________
* What does it write to its output? __________________________
* Where does the output go to? ____________________________

1. What does the third transform do? _____________________

## Execute the pipeline locally

*Duration is 2 min*

### **Step 1**

Execute locally:

```bash
python grep.py
```

Note: if you see an error that says "`No handlers could be found for logger "oauth2client.contrib.multistore_file",` you may ignore it. The error is simply saying that logging from the oauth2 library will go to stderr.

### **Step 2**

Examine the output file:

```bash
cat /tmp/output-*
```

```bash
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import org.apache.beam.runners.dataflow.options.DataflowPipelineOptions;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO;
import org.apache.beam.sdk.io.gcp.pubsub.PubsubIO;
import org.apache.beam.sdk.options.Default;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.transforms.Sum;
import org.apache.beam.sdk.transforms.windowing.SlidingWindows;
import org.apache.beam.sdk.transforms.windowing.Window;
import org.joda.time.Duration;
import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;
import java.util.ArrayList;
import java.util.List;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.io.TextIO;
import org.apache.beam.sdk.options.Default;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.PipelineOptions;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.transforms.Sum;
import org.apache.beam.sdk.transforms.Top;
import org.apache.beam.sdk.values.KV;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import com.google.api.services.bigquery.model.TableRow;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO;
import org.apache.beam.sdk.io.TextIO;
import org.apache.beam.sdk.options.Default;
import org.apache.beam.sdk.options.Description;
import org.apache.beam.sdk.options.PipelineOptions;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.transforms.Sum;
import org.apache.beam.sdk.transforms.Top;
import org.apache.beam.sdk.transforms.View;
import org.apache.beam.sdk.values.KV;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.sdk.values.PCollectionView;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.io.TextIO;
import org.apache.beam.sdk.options.PipelineOptions;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.ParDo;
```



Does the output seem logical? ______________________

## Execute the pipeline on the cloud

*Duration is 10 min*

### **Step 1**

If you don't already have a bucket on Cloud Storage, create one from the [Storage section of the GCP console](http://console.cloud.google.com/storage). Bucket names have to be globally unique.

### **Step 2**

Copy some Java files to the cloud (make sure to replace `<YOUR-BUCKET-NAME>` with the bucket name you created in the previous step):

```bash
export BUCKET=inna-gcp-trn-bckt
```



```bash
gsutil cp ../javahelp/src/main/java/com/google/cloud/training/dataanalyst/javahelp/*.java gs://$BUCKET/javahelp
```

### **Step 3**

Edit the Dataflow pipeline in `grepc.py` by opening up in the Cloud Shell in-browser editor again or by using the command line with nano:

![2267f36fb97f67cc.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/15f3ecc6639d922ce93dac74c10d59c06cf0734e1469f09a5c96f9f1f38614c2.png)

```bash
nano grepc.py
```

```python
#!/usr/bin/env python

"""
Copyright Google Inc. 2016
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import apache_beam as beam

def my_grep(line, term):
   if line.startswith(term):
      yield line

PROJECT='cloud-training-demos'
BUCKET='cloud-training-demos'

def run():
   argv = [
      '--project={0}'.format(PROJECT),
      '--job_name=examplejob2',
      '--save_main_session',
      '--staging_location=gs://{0}/staging/'.format(BUCKET),
      '--temp_location=gs://{0}/staging/'.format(BUCKET),
      '--runner=DataflowRunner'
   ]

   p = beam.Pipeline(argv=argv)
   input = 'gs://{0}/javahelp/*.java'.format(BUCKET)
   output_prefix = 'gs://{0}/javahelp/output'.format(BUCKET)
   searchTerm = 'import'

   # find all lines that contain the searchTerm
   (p
      | 'GetJava' >> beam.io.ReadFromText(input)
      | 'Grep' >> beam.FlatMap(lambda line: my_grep(line, searchTerm) )
      | 'write' >> beam.io.WriteToText(output_prefix)
   )

   p.run()

if __name__ == '__main__':
   run()

```



and changing the **PROJECT** and **BUCKET** variables appropriately.

### **Step 4**

Submit the Dataflow to the cloud:

```bash
python grepc.py
```

Because this is such a small job, running on the cloud will take significantly longer than running it locally (on the order of 2-3 minutes).

### **Step 5**

On your [Cloud Console](https://console.cloud.google.com/), navigate to the **Dataflow** section (from the 3 bars on the top-left menu), and look at the Jobs. Select your job and monitor its progress. You will see something like this:

![f55e71303e86b156.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/80caf1640b93a21d9b64282af0fe98c23ee157b6b5b7d5f9561bcb88418693e9.png)

### **Step 6**

Wait for the job status to turn to **Succeeded**. At this point, your CloudShell will display a command-line prompt. In CloudShell, examine the output:

```bash
gsutil cat gs://$BUCKET/javahelp/output-*
```

## What you learned

*Duration is 1 min*

In this lab, you:

* Executed a Dataflow pipeline locally
* Executed a Dataflow pipeline on the cloud.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 3: [ML on GCP C4] MapReduce in Dataflow (Python)

## Overview

*Duration is 1 min*

In this lab, you learn how to use pipeline options and carry out Map and Reduce operations in Dataflow.

### **What you need**

You must have completed Lab 0 and have the following:

* Logged into GCP Console with your Qwiklabs generated account

### **What you learn**

In this lab, you learn how to:

* Use pipeline options in Dataflow
* Carry out mapping transformations
* Carry out reduce aggregations

## Introduction

*Duration is 1 min*

The goal of this lab is to learn how to write MapReduce operations using Dataflow.

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

1. Make sure you signed into Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, ![img/time.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/699409e014fbb8298cf5747a0535e04d13f51dfbb54fbaba57e7276dd12c6e95.png) and make sure you can finish in that time block.

There is no pause feature. You can restart if needed, but you have to start at the beginning.

1. When ready, click ![img/start_lab.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5c4f31eeebd0a24cae6cdc2760a29cfaf04136b325a3988d6620c3cd04370dda.png).
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console. ![img/open_console.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5eaee037e6cedbf49f6a702ab8a9ef820bb8e717332946ff76a0831a6396aafc.png)
3. Click **Open Google Console**.
4. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.

If you use other credentials, you'll get errors or **incur charges**.

1. Accept the terms and skip the recovery resource page.

Do not click **End** unless you are finished with the lab or want to restart it. This clears your work and removes the project.

### Activate Google Cloud Shell

Google Cloud Shell provides command-line access to your GCP resources.

From the GCP Console click the **Cloud Shell** icon on the top right toolbar:

![Cloud Shell Icon](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/718029dee0e562c61c14536c5a636a5bae0ef5136e9863b98160d1e06123908a.png)

Then click **START CLOUD SHELL**:

![Start Cloud Shell](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/feb5ea74b4a4f6dfac7800f39c3550364ed7a33a7ab17b6eb47cab3e65c33b13.png)

You can click **START CLOUD SHELL** immediately when the dialog comes up instead of waiting in the dialog until the Cloud Shell provisions.

It takes a few moments to provision and connects to the environment:

![Cloud Shell Terminal](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/11def2e8f4cfd6f1590f3fd825d4566658501ca87e1d5d1552aa17339050c194.png)

The Cloud Shell is a virtual machine loaded with all the development tools you’ll need. It offers a persistent 5GB home directory, and runs on the Google Cloud, greatly enhancing network performance and authentication.

Once connected to the cloud shell, you'll see that you are already authenticated and the project is set to your *PROJECT_ID*:

```
gcloud auth list
```

Output:

```output
Credentialed accounts:
 - <myaccount>@<mydomain>.com (active)
```

**Note:** `gcloud` is the powerful and unified command-line tool for Google Cloud Platform. Full documentation is available on [Google Cloud gcloud Overview](https://cloud.google.com/sdk/gcloud). It comes pre-installed on Cloud Shell and supports tab-completion.

```
gcloud config list project
```

Output:

```output
[core]
project = <PROJECT_ID>
```

### Launch Google Cloud Shell Code Editor

Use the Google Cloud Shell Code Editor to easily create and edit directories and files in the Cloud Shell instance.

Once you activate the Google Cloud Shell, click the **Launch code editor**button (looks like a pencil) to open the Cloud Shell Code Editor.

![pencil_icon.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/ea6174c3216541a37e0307fa75c1b7907be2adcbcb83ebddb68fb2bc7b28f41d.png)

**Note**: The **Launch code editor** button may be off screen to the right. You may need to click the **Navigation menu** button to close the menu to see the buttons.

![pencil_icon_not_appearing.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/7baa04790fcb9cfd319be0940a2467daa30b80ad9cde90fb9b5143e3fae6b8c7.png)

You now have three interfaces available:

* The Cloud Shell Code Editor
* The Cloud Shell Command Line
* Console (By clicking on the tab). You can switch back and forth between the Console and Cloud Shell by clicking on the tab.

![cloud_shell_code_editor.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/db346792a7dc2cca318c70dde986d0bd541e2490e43fb55c96b76fba29f54779.png)

## Identify Map and Reduce operations

*Duration is 5 min*

### **Step 1**

In CloudShell clone the source repo which has starter scripts for this lab:

```
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
```

Then navigate to the code for this lab.

```
cd training-data-analyst/courses/data_analysis/lab2/python
```

### **Step 2**

Click **File** > **Refresh**.

View the source code for **is_popular.py** for the pipeline using the Cloud Shell in-browser editor or with the command line using nano:

![bfb169bc4af8c982.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/ee03b38b6572e67a54329d941ea42032120790f960a0cd7baae369ddc072e49a.png)

```
nano is_popular.py
```

```python
#!/usr/bin/env python

"""
Copyright Google Inc. 2016
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import apache_beam as beam
import argparse

def startsWith(line, term):
   if line.startswith(term):
      yield line

def splitPackageName(packageName):
   """e.g. given com.example.appname.library.widgetname
           returns com
	           com.example
                   com.example.appname
      etc.
   """
   result = []
   end = packageName.find('.')
   while end > 0:
      result.append(packageName[0:end])
      end = packageName.find('.', end+1)
   result.append(packageName)
   return result

def getPackages(line, keyword):
   start = line.find(keyword) + len(keyword)
   end = line.find(';', start)
   if start < end:
      packageName = line[start:end].strip()
      return splitPackageName(packageName)
   return []

def packageUse(line, keyword):
   packages = getPackages(line, keyword)
   for p in packages:
      yield (p, 1)

def by_value(kv1, kv2):
   key1, value1 = kv1
   key2, value2 = kv2
   return value1 < value2

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Find the most used Java packages')
   parser.add_argument('--output_prefix', default='/tmp/output', help='Output prefix')
   parser.add_argument('--input', default='../javahelp/src/main/java/com/google/cloud/training/dataanalyst/javahelp/', help='Input directory')

   options, pipeline_args = parser.parse_known_args()
   p = beam.Pipeline(argv=pipeline_args)

   input = '{0}*.java'.format(options.input)
   output_prefix = options.output_prefix
   keyword = 'import'

   # find most used packages
   (p
      | 'GetJava' >> beam.io.ReadFromText(input)
      | 'GetImports' >> beam.FlatMap(lambda line: startsWith(line, keyword))
      | 'PackageUse' >> beam.FlatMap(lambda line: packageUse(line, keyword))
      | 'TotalUse' >> beam.CombinePerKey(sum)
      | 'Top_5' >> beam.transforms.combiners.Top.Of(5, by_value)
      | 'write' >> beam.io.WriteToText(output_prefix)
   )

   p.run().wait_until_finish()


```



### **Step 3**

What custom arguments are defined? ____________________

What is the default output prefix? _________________________________________

How is the variable output_prefix in main() set? _____________________________

How are the pipeline arguments such as --runner set? ______________________

### **Step 4**

What are the key steps in the pipeline? _____________________________________________________________________________

Which of these steps happen in parallel? ____________________________________

Which of these steps are aggregations? _____________________________________

## Execute the pipeline

*Duration is 2 min*

### **Step 1**

Install the necessary dependencies for Python dataflow:

```
sudo ./install_packages.sh
```

Verify that you have the right version of pip (should be > 8.0):

```
pip -V
```

If not, open a new CloudShell tab and it should pick up the updated pip.

### **Step 2**

Run the pipeline locally:

```
./is_popular.py
```

**Note:** If you see an error that says "`No handlers could be found for logger "oauth2client.contrib.multistore_file",` you may ignore it. The error is simply saying that logging from the oauth2 library will go to stderr.

### **Step 3**

Examine the output file:

```
cat /tmp/output-*
```

## Use command line parameters

*Duration is 2 min*

### **Step 1**

Change the output prefix from the default value:

```
./is_popular.py --output_prefix=/tmp/myoutput
```

What will be the name of the new file that is written out?

### **Step 2**

Note that we now have a new file in the /tmp directory:

```
ls -lrt /tmp/myoutput*
```

## What you learned

*Duration is 1 min*

In this lab, you:

* Used pipeline options in Dataflow
* Identified Map and Reduce operations in the Dataflow pipeline

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 4: [ML on GCP C4] Computing Time-Windowed Features in Cloud Dataprep

## Overview

In this lab you will ingest, transform, and analyze a taxi cab dataset using Google Cloud Dataprep. We will calculate key reporting metrics like the average number of passengers picked up in the past hour.

### What you learn

In this lab, you:

* Build a new Flow using Cloud Dataprep
* Create and chain transformation steps with recipes
* Running Cloud Dataprep jobs (Dataflow behind-the-scenes)

Cloud Dataprep is Google's self-service data preparation tool. In this lab, you will learn how to clean and enrich multiple datasets using Cloud Dataprep.

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

1. Make sure you signed into Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, ![img/time.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/699409e014fbb8298cf5747a0535e04d13f51dfbb54fbaba57e7276dd12c6e95.png) and make sure you can finish in that time block.

There is no pause feature. You can restart if needed, but you have to start at the beginning.

1. When ready, click ![img/start_lab.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5c4f31eeebd0a24cae6cdc2760a29cfaf04136b325a3988d6620c3cd04370dda.png).
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console. ![img/open_console.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5eaee037e6cedbf49f6a702ab8a9ef820bb8e717332946ff76a0831a6396aafc.png)
3. Click **Open Google Console**.
4. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.

If you use other credentials, you'll get errors or **incur charges**.

1. Accept the terms and skip the recovery resource page.

Do not click **End** unless you are finished with the lab or want to restart it. This clears your work and removes the project.

## Create a new Storage Bucket

*Skip this section if you already have a GCS Bucket*

#### Step 1

Open the **Google Cloud Console** at [console.cloud.google.com](http://console.cloud.google.com/).

#### Step 2

Go to **Storage** in the **Navigation menu** (left-side navigation).

#### Step 3

Click **Create Bucket** (or use an existing bucket).

```bash
export BUCKET=qwiklabs-gcp-fa239fe30dfde3d8
```



#### Step 4

In the Create a bucket window that will appear, add a unique bucket name and leave the remaining settings at their default values.

#### Step 5

Click **Create**.

#### Step 6

You now have a Cloud Storage Bucket which we will be using to store raw data for ingestion into Google BigQuery later and for storing Cloud Dataprep settings.

## Create BigQuery Dataset to store Cloud Dataprep Output

#### Step 1

Open **BigQuery** at <https://console.cloud.google.com/bigquery>.

#### Step 2

In the left side bar, **click on your project** name.

#### Step 3

Click **CREATE DATASET**.

#### Step 4

For Dataset ID, type **taxi_cab_reporting** and select **Create dataset**.

Now you have a new empty dataset that we can populate with tables.

## Launch Cloud Dataprep

#### Step 1

Open the **Navigation menu**.

#### Step 2

Under Big Data, click on **Dataprep**.

#### Step 3

**Agree** to the Terms of Service.

#### Step 4

Click **Agree and Continue**.

![5eabff1419ebfaea.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/e978b730e7f561f808e295b68ab3c09c34d746ec5300b1ecb064d626d58e9300.png)

Click **Allow** for Trifacta to access project data. Dataprep is provided in collaboration with Trifacta, a Google partner. ![59ddf08c1bbcf24b.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/f4ec0926f12d9c3b81792846fd20c02ec4ac28116358d432ddff3245e4853e06.png)

#### Step 5

Click **Allow**.

![18fc12676fc080ed.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/bf512714a7bb412bd2df640719afea545929ecbd8cc02ce911f6cf597c835311.png)

#### Step 6

When prompted for "First Time Setup", click **Continue**.

#### Step 7

Wait for Cloud Dataprep to initialize (less than a minute typically).

## Import NYC Taxi Data from GCS into a Dataprep Flow

#### Step 1

In the Cloud Dataprep UI, click **Create Flow**.

#### Step 2

Specify the following **Flow details:**

| **Flow Name**               | **Flow Description**                             |
| --------------------------- | ------------------------------------------------ |
| NYC Taxi Cab Data Reporting | Ingesting, Transforming, and Analyzing Taxi Data |

Click **Create**.

If prompted, dismiss the helper tutorial.

#### Step 3

Click **Import & Add Datasets**.

#### Step 4

In the data importer left side menu, click **GCS (Google Cloud Storage)**.

#### Step 5

Click the **Pencil Icon** to edit the GCS path.

#### Step 6

Paste in the 2015 taxi rides dataset CSV from Google Cloud Storage:

```bash
gs://asl-ml-immersion/nyctaxicab/tlc_yellow_trips_2015.csv
```

Click **Go**.

#### Step 7

Before selecting Import, click the Pencil Icon to **edit the GCS path** a second time and paste in the 2016 CSV below:

```bash
gs://asl-ml-immersion/nyctaxicab/tlc_yellow_trips_2016.csv
```

Click **Go**.

#### Step 8

Click **Import & Add to Flow**.

![c84d286e0bf83367.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/327ffb9b849f67a4a92837dab3cda1a07da2b70659ae764561b338d42f3633e7.png)

#### Step 9

**Wait** for the datasets to be loaded into DataPrep.

The tool load a 10MB sample of the underlying data as well as connects to and ingests the original data source when the flow is ran.

#### Step 10

Click on the **tlc_yellow_trips_2015** icon and select **Add New Recipe**.

![739caa0b0fc2d860.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6a5ef84a7ca941ef69548252bb974c2efd8e8148b0e7741a42e6ce6cac40bfcf.png)

#### Step 11

Click **Edit Recipe**.

Wait for Dataprep to load your data sample into the explorer view

#### Step 12

In the explorer view, find the **trip_distance** column and examine the histogram.

True or False, the majority of the cab rides for 2015 were less than 5 miles.

![f183c09abec52669.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/7873c9320a0c54d360c9445ffa3092c8bc575bd8de19ee6d37f15d2683f3dcc5.png)

**True.** In our sample, 68% were between 0 to 5 miles.

Now, let's combine our 2016 and 2015 datasets.

#### Step 13

In the navigation bar, find the icon for **Union** and select it.

![cc1a658aa4b1a32e.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/b690fb3715ae953844aa7418fbf3c1694502566b4dfbd9996e0d24463bcf70a7.png)

#### Step 14

In the Union Page, click **Add data**.

In the popup window, select **tlc_yellow_trips_2016** and click **Apply**.

#### Step 15

Confirm the union looks like below (UNION DATA (2)) and then click **Add to Recipe**.

![5275e266f3a7d849.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5e7798ae00f49b76602fbf079b362fb6b4d58858d9d4a176b6b0434586ad9f20.png)

Wait for Dataprep to Apply the Union.

Now we have a single table with 2016 and 2015 taxicab data.

### **Exploring your Data**

#### Step 16

Examine the **pickup_time** histogram. Which hours had the fewest amount of pickups? The most?

In our sample, the early morning hours (3 - 4am) had the **fewest** taxicab pickups.

![8db12f8faf352ae8.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/5031b10f7affe92fff4e0d6d77c15499c2112858fa0fafdb237ffc1feca56799.png)

The most taxi cab pickups were in the evening hours with 21:00 (9pm) having slightly more than others.

![bfb3ac39d47e8a6e.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/7680c6b7fb12152c41a2c9515467c3b22bfe52cd5de7e981a003f5e5aab9b22e.png)

Is this unusual? Would you expect NYC taxi cab trips to be clustered around lunch and earlier hours in the day? Let's continue exploring.

Examine the **pickup_day** histogram. Which months and years of data do we have in our dataset?

* Only December 2015 and December 2016

![360c1c8ce7196679.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/72b76f7bf849228a1f9f5455dccfb1792a6d171c5bcbb53f9cfe57976e1e3ae6.png)

Examine the **dropoff_day** histogram. Is there anything unusual about it when compared to pickup_day? Why are there records for January 2017?

![4ad948eeb3d7aba2.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/793853f108b741c3c6ce9923603293b779ef8929538e960472dfef5e3412df7d.png)

**Answer:** There are quite a few trips that start in December and end in January (spending New Years in a taxicab!).

Next, we want to concatenate our date and time fields into a single timestamp.

#### Step 17

In the navigation bar, find **Merge columns**.

![864cafbb906a2cca.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/ef68fbe72f2e99d9a034ad0d1a24275ba00155f6a0d0d09d458eeda938eba06c.png)

For columns to merge, specify **pickup_day** and **pickup_time**.

For separator **type a single space**.

Name the new column **pickup_datetime**.

Preview and click **Add**.

![d6a4d0d6a9a348ef.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/205b83ba3af88d2751057698df777260102430efc19567a198a533d22c96f59e.png)

Confirm your new field is properly registering now as a datetime datatype (clock icon).

![9273009ca2303d8b.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/733c6b317c6d63bea355459e4eaa3a74fe74fea183a3feb220776dec4fcd3145.png)

#### Step 18

Next, we want to **create a new derived column** to count the average amount of passengers in the last hour. To do that, we need to create to get hourly data and perform a calculation.

Find the **Functions** list in the navigation bar.

Select **Dates and times**.

Select **DATEFORMAT**.

![9f634021f028bd3f.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/c098b53b0d335879781bc2b13ddac4a39a91ad4382eef6c6da29aaa64302964d.png)

In the formula, paste the following which will truncate the pickup time to just the hour:

```bash
DATEFORMAT(pickup_datetime,"yyyyMMddHH")
```

![8a396bfb77d9854e.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/267a7462b6bd067385ed150847d7d6c5afb909e7205d452ae4b19b9991005f2a.png)

Specify the New Column as **hour_pickup_datetime**.

Confirm the new derived column is shown correctly in the **preview**.

![91d6c220515efcc1.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/c12a4744760068864c8f06505463a5d07341615c7b163b70c3474d7a8f27f674.png)

Click **Add**.

#### Step 19

In order to get the field properly recognized as a DATETIME data type, we are going to add back zero minutes and zero seconds through a MERGE concatenation.

In the navigation bar, find **Merge columns**.

![864cafbb906a2cca](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/ef68fbe72f2e99d9a034ad0d1a24275ba00155f6a0d0d09d458eeda938eba06c.png)

For columns to merge, specify **hour_pickup_datetime** and **'0000'**.

Name the column to **pickup_hour**.

![811495c4ff7e9022.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/71383e2ea866fa35616426fd7d401d75051d37b6837911ec21a555fee422f309.png)

Click **Add**.

We now have our taxicab hourly pickup column. Next, we will calculate the average count of passengers over the past hour. We will do this through aggregations and a rolling window average function.

#### Step 20

In the navigation toolbar select **Functions** > **Aggregation** > **AVERAGE**.

![ad1bb11a7fba20b0.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/f01c4a917cdab0c5abf632dc1ac8e305a7fd8b2f05cdc16f4cbbe88758f5fb79.png)

For **Formula**, specify:

```bash
AVERAGE(fare_amount)
```

For **Sort** rows, specify:

```bash
pickup_datetime
```

For **Group by**, specify:

```bash
pickup_hour
```

![7f4646dc02c251c9.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/7110d57ca3f56305b8286c9acbd77cf740b3fb9407ceae97790e10f4a1d15478.png)

Click **Add**.

We now have our average cab fares statistic.

#### Step 21

Explore the **average_fare_amount** histogram. Is there a range of fares that are most common?

![db1b0979177de255.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/a28fb1c050cdf8e12b8cebeb803a3e4fe114d06ff89c90ac7598000c646030f0.png)

In our sample, most NYC cab fares are in the $18-19 range.

Next, we want to calculate a rolling window of average fares over the past 3 hours.

#### Step 22

In the navigation toolbar, select **Functions** > **Window** > **ROLLINGAVERAGE**.

![3ba77249ce0b96b1.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/0776b22c8a75cb0a63ed0f3b12cedb278853f4b4ac41cab0b5b803b0959f75e3.png)

Copy in the below formula which computes the rolling average of passenger count for the last hour.

Formula:

```bash
ROLLINGAVERAGE(average_fare_amount, 3, 0)
```

Sort rows by:

```bash
-pickup_hour
```

Note that we are sorting recent taxicab rides first (the negative sign -**pickup_hour**indicates descending order) and operating over a rolling 3 hour period.

![5d65593d8ab10586.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/e533e8a1870147dc92944ed051a7d9a74de90a81764dbc8229a9641ce99c45f5.png)

Click **Add**.

#### Step 23

Toggle open the **Recipe icon** to preview your final transformation steps.

![ac7bfdfafab6e41.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/d71c7cae28461091c02d9c0e8f93c77c0a435568d762a32ab5314bc1bb1224b9.png)

![775040262524cfd8.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/7488e0134fbc1735faffb145bcfe48cfbffe5f719245d632daf1a88f8899398d.png)

#### Step 24

Click **Run Job**.

#### Step 25

In **Publishing Actions page**, under Settings, edit the path by clicking the pencil icon

![770160f86eb41a96.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/6fcfbd258aa17f5ca8889097f335f67413fe03fed12608fcee273360b40b326c.png)

Choose **BigQuery** and choose your **taxi_cab_reporting** BigQuery dataset where you want to create the output table.

(**Note:** if you do not see a taxi_cab_reporting dataset, refer to the start of this lab for instructions on how to create it in BigQuery)

Choose **Create a new table**.

Name the table **tlc_yellow_trips_reporting**.

Choose **Drop the table every run**.

Select **Update**.

#### Step 26

Select **Run Job**.

#### Step 27

Optional: View the Cloud Dataflow Job by selecting [...] and **View dataflow job**.

![64c9d114b1186328.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/31802b5ee2a3cb80d7f13bee946dbcc4794b31b419f0493602872710103dd8ae.png)

Wait for your Cloud Dataflow job to complete and confirm your new new table shows in BigQuery.

#### Step 28

While your Cloud Dataprep flow starts and manages your Cloud Dataflow job, you can see the data results by running this pre-ran query in BigQuery:

```sql
#standardSQL
SELECT
  pickup_hour,
  FORMAT("$%.2f",ROUND(average_3hr_rolling_fare,2)) AS avg_recent_fare,
  ROUND(average_trip_distance,2) AS average_trip_distance_miles,
  FORMAT("%'d",sum_passenger_count) AS total_passengers_by_hour
FROM
  `asl-ml-immersion.demo.nyc_taxi_reporting`
ORDER BY
  pickup_hour DESC;
```

Extra credit:

You can schedule Cloud Dataprep jobs to run at set intervals. Select a flow and click [...] and **Schedule Flow**.

![schedule_flow.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/abbf030adc8f22ab01d4a7d8a1db1a44ae2c3aeb56c087ef1c0eec366b80995d.png)

Congratulations! You have now built a data transformation pipeline using the Cloud Dataprep UI.

For full documentation and additional tutorials, refer to the [Cloud Dataprep support page](https://cloud.google.com/dataprep/).

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.





# Lab 5: ML on GCP [v1.0] - C4 - Improve ML model with Feature Engineering

## Overview

*Duration is 1 min*

This lab is part of a lab series, where you go from exploring a taxicab dataset to training and deploying a high-accuracy distributed model with Cloud ML Engine.

### **What you learn**

In this lab, you will improve the ML model using feature engineering. In the process, you will learn how to:

* Work with feature columns
* Add feature crosses in TensorFlow
* Read data from BigQuery
* Create datasets using Dataflow
* Use a wide-and-deep model

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

1. Make sure you signed in to Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, 01:30:00) and make sure you can finish in that time block.

There is no pause feature. You can restart if needed, but you have to start at the beginning.

1. When ready, click **Start Lab.**
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console.![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87417/original/img/eb2f9b843e5c184c.png)
3. Click **Open Google Console**.
4. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.

If you use other credentials, you'll get errors or **incur charges**.

1. Accept the terms and skip the recovery resource page.

Do not click **End** unless you are finished with the lab or want to restart it. This clears your work and removes the project.

## Launch Cloud Datalab

To launch Cloud Datalab:

### **Step 1**

Open Cloud Shell. In the upper-right, click **Activate Cloud Shell** (![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87417/original/img/d258c43ee8321ba.png)). And then click **Start Cloud Shell**.

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

Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. After clicking on "Web preview", click on "Change port" and change the port number to 8081. Click "Change and Preview" to open the Datalab web interface.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87417/original/img/a10fdc06c9fd5db0.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87417/original/img/9ce67b5f42c53a37.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87417/original/img/3f04d163f957a625.png)

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell.

## Clone course repo within your Datalab instance

To clone the course repo in your Datalab instance:

### **Step 1**

In Cloud Datalab home page (browser), navigate into "**notebooks**" and add a new notebook using the icon ![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87417/original/img/5fdee4bbcdee4b9a.png) on the top left.

### **Step 2**

Rename this notebook as ‘**repocheckout**'.

### **Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87417/original/img/1a3d26cf1e7e1e5f.png)

### **Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87417/original/img/ce070262a667324b.png)

## Feature Engineering

*Duration is 15 min*

### **Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab/notebooks/training-data-analyst/courses/machine_learning/deepdive/04_features/taxifeateng/** and open **feateng.ipynb**.

**Note**: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

### **Step 2**

In Datalab, click on **Clear**, then click on **Clear | All Cells**. Now read the narrative and execute each cell in turn.

The solution video will demo notebooks that contain hyper-parameter tuning and training on 500 million rows of data. The changes to the model are minor -- essentially just command-line parameters, but the impact on model accuracy is huge:

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87417/original/img/21d055d73bf7974e.png)





## Code Additions

Add to notebook:

```python
## add to notebook:

We have two new inputs in the INPUT_ColumNS, three engineered features, and the estimator involves bucketization and feature crosses:

!grep -A 20 "INPUT_COLUMNS = " taxifare/trainer/model.py
!grep -A 20 "build_estimator" taxifare/trainer/model.py
!grep -A 20 "add_engineered(" taxifare/trainer/model.py


## model.py: 

* check for [[here]]

def serving_input_fn():
	feature_placehodlers = {
		column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS[2:]
	}
	feature_placeholders['dayofweek'] = tf.placeholder(tf.string, [None])
	feature_placeholders['hourofday'] = tf.placeholder(tf.int32, [None])

	features = {
		key: tf.expand_dims(tensor, -1) for key, tensor in feature_placeholders.items()
	}
	return tf.estimator.export.ServingInputReceiver(
		add_engineered(features),
		feature_placeholders
	)
```



Model.py:

```python
#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = 'fare_amount,dayofweek,hourofday,pickuplon,pickuplat,dropofflon,dropofflat,passengers,key'.split(',')
LABEL_COLUMN = 'fare_amount'
KEY_FEATURE_COLUMN = 'key'
DEFAULTS = [[0.0], ['Sun'], [0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

# These are the raw input columns, and will be provided for prediction also
INPUT_COLUMNS = [
    # TODO: Define feature columns for dayofweek, hourofday, pickuplon, pickuplat, dropofflat, dropofflon, passengers

    ## categorical columns:
    tf.feature_column.categorical_column_with_vocabulary_list(
        'dayofweek', 
        vocabulary_list = ['Sun', 'Mon', 'Tues', 'Wed', 'Thu', 'Fri', 'Sat']),
    tf.feature_column.categorical_column_with_identity('hourofday', num_buckets = 24),
    
    ## numeric columns:
    tf.feature_column.numeric_column('pickuplat'),
    tf.feature_column.numeric_column('pickuplon'),
    tf.feature_column.numeric_column('dropofflat'),
    tf.feature_column.numeric_column('dropofflon'),
    tf.feature_column.numeric_column('passengers'),
    
    # TODO: Add any engineered columns here
    tf.feature_column.numeric_column('latdiff'),
    tf.feature_column.numeric_column('londiff'),
    tf.feature_column.numeric_column('euclidean')
]

# Build the estimator
def build_estimator(model_dir, nbuckets, hidden_units):
    """
     TODO: Build an estimator starting from INPUT COLUMNS.
     These include feature transformations and synthetic features.
     The model is a wide-and-deep model, i.e. [wide_cols] & [deep_cols].
    """
    
    ## [[here]] -- change "euclidean" to "pcount" (12:49)

    ## retrieve input features into separate variables:
    #(dayofweek, hourofday, plat, plon, dlat, dlon, pcount, latdiff, londiff, euclidean) = INPUT_COLUMNS
    (dayofweek, hourofday, plat, plon, dlat, dlon, pcount, latdiff, londiff, pcount) = INPUT_COLUMNS
    
    ## transform features: 
    ## bucketize lats & lons:
    latbuckets = np.linspace(38.0, 42.0, nbuckets).tolist()
    lonbuckets = np.linspace(-76.0, -72.0, nbuckets).tolist()
    b_plat = tf.feature_column.bucketized_column(plat, latbuckets)
    b_dlat = tf.feature_column.bucketized_column(dlat, latbuckets)
    b_plon = tf.feature_column.bucketized_column(plon, lonbuckets)
    b_dlon = tf.feature_column.bucketized_column(dlon, lonbuckets)
    
    ## feature crosses (a.k.a. interactions):
    ploc = tf.feature_column.crossed_column([b_plat, b_plon], nbuckets ** 2)
    dloc = tf.feature_column.crossed_column([b_dlat, b_dlon], nbuckets ** 2)
    pd_pair = tf.feature_column.crossed_column([ploc, dloc], nbuckets ** 4)
    day_hr = tf.feature_column.crossed_column([dayofweek, hourofday], 24 * 7)
    
    ## define linear features (a.k.a. wide columns):
    linear_feature_columns = [
        ## feature crosses:
        dloc, ploc, pd_pair, day_hr,
        ## sparse columns:
        dayofweek, hourofday,
        ## anything with a linear relationship:
        pcount
    ]
    ## define dnn features (a.k.a. deep columns):
    dnn_feature_columns = [
        ## embedding column to "group" similar columns of feature crosses:
        tf.feature_column.embedding_column(pd_pair, 10),
        tf.feature_column.embedding_column(day_hr, 10),
        ## numeric columns:
        plat, plon, dlat, dlon,
        latdiff, londiff, euclidean
    ]
    
    ## define run config for a much longer interval
    run_config = tf.estimator.RunConfig(save_checkpoints_secs = 30,
                                        keep_checkpoint_max = 3)
    ## define DNN wide & deep estimator:
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir = model_dir,
        linear_feature_columns = linear_feature_columns, ## wide features
        dnn_feature_columns = dnn_feature_columns,       ## deep features
        dnn_hidden_units = hidden_units,
        config = run_config
    )
    # add extra evaluation metric for hyperparameter tuning
    #estimator = tf.contrib.estimator.add_metrics(estimator, add_eval_metrics)

    return estimator # TODO: Add estimator definition here

# Create feature engineering function that will be used in the input and serving input functions
def add_engineered(features):
    # TODO: Add any engineered features to the dict
    lat1 = features['pickuplat']
    lat2 = features['dropofflat']
    lon1 = features['pickuplon']
    lon2 = features['dropofflon']
    latdiff = (lat1 - lat2)
    londiff = (lon1 - lon2)
    
    ## add features to feature vector
    ## for distance with sign that indicates direction:
    features['latdiff'] = latdiff
    features['londiff'] = londiff
    dist = tf.sqrt(latdiff * latdiff + londiff * londiff)
    features['euclidean'] = dist
    
    return features   

# # Create serving input function to be able to serve predictions
# def serving_input_fn():
#     # ## code in prepared file for lab:
#     # feature_placeholders = {  
#     #     # TODO: What features will user provide? What will their types be?
#     #     ## numeric features:
#     #     column.name : tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS[2:7]
#     # }
#     ## code trials:
#     feature_placeholders = {}
#     ## non-numeric features:
#     feature_placeholders['dayofweek'] = tf.placeholder(tf.string, [None])
#     feature_placeholders['hourofday'] = tf.placeholder(tf.int32, [None])
#     ## numeric features:
#     feature_placeholders['pickuplat'] = tf.placeholder(tf.float32, [None])
#     feature_placeholders['pickuplon'] = tf.placeholder(tf.float32, [None])
#     feature_placeholders['dropofflat'] = tf.placeholder(tf.float32, [None])
#     feature_placeholders['dropofflon'] = tf.placeholder(tf.float32, [None])
#     feature_placeholders['passengers'] = tf.placeholder(tf.float32, [None])
# 
#     # TODO: Add any extra placeholders for inputs you'll generate
#     features = add_engineered(feature_placeholders.copy())
# 
#     # ## [[?]] this part is not part of the model solution... 
#     # what is going on? --> hence, commented out.
#     # features = {
#     #     key: tf.expand_dims(tensor, -1)
#     #     for key, tensor in feature_placeholders.items()
#     # }
#     return tf.estimator.export.ServingInputReceiver(
#       features, # TODO: Wrap this with a call to add_engineered
#       feature_placeholders
#     )

## new implementation -- [[here]]:
## similar to course video:

# Create serving input function to be able to serve predictions
def serving_input_fn():
    feature_placeholders = {  
        ## numeric features:
        ## (ignoring the first two columns):
        column.name : tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS[2:]
    }
    feature_placeholders['dayofweek'] = tf.placeholder(tf.string, [None])
    feature_placeholders['hourofday'] = tf.placeholder(tf.int32, [None])
    ## no add engineered here, but below
    ## needs to be done _after_ the features are created!!!

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(
      add_engineered(features), 
      feature_placeholders
    )


# Create input function to load data into datasets
def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults = DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return add_engineered(features), label # TODO: Wrap this with a call to add_engineered
        
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
        batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()
        return batch_features, batch_labels
    return _input_fn

# Create estimator train and evaluate function
def train_and_evaluate(args):
    estimator = build_estimator(args['output_dir'], args['nbuckets'], args['hidden_units'])
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(
            filename = args['train_data_paths'],
            mode = tf.estimator.ModeKeys.TRAIN,
            batch_size = args['train_batch_size']),
        max_steps = args['train_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(
            filename = args['eval_data_paths'],
            mode = tf.estimator.ModeKeys.EVAL,
            batch_size = args['eval_batch_size']),
        steps = None,
        exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


## new implementation copied from coursera course video (18:45):
def train_and_evaluate(args):
    estimator = build_estimator(args['output_dir'], args['nbuckets'], args['hidden_units'].split(' '))
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(
            filename = args['train_data_paths'],
            mode = tf.estimator.ModeKeys.TRAIN,
            batch_size = args['train_batch_size']),
        max_steps = args['train_steps'])
    exporter = tf.estimator.LatestExporter('exporter, serving_input_fn')
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(
            filename = args['eval_data_paths'],
            mode = tf.estimator.ModeKeys.EVAL,
            batch_size = args['eval_batch_size']),
        steps = None,
        expoerters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def get_eval_metrics():
    return {
        'rmse': tflearn.MetricSpec(metric_fn = metrics.streaming_root_mean_squared_error),
        'training/hptuning/metric': tflearn.MetricSpec(metric_fn = metrics.streaming_root_mean_squared_error),
    }
```



# Lab 6: ML on GCP [v1.0] - C4 - Exploring tf.transform

## Overview

*Duration is 1 min*

tf.transform allows users to define preprocessing pipelines and run these using large scale data processing frameworks, while also exporting the pipeline in a way that can be run as part of a TensorFlow graph

### **What you learn**

* Implement feature preprocessing and feature creation using tf.transform
* Carry out feature processing efficiently, at scale and on streaming data

## Setup

For each lab, you get a new GCP project and set of resources for a fixed time at no cost.

1. Make sure you signed in to Qwiklabs using an **incognito window**.
2. Note the lab's access time (for example, 01:30:00) and make sure you can finish in that time block.

There is no pause feature. You can restart if needed, but you have to start at the beginning.

1. When ready, click **Start Lab.**
2. Note your lab credentials. You will use them to sign in to Cloud Platform Console.![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87424/original/img/eb2f9b843e5c184c.png)
3. Click **Open Google Console**.
4. Click **Use another account** and copy/paste credentials for **this** lab into the prompts.

If you use other credentials, you'll get errors or **incur charges**.

1. Accept the terms and skip the recovery resource page.

Do not click **End** unless you are finished with the lab or want to restart it. This clears your work and removes the project.

## Launch Cloud Datalab

To launch Cloud Datalab:

### **Step 1**

Open Cloud Shell. In the upper-right, click **Activate Cloud Shell** (![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87424/original/img/d258c43ee8321ba.png)). And then click **Start Cloud Shell**.

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

Click on the **Web Preview** icon on the top-right corner of the Cloud Shell ribbon. After clicking on "Web preview", click on "Change port" and change the port number to 8081. Click "Change and Preview" to open the Datalab web interface.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87424/original/img/a10fdc06c9fd5db0.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87424/original/img/9ce67b5f42c53a37.png)

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87424/original/img/3f04d163f957a625.png)

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell.

## Clone course repo within your Datalab instance

To clone the course repo in your Datalab instance:

### **Step 1**

In Cloud Datalab home page (browser), navigate into "**notebooks**" and add a new notebook using the icon ![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87424/original/img/5fdee4bbcdee4b9a.png) on the top left.

### **Step 2**

Rename this notebook as ‘**repocheckout**'.

### **Step 3**

In the new notebook, enter the following commands in the cell, and click on **Run** (on the top navigation bar) to run the commands:

```
%bash
git clone https://github.com/GoogleCloudPlatform/training-data-analyst
rm -rf training-data-analyst/.git
```

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87424/original/img/1a3d26cf1e7e1e5f.png)

### **Step 4**

Confirm that you have cloned the repo by going back to Datalab browser, and ensure you see the **training-data-analyst** directory. All the files for all labs throughout this course are available in this directory.

![img](https://run-qwiklab-website-prod.s3.amazonaws.com/instructions/documents/87424/original/img/ce070262a667324b.png)

## Exploring tf.transform

### **Step 1**

In Cloud Datalab, click on the Home icon, and then navigate to **notebooks/training-data-analyst > courses > machine_learning > deepdive > 04_features > taxifeateng** and open **tftransform.ipynb**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

### **Step 2**

In Datalab, click on **Clear**, then click on **Clear | All Cells**. Now read the narrative and execute each cell in turn.





# [[eof]]



[[todo]]:

* make pdfs of all notebooks in all courses! 
* make python and html files of all notebooks in all courses!
* make pdfs of all .md-files in all courses!