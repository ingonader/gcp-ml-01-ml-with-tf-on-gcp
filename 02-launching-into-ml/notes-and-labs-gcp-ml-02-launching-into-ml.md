# Launching into ML



## to download materials

software to download materials: <https://github.com/coursera-dl/coursera-dl>

link to coursera material: <https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp/lecture/wWBkE/effective-ml>

```bash
coursera-dl -u ingo.nader@gmail.com launching-machine-learning --download-quizzes --download-notebooks --about
```



## Notes

* Do labs in Google Chrome (recommended)
* Open Incognito window, go to <http://console.cloud.google.com/>



# Lab 1: Creating repeatable splits in BigQuery

## Overview

*Duration is 1 min*

Repeatability is important in machine learning. If you do the same thing now and 5 minutes from now and get different answers, then it makes experimentation is difficult. In other words, you will find it difficult to gauge whether a change you made has resulted in an improvement or not.

### What you need

* Logged into GCP Console with your Qwiklabs generated account

### What you learn

In this lab, you will:

* Explore the impact of different ways of creating machine learning datasets.

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

## Creating Repeatable Dataset splits

*Duration is 15 min*

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 02_generalization** and open **repeatable_splitting.ipynb**.

![4ae227dbe1775948.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/fde9842df85c72a22cc25f96ef8ec97d345d8046837896f7162ffe3133dc2520.png)

This notebook is your cloned copy of the one in the course repository. Any code changes you make and run here are reflected only in your working copy (until you share with others). This way you can edit and run the code to practice the concepts.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

In Datalab, click on **Clear | Clear All Cells**. Now read the narrative and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 2: Exploring and creating ML datasets

## Overview

*Duration is 1 min*

This lab is part of a lab series, where you go from exploring a taxicab dataset to training and deploying a high-accuracy distributed model with Cloud ML Engine.

### What you need

* Logged into GCP Console with your Qwiklabs generated account

### What you learn

In this lab, you will:

* explore a dataset using BigQuery and Datalab
* sample the dataset and create training, validation, and testing datasets for local development of TensorFlow models
* create a benchmark to evaluate the performance of ML against

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

## Explore dataset, create ML datasets, create benchmark

*Duration is 15 min*

### Step 1

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 02_generalization > labs** and open **create_datasets.ipynb**.

![c695d2df3b148288.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/e6cc75375dc07b9985c1f7cdb0f14e7f7c6aa13579c07bdd082ccefe9dfce34f.png)

This notebook is your cloned copy of the one in the course repository. Any code changes you make and run here are reflected only in your working copy (until you share with others). This way you can edit and run the code to practice the concepts.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

### Step 2

In Datalab, click on **Clear | Clear all Cells**. Now read the narrative and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.









# [[eof]]