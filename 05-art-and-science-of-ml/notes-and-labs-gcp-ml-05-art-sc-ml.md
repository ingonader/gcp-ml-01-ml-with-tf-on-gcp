# Art and Science of Machine Learning (C05)



## to download materials

software to download materials: <https://github.com/coursera-dl/coursera-dl>

link to coursera material: <https://www.coursera.org/learn/end-to-end-ml-tensorflow-gcp/lecture/wWBkE/effective-ml>

```bash
coursera-dl -u "ingo.nader@gmail.com" art-science-ml --download-quizzes --download-notebooks --about
```



## Notes

* Do labs in Google Chrome (recommended)
* Open Incognito window, go to <http://console.cloud.google.com/>



# Lab 1: [ML on GCP C5] Improve model accuracy by hand-tuning hyperparameters

## Overview

*Duration is 1 min*

This lab is part of a lab series, where you go from exploring a taxicab dataset to training and deploying a high-accuracy distributed model with Cloud ML Engine.

### What you learn

In this lab, you will improve accuracy by hand-tuning its hyperparameters. In the process, you will learn how to:

* Use the `LinearRegressor` class in TensorFlow
* Evaluate the accuracy of a model's predictions using Root Mean Squared Error (RMSE)

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

## Hand-tuning hyperparameters

*Duration is 15 min*

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 05_artandscience > labs** and open **a_handtuning.ipynb**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

In Datalab, click on **Clear | Clear all Cells**. Now read the narrative and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 2: [ML on GCP C5] Improve model accuracy by hyperparameter tuning with Cloud MLE

## Overview

*Duration is 1 min*

This lab is part of a lab series, where you go from exploring a taxicab dataset to training and deploying a high-accuracy distributed model with Cloud ML Engine.

### **What you learn**

In this lab, you will improve accuracy of a model by hyperparameter tuning with Cloud MLE

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

## Create Storage Bucket

*Duration is 2 min*

Create a bucket using the GCP console:

**Step 1**

In your GCP Console, click on the **Navigation menu** (![menu.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/2f22c727923782d61d44dd691c367624adaf6ddd6c33a5b6be24f43b3c913d3b.png)), and select **Storage**.

**Step 2**

Click on **Create bucket**.

**Step 3**

Choose a Regional bucket and set a unique name (use your project ID because it is unique). Then, click **Create**.

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

## Enable APIs

On the **Navigation menu** (![Navigation menu](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/b64830d530e08f843e60a4145b88d415dd0ee4e10a9543014586e1942ac5d166.png)), click **APIs & services**.

Scroll down and confirm that your APIs are enabled.

If an API is missing, click **ENABLE APIS AND SERVICES** at the top, search for the API by name, and enable it for your project.

* [Cloud ML Engine](https://console.cloud.google.com/apis/library/ml.googleapis.com/?q=cloud%20ml%20engine)

## Hyperparameter Tuning

*Duration is 15 min*

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebook > training-data-analyst > courses > machine_learning > deepdive > 05_artandscience > labs** and open **b_hyperparam.ipynb**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

In Datalab, click on **Clear | Clear all Cells**. Now read the narrative and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.





# Lab 3: [ML on GCP C5] Using Neural Networks to build ML model

## Overview

*Duration is 1 min*

This lab is part of a lab series, where you go from exploring a taxicab dataset to training and deploying a high-accuracy distributed model with Cloud ML Engine.

### **What you learn**

In this lab, you will use the `DNNRegressor` class in TensorFlow to predict median housing price

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

## Neural Networks

*Duration is 15 min*

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 05_artandscience > labs** and open **c_neuralnetwork.ipynb**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

In Datalab, click on **Clear | Clear all Cells**. Now read the narrative and execute each cell in turn.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.



# Lab 4: [ML on GCP C5] Using Custom Estimators

## Overview

*Duration is 1 min*

In this lab you will build a custom estimator.

### What you learn

In this lab, you will learn how to build a custom estimator.

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

## Create Storage Bucket

*Duration is 2 min*

Create a bucket using the GCP console:

**Step 1**

In your GCP Console, click on the **Navigation menu** (![menu.png](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/2f22c727923782d61d44dd691c367624adaf6ddd6c33a5b6be24f43b3c913d3b.png)), and select **Storage**.

**Step 2**

Click on **Create bucket**.

**Step 3**

Choose a Regional bucket and set a unique name (use your project ID because it is unique). Then, click **Create**.

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

## Enable APIs

On the **Navigation menu** (![Navigation menu](https://gcpstaging-qwiklab-website-prod.s3.amazonaws.com/bundles/assets/b64830d530e08f843e60a4145b88d415dd0ee4e10a9543014586e1942ac5d166.png)), click **APIs & Services**.

Scroll down and confirm that below API is enabled.

* Cloud Machine Learning Engine

If the API is missing, click **ENABLE APIS AND SERVICES** at the top, search for the API by name, and enable it for your project.

## Custom Estimator

*Duration is 15 min*

**Step 1**

In Cloud Datalab, click on the **Home** icon, and then navigate to **datalab > notebooks > training-data-analyst > courses > machine_learning > deepdive > 05_artandscience > labs** and open **d_customestimator.ipynb**.

Note: If the cloud shell used for running the datalab command is closed or interrupted, the connection to your Cloud Datalab VM will terminate. If that happens, you may be able to reconnect using the command ‘**datalab connect mydatalabvm**' in your new Cloud Shell. Once connected, try the above step again.

**Step 2**

In Datalab, click on **Clear | Clear all Cells**. Now read the narrative and execute each cell in turn.

**Step 3**

Compare to solution

In the parent level 05_artandscience folder (above labs/) there is the solution notebook with the same title for comparison.

## End your lab

When you have completed your lab, click **End Lab**. Qwiklabs removes the resources you’ve used and cleans the account for you.









# [[eof]]



[[todo]]:

* make python and html files of all notebooks in all courses!
* make pdfs of all .md-files in all courses! (add to .gitignore)