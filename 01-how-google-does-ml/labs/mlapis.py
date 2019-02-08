
# coding: utf-8

# <h1> Using Machine Learning APIs </h1>
# 
# First, visit <a href="http://console.cloud.google.com/apis">API console</a>, choose "Credentials" on the left-hand menu.  Choose "Create Credentials" and generate an API key for your application. You should probably restrict it by IP address to prevent abuse, but for now, just  leave that field blank and delete the API key after trying out this demo.
# 
# Copy-paste your API Key here:

# In[1]:


APIKEY="AIzaSyCedn34DrOD6c8ybV-1LmHS1CMtSGhX7EA"  # Replace with your API key


# <b> Note: Make sure you generate an API Key and replace the value above. The sample key will not work.</b>
# 
# From the same API console, choose "Dashboard" on the left-hand menu and "Enable API".
# 
# Enable the following APIs for your project (search for them) if they are not already enabled:
# <ol>
# <li> Google Translate API </li>
# <li> Google Cloud Vision API </li>
# <li> Google Natural Language API </li>
# <li> Google Cloud Speech API </li>
# </ol>
# 
# Finally, because we are calling the APIs from Python (clients in many other languages are available), let's install the Python package (it's not installed by default on Datalab)

# In[2]:


get_ipython().system('pip install --upgrade google-api-python-client')


# <h2> Invoke Translate API </h2>

# In[3]:


# running Translate API
from googleapiclient.discovery import build
service = build('translate', 'v2', developerKey=APIKEY)

# use the service
inputs = ['is it really this easy?', 'amazing technology', 'wow']
outputs = service.translations().list(source='en', target='fr', q=inputs).execute()
# print outputs
for input, output in zip(inputs, outputs['translations']):
  print("{0} -> {1}".format(input, output['translatedText']))


# <h2> Invoke Vision API </h2>
# 
# The Vision API can work off an image in Cloud Storage or embedded directly into a POST message. I'll use Cloud Storage and do OCR on this image: <img src="https://storage.googleapis.com/cloud-training-demos/vision/sign2.jpg" width="200" />.  That photograph is from http://www.publicdomainpictures.net/view-image.php?image=15842
# 

# In[4]:


# Running Vision API
import base64
IMAGE="gs://cloud-training-demos/vision/sign2.jpg"
vservice = build('vision', 'v1', developerKey=APIKEY)
request = vservice.images().annotate(body={
        'requests': [{
                'image': {
                    'source': {
                        'gcs_image_uri': IMAGE
                    }
                },
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': 3,
                }]
            }],
        })
responses = request.execute(num_retries=3)
print(responses)


# In[5]:


foreigntext = responses['responses'][0]['textAnnotations'][0]['description']
foreignlang = responses['responses'][0]['textAnnotations'][0]['locale']
print(foreignlang, foreigntext)


# <h2> Translate sign </h2>

# In[6]:


inputs=[foreigntext]
outputs = service.translations().list(source=foreignlang, target='en', q=inputs).execute()
# print(outputs)
for input, output in zip(inputs, outputs['translations']):
  print("{0} -> {1}".format(input, output['translatedText']))


# <h2> Sentiment analysis with Language API </h2>
# 
# Let's evaluate the sentiment of some famous quotes using Google Cloud Natural Language API.

# In[7]:


lservice = build('language', 'v1beta1', developerKey=APIKEY)
quotes = [
  'To succeed, you must have tremendous perseverance, tremendous will.',
  'It’s not that I’m so smart, it’s just that I stay with problems longer.',
  'Love is quivering happiness.',
  'Love is of all passions the strongest, for it attacks simultaneously the head, the heart, and the senses.',
  'What difference does it make to the dead, the orphans and the homeless, whether the mad destruction is wrought under the name of totalitarianism or in the holy name of liberty or democracy?',
  'When someone you love dies, and you’re not expecting it, you don’t lose her all at once; you lose her in pieces over a long time — the way the mail stops coming, and her scent fades from the pillows and even from the clothes in her closet and drawers. '
]
for quote in quotes:
  response = lservice.documents().analyzeSentiment(
    body={
      'document': {
         'type': 'PLAIN_TEXT',
         'content': quote
      }
    }).execute()
  polarity = response['documentSentiment']['polarity']
  magnitude = response['documentSentiment']['magnitude']
  print('POLARITY=%s MAGNITUDE=%s for %s' % (polarity, magnitude, quote))


# <h2> Speech API </h2>
# 
# The Speech API can work on streaming data, audio content encoded and embedded directly into the POST message, or on a file on Cloud Storage. Here I'll pass in this <a href="https://storage.googleapis.com/cloud-training-demos/vision/audio.raw">audio file</a> in Cloud Storage.

# In[8]:


sservice = build('speech', 'v1beta1', developerKey=APIKEY)
response = sservice.speech().syncrecognize(
    body={
        'config': {
            'encoding': 'LINEAR16',
            'sampleRate': 16000
        },
        'audio': {
            'uri': 'gs://cloud-training-demos/vision/audio.raw'
            }
        }).execute()
print(response)


# In[9]:


print(response['results'][0]['alternatives'][0]['transcript'])
print('Confidence=%f' % response['results'][0]['alternatives'][0]['confidence'])


# <h2> Clean up </h2>
# 
# Remember to delete the API key by visiting <a href="http://console.cloud.google.com/apis">API console</a>.
# 
# If necessary, commit all your notebooks to git.
# 
# If you are running Datalab on a Compute Engine VM or delegating to one, remember to stop or shut it down so that you are not charged.
# 

# ## Challenge Exercise
# 
# Here are a few portraits from the Metropolitan Museum of Art, New York (they are part of a [BigQuery public dataset](https://bigquery.cloud.google.com/dataset/bigquery-public-data:the_met) ):
# 
# * gs://cloud-training-demos/images/met/APS6880.jpg
# * gs://cloud-training-demos/images/met/DP205018.jpg
# * gs://cloud-training-demos/images/met/DP290402.jpg
# * gs://cloud-training-demos/images/met/DP700302.jpg
# 
# Use the Vision API to identify which of these images depict happy people and which ones depict unhappy people.
# 
# Hint (highlight to see): <p style="color:white">You will need to look for joyLikelihood and/or sorrowLikelihood from the response.</p>

# Copyright 2018 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.