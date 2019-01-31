
# coding: utf-8

# <h1> Getting started with TensorFlow </h1>
# 
# In this notebook, you play around with the TensorFlow Python API.

# In[1]:


import tensorflow as tf
import numpy as np

print(tf.__version__)


# <h2> Adding two tensors </h2>
# 
# First, let's try doing this using numpy, the Python numeric package. numpy code is immediately evaluated.

# In[7]:


a = np.array([5, 3, 8])
b = np.array([3, -1, 2])
c = np.add(a, b)
print(c)
print(c.shape)


# The equivalent code in TensorFlow consists of two steps:
# <p>
# <h3> Step 1: Build the graph </h3>

# In[9]:


a = tf.constant([5, 3, 8])
b = tf.constant([3, -1, 2])
c = tf.add(a, b)
print(c)


# c is an Op ("Add") that returns a tensor of shape (3,) and holds int32. The shape is inferred from the computation graph.
# 
# Try the following in the cell above:
# <ol>
# <li> Change the 5 to 5.0, and similarly the other five numbers. What happens when you run this cell? </li>
# <li> Add an extra number to a, but leave b at the original (3,) shape. What happens when you run this cell? </li>
# <li> Change the code back to a version that works </li>
# </ol>
# 
# <p/>
# <h3> Step 2: Run the graph

# In[10]:


with tf.Session() as sess:
  result = sess.run(c)
  print(result)


# <h2> Using a feed_dict </h2>
# 
# Same graph, but without hardcoding inputs at build stage

# In[18]:


a = tf.placeholder(dtype=tf.int32, shape=(None,))  # batchsize x scalar
b = tf.placeholder(dtype=tf.int32, shape=(None,))
c = tf.add(a, b)
## trying expand_dims:
d = tf.expand_dims(c, axis = 0)
e = tf.expand_dims(c, axis = 1)

with tf.Session() as sess:
  result = sess.run([c, d, e], feed_dict={
      a: [3, 4, 5],
      b: [-1, 2, 3]
    })
  #print(result)
  
print("Prettier results (one result per line):")
for i in range(len(result)):
    print(result[i])


# <h2> Heron's Formula in TensorFlow </h2>
# 
# The area of triangle whose three side lengths are $(a, b, c)$ is $\sqrt{s(s-a)(s-b)(s-c)}$ where $s=\frac{a+b+c}{2}$ 
# 
# Look up the available operations at: https://www.tensorflow.org/api_docs/python/tf. 
# 
# You'll need the `tf.sqrt()` operation. Remember `tf.add()`, `tf.subtract()` and `tf.multiply()` are overloaded with the +,- and * operators respectively.
# 
# You should get: 6.278497

# In[24]:


def compute_area(sides):
  #TODO: Write TensorFlow code to compute area of a triangle
  #  given its side lengths
  #a, b, c = sides  # TypeError: Tensor objects are not iterable when eager execution is not enabled. 
                    # To iterate over this tensor use tf.map_fn
  a = sides[0]
  b = sides[1]
  c = sides[2]
  s = tf.reduce_sum(sides, keepdims = False) * 1/2
  ret = tf.sqrt(s * (s - a) * (s - b) * (s - c))
  return(ret)
  
  return area

with tf.Session() as sess:
  area = compute_area(tf.constant([5.0, 3.0, 7.1]))
  result = sess.run(area)
  print(result)


# Extend your code to be able to compute the area for several triangles at once.
# 
# You should get: [6.278497 4.709139]

# In[52]:


def compute_area(sides):
  #TODO: Write TensorFlow code to compute area of a 
  #  SET of triangles given by their side lengths
  a = sides[:, 0]
  b = sides[:, 1]
  c = sides[:, 2]
  s = tf.reduce_sum(sides, axis = 1, keepdims = False) * 1/2
  #s = (a + b + c) * 1/2
  list_of_areas = tf.sqrt(s * (s - a) * (s - b) * (s - c))
  return list_of_areas

with tf.Session() as sess:
  # pass in two triangles
  area = compute_area(tf.constant([
      [5.0, 3.0, 7.1],
      [2.3, 4.1, 4.8]
    ]))
  result = sess.run(area)
  print(result)


# <h2> Placeholder and feed_dict </h2>
# 
# More common is to define the input to a program as a placeholder and then to feed in the inputs. The difference between the code below and the code above is whether the "area" graph is coded up with the input values or whether the "area" graph is coded up with a placeholder through which inputs will be passed in at run-time.

# In[55]:


with tf.Session() as sess:
  #TODO: Rather than feeding the side values as a constant, 
  #  use a placeholder and fill it using feed_dict instead.
  sides = tf.placeholder(dtype=tf.float32, shape=(None, 3))
  area = compute_area(sides)
  data_sides = [
      [5.0, 3.0, 7.1],
      [2.3, 4.1, 4.8]
    ]
  result = sess.run(area, feed_dict = {
    sides: data_sides
  })
  print(result)


# ## tf.eager
# 
# tf.eager allows you to avoid the build-then-run stages. However, most production code will follow the lazy evaluation paradigm because the lazy evaluation paradigm is what allows for multi-device support and distribution. 
# <p>
# One thing you could do is to develop using tf.eager and then comment out the eager execution and add in the session management code.
# 
# <b> You will need to click on Reset Session to try this out </b>

# In[1]:


import tensorflow as tf

## restart jupyiter kernel before this will work:
tf.enable_eager_execution()

#TODO: Using your non-placeholder solution, 
#  try it now using tf.eager by removing the session

def compute_area_debug(sides):
  #TODO: Write TensorFlow code to compute area of a 
  #  SET of triangles given by their side lengths
  a = sides[:, 0]
  b = sides[:, 1]
  c = sides[:, 2]
  s = tf.reduce_sum(sides, axis = 1, keepdims = False) * 1/2
  #s = (a + b + c) * 1/2
  list_of_areas = tf.sqrt(s * (s - a) * (s - b) * (s - c))
  return list_of_areas

compute_area_debug(tf.constant([
      [5.0, 3.0, 7.1],
      [2.3, 4.1, 4.8]
    ]))


# ## Challenge Exercise
# 
# Use TensorFlow to find the roots of a fourth-degree polynomial using [Halley's Method](https://en.wikipedia.org/wiki/Halley%27s_method).  The five coefficients (i.e. $a_0$ to $a_4$) of 
# <p>
# $f(x) = a_0 + a_1 x + a_2 x^2 + a_3 x^3 + a_4 x^4$
# <p>
# will be fed into the program, as will the initial guess $x_0$. Your program will start from that initial guess and then iterate one step using the formula:
# <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/142614c0378a1d61cb623c1352bf85b6b7bc4397" />
# <p>
# If you got the above easily, try iterating indefinitely until the change between $x_n$ and $x_{n+1}$ is less than some specified tolerance. Hint: Use [tf.where_loop](https://www.tensorflow.org/api_docs/python/tf/while_loop)

# In[19]:


import tensorflow as tf

## not using eager execution: doesn't work with tf.gradients)

a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
x = tf.placeholder(dtype=tf.float32, shape=[1])

def f(a, x):
  ret = a[0] + a[1] * x + a[2] * tf.pow(x, 2) + a[3] * tf.pow(x, 3) + a[4] * tf.pow(x, 4)
  return(ret)

def halley_step(a, x):
  f1 = tf.gradients(f(a, x), x)
  numerator = tf.multiply(2.0, f(a, x))
  numerator = tf.multiply(numerator, f1)
  denom_left = tf.multiply(2.0, tf.pow(f1, 2))
  f2 = tf.gradients(f1, x)
  denom_right = f(a, x) * f2
  denom = denom_left - denom_right
  ret = x - numerator / denom
  return(ret)

with tf.Session() as sess:
  ret = sess.run(halley_step(a, x), feed_dict = {x : [1.0]})
  print(ret)

## still to do: loop...


# Copyright 2017 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License