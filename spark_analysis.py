
# coding: utf-8

# ** This file is adapted from a Jupyter notebook created as an assignment by Professors Zachary Ives and Clayton Greenberg for CIS 545 Big Data Analytics **

# 
# ### Step 2.1 Initializing a Connection to Spark
# 
# We'll open a connection to Spark as follows. Note that Spark has multiple interfaces, as you will see if you look at sample code elsewhere. `SparkSession` is the “most modern” one and we’ll be using it for this course.  From `SparkSession`, you can load data into Spark DataFrames as well as `RDD`s.

# In[1]:


# If you want to run from an environment outside of the Docker container you'll need to uncomment 
# and run this.  Otherwise you can skip through.
# ! pip install pyspark --user
# ! pip install seaborn --user
# ! pip install plotly --user
# ! pip install imageio --user
# ! pip install folium --user
# ! pip install heapq

import numpy as np
import pandas as pd

#misc
import gc
import time
import warnings


#viz
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib.gridspec as gridspec 
import matplotlib.gridspec as gridspec 

# graph viz
import plotly.offline as pyo
from plotly.graph_objs import *
import plotly.graph_objs as go

#map section
import imageio
import folium
import folium.plugins as plugins
# from mpl_toolkits.basemap import Basemap


#graph section
import networkx as nx
import heapq  # for getting top n number of things from list,dict


# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F

import os
# make sure pyspark tells workers to use python3 not 2 if both are installed
# os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/bin/ipython3'

spark = SparkSession.builder.appName('Graphs-HW2').getOrCreate()


# ### Step 2.2 Download data
# 
# The following code retrieves the Yelp dataset in a zipfile and decompresses it.  It will take quite a while - you may want to take a break while it runs.

# In[3]:


# Based on https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url

import zipfile
import urllib.request
import shutil
import os

def get_and_unzip(url,file_name):
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        with zipfile.ZipFile(file_name) as zf:
            zf.extractall()

if not os.path.exists("data"):
    os.mkdir("data")
    os.chdir("data")
    get_and_unzip("http://www.cis.upenn.edu/~cis545/yelp-dataset.zip","yelp-dataset.zip")
    os.chdir("..")


# ### Step 2.3 Load Our Graph Datasets.
# 
# For this assignment, we’ll be looking at graph data (reviews, reviewers, businesses) downloaded from Yelp.
# 
# **A very brief review of graph theory**. Recall that a graph $G$ is composed of a set of vertices $V$ (also called nodes) and edges $E$ (sometimes called links).  Each vertex $v \in V$ has an identity (often represented in the real world as a string or numeric “node ID”).  Each edge $e \in E$ is a tuple $(v_i,...,v_j)$ where $v_i$ represents the source or origin of the edge, and $v_j$ represents the target or destination.  In the simplest case, the edge tuple above is simply the pair $(v_i,v_j)$ but in many cases we may have additional fields such as a label or a distance.  Recall also that graphs may be undirected or directed; in undirected graphs, all edges are symmetric whereas in directed graphs, they are not.  For instance, airline flights are directed, whereas Facebook friend relationships are undirected.
# 
# Let’s read our social graph data from Yelp, which forms a directed graph.  Here, the set of nodes is also not specified; the assumption is that the only nodes that matter are linked to other nodes, and thus their IDs will appear in the set of edges.  To load the file `input.txt` into a Spark DataFrame, you can use lines like the following.
# 
# ```
# # Read lines from the text file
# input_sdf = spark.read.load('input.txt', format="text")
# ```
# 
# We’ll use the suffix `_sdf` to represent “Spark DataFrame,” much as we used `_df` to denote a Pandas DataFrame in Homework 1.  Load the various files from Yelp.
# 
# Your datasets should be named `yelp_business_sdf`, `yelp_business_attributes_sdf`, `yelp_business_horus_sdf`, `yelp_check_in_sdf`, `yelp_reviews_sdf`, and `yelp_users_sdf`.

# In[5]:


# TODO: load Yelp datasets

# Worth 1 point per successful load, 5 additional points if valid schema

# YOUR CODE HERE
yelp_business_sdf = spark.read.load('/home/jovyan/work/hw2/data/yelp_business.csv', format="csv", header='true', inferSchema='true')
yelp_business_attributes_sdf = spark.read.load('/home/jovyan/work/hw2/data/yelp_business_attributes.csv', format="csv", header='true', inferSchema='true')
yelp_business_hours_sdf = spark.read.load('/home/jovyan/work/hw2/data/yelp_business_hours.csv', format="csv", header='true', inferSchema='true')
yelp_check_in_sdf = spark.read.load('/home/jovyan/work/hw2/data/yelp_checkin.csv', format="csv", header='true', inferSchema='true')
yelp_reviews_sdf = spark.read.load('/home/jovyan/work/hw2/data/yelp_review2.csv', format="csv", header='true', inferSchema='true')
yelp_users_sdf = spark.read.load('/home/jovyan/work/hw2/data/yelp_user.csv', format="csv", header='true', inferSchema='true')


# In[6]:


yelp_business_sdf.show(10)


# In[7]:


yelp_reviews_sdf.show(10)


# In[8]:


yelp_business_hours_sdf.show(10)


# In[9]:


yelp_business_attributes_sdf.show(10)


# In[10]:


yelp_check_in_sdf.show(10)


# In[11]:


yelp_users_sdf.show(10)


# In[12]:


yelp_reviews_sdf.dtypes


# In[13]:


if yelp_reviews_sdf.dtypes[0][1] != 'string':
    raise ValueError('Unexpected datatype on ' + yelp_reviews.dtypes[0][0])
    


# ### Step 2.4 Simple Wrangling in Spark DataFrames
# 
# Currently, some of the data from the Yelp dataset is a bit ugly.
# 
# You should:
# 
# * Clean `yelp_business_hours` so `"None"` is replaced by a Spark `null`.
# * Clean `yelp_users` so `"None"` is replaced by a Spark `null`.
# 
# 
# 
# First, create SQL tables for each of your Spark DataFrames.  Take the same names as you've used previously, except remove the `_sdf` suffix.

# In[14]:


# TODO: save tables with names such as yelp_business, yelp_users

# Worth 5 points if done successfully

# YOUR CODE HERE
yelp_business_sdf.createOrReplaceTempView("yelp_business")
yelp_business_attributes_sdf.createOrReplaceTempView("yelp_business_attributes")
yelp_business_hours_sdf.createOrReplaceTempView("yelp_business_hours")
yelp_check_in_sdf.createOrReplaceTempView("yelp_check_in")
yelp_reviews_sdf.createOrReplaceTempView("yelp_reviews")
yelp_users_sdf.createOrReplaceTempView("yelp_users")


# In[15]:


if spark.sql('SELECT COUNT(*) AS count FROM yelp_business').take(1)[0]['count'] != 174567:
    raise ValueError("Unexpected count or table not found")
    


# Now you'll need to define a **user-defined function** `replace_none_with_null` to apply to your string fields.  It should take a string parameter and compare it with `None` and `Na`.  If there is a match to either, it should return the Python None value (which will become a Spark null), otherwise it shoudl return the value.

# In[16]:


# Define your function here.
# Worth 7 points if done successfully

def replace_none_with_null(x):
    if x == "None" or x == "Na":
        return None
    else:
        return x


# In[17]:


if replace_none_with_null('None'):
    raise ValueError('Your function does not work')
    
if not replace_none_with_null('x'):
    raise ValueError('Your function does not work')


# In[18]:


# Wrap the Python code in a Spark UDF

# We're providing this since it's basically a template

from pyspark.sql.functions import udf
import pyspark.sql.types

spark.udf.register("replace_none_with_null", replace_none_with_null)

spark_replace_none_with_null = udf(replace_none_with_null, pyspark.sql.types.StringType())


# Now use `replace_with_none` in SQL, or `spark_replace_none_with_null` if you prefer Pandas-style Spark statements, to clean the data as described above.

# In[19]:


yelp_business_hours_sdf.show()


# In[20]:


# TODO: Clean yelp_business_hours_sdf

# Worth 5 points for each of the two tables specified in 2.4 instructions, if done successfully

# YOUR CODE HERE
yelp_business_hours_sdf = spark.sql('SELECT replace_none_with_null(business_id) AS business_id, replace_none_with_null(monday) AS monday, replace_none_with_null(tuesday) AS tuesday, replace_none_with_null(wednesday) AS wednesday, replace_none_with_null(thursday) AS thursday, replace_none_with_null(friday) AS friday, replace_none_with_null(saturday) AS saturday, replace_none_with_null(sunday) AS sunday FROM yelp_business_hours')
yelp_business_hours_sdf.createOrReplaceTempView("yelp_business_hours")


# In[21]:


if spark.sql('select count(*) as count from yelp_business_hours where wednesday=\'None\'').take(1)[0]['count'] > 0:
    raise ValueError('Did not successfully clean business hours')
    


# In[22]:


# TODO: Clean yelp_users, which has schema
# |user_id|   name|review_count|yelping_since|             friends|useful|funny|cool|fans|elite|average_stars|compliment_hot
# |compliment_more|compliment_profile|compliment_cute|compliment_list|compliment_note|compliment_plain|compliment_cool
# |compliment_funny|compliment_writer|compliment_photos|

# YOUR CODE HERE
yelp_users_sdf = spark.sql('SELECT replace_none_with_null(user_id) AS user_id, replace_none_with_null(name) AS name, replace_none_with_null(review_count) AS review_count, replace_none_with_null(yelping_since) AS yelping_since, replace_none_with_null(friends) AS friends, replace_none_with_null(useful) AS useful, replace_none_with_null(funny) AS funny, replace_none_with_null(cool) AS cool, replace_none_with_null(fans) AS fans, replace_none_with_null(elite) AS elite, replace_none_with_null(average_stars) AS average_stars, replace_none_with_null(compliment_hot) AS compliment_hot, replace_none_with_null(compliment_more) AS compliment_more, replace_none_with_null(compliment_profile) AS compliment_profile, replace_none_with_null(compliment_cute) AS compliment_cute, replace_none_with_null(compliment_list) AS compliment_list, replace_none_with_null(compliment_note) AS compliment_note, replace_none_with_null(compliment_plain) AS compliment_plain, replace_none_with_null(compliment_cool) AS compliment_cool, replace_none_with_null(compliment_funny) AS compliment_funny, replace_none_with_null(compliment_writer) AS compliment_writer, replace_none_with_null(compliment_photos) AS compliment_photos FROM yelp_users')
yelp_users_sdf.createOrReplaceTempView("yelp_users")


# In[23]:


if spark.sql('select count(*) as count from yelp_users where elite=\'None\'').take(1)[0]['count'] > 0:
    raise ValueError('Did not successfully clean users')
    


# ### Step 2.5 Simple Analytics on the Data
# 
# In this section, we shall be executing Spark operations on the data given. Beyond simply executing the queries, you may try using `.explain()` method to see more about the query execution. Also, please read the data description prior to attempting the following questions to understand the data.
# 
# #### 2.5.1 Businesses with the best average review
# 
# Compute, stored in `best_average_sdf`, the list of names of businesses based on their average review score (review stars), in decreasing order, sorted lexicographically (in increasing order) by name if they have the same score.  Output the number of reviews also.  Call the columns `name`, `avg_rating`, and `count`.

# In[24]:


# TODO: Businesses with best average review

# Worth 5 points if done successfully
# YOUR CODE HERE
import pyspark.sql.functions as func
from pyspark.sql.functions import desc
#best_average_sdf = yelp_reviews_sdf.groupBy("business_id").agg({'stars': 'mean'})
best_average_sdf = yelp_reviews_sdf.groupBy("business_id").agg(func.mean('stars').alias('avg_rating'))
best_average_sdf = best_average_sdf.join(yelp_business_sdf, best_average_sdf.business_id == yelp_business_sdf.business_id)
best_average_sdf = best_average_sdf.drop('neighborhood', 'address', 'city', 'state', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'is_open', 'categories', 'business_id')
best_average_sdf = best_average_sdf.selectExpr("avg_rating as avg_rating", "name as name", "review_count as count")
best_average_sdf = best_average_sdf.orderBy(desc('avg_rating'), 'name')
best_average_sdf.show()


# In[25]:


row = best_average_sdf.take(10)


# #### 2.5.2 Users who are more negative than average
# 
# Find the users whose average review is below the *average of the per-user* average reviews.  Think about how to factor that into steps!
# 
# * Compute the (floating-point) variable `overall_avg` as the average of the users' average reviews. (You might compute this in a Spark DataFrame first).
# * Then output `negative_users_sdf` as the users whose average rating is below that.  This Spark dataframe should have `name` and `avg_rating` and should be sorted first (from lowest to highest) by average rating, then lexicographically (in ascending order) by name.  You should drop cases where the name is null.
# 

# In[26]:


# TODO: compute overall_avg as a VALUE, and negative_users_sdf as a Spark dataframe
# Worth 5 points each

# YOUR CODE HERE
from pyspark.sql.functions import mean
overall_avg = yelp_users_sdf.select(mean('average_stars').alias('mean_stars'))
overall_avg.createOrReplaceTempView("overall_avg")
negative_users_sdf = spark.sql('SELECT y.name, y.average_stars FROM yelp_users AS y, overall_avg as o WHERE y.average_stars < o.mean_stars')
negative_users_sdf = negative_users_sdf.orderBy(['average_stars', 'name'], ascending=[1, 1])
negative_users_sdf = negative_users_sdf.filter(negative_users_sdf.name.isNotNull())
negative_users_sdf = negative_users_sdf.selectExpr('name as name', 'average_stars as avg_rating')
negative_users_sdf.show(10)


# In[27]:


if not overall_avg:
    raise ValueError('Forgot to compute the overall average')
    


# In[28]:


negative_users_sdf.show(10)


# #### 2.5.4 Cities by number of businesses
# 
# Find the top 10 cities by number of (Yelp-listed) businesses.
# 
# This time, use the `take()` function to create a *list* of the top 10 cities (as Rows).  Call this list `top10_cities` and make sure it includes city `name` and `num_restaurants`.

# In[29]:


# TODO: cities by number of businesses
# Worth 5 points

# YOUR CODE HERE
import pyspark.sql.functions as func
cities = yelp_business_sdf.groupBy(yelp_business_sdf['city']).count()
cities = cities.orderBy(func.desc('count'))
cities = cities.selectExpr('city as name', 'count as num_restaurants')
top10_cities = cities.take(10)
top10_cities


# In[30]:


if len(top10_cities) != 10:
    raise ValueError('Not top10!')


# # Step 3. Computing Simple Graph Centrality
# 
# The study of networks has proposed a wide variety of measures for measuring importance of nodes.  A popular metric that is easy to compute is the degree centrality.  The degree centrality of a node is simply the number of connections to the node.  In a directed graph such as ours, you will want to compute both the indegree centrality (number of nodes with edges coming to this node) and outdegree centrality (number of nodes with edges coming from this node).
# 
# ## 3.1 Generate user-business graph
# 
# For this step, we want to construct a *directed* graph with edges from users to business for every interaction (review, in our case). To do this, we will use the `yelp_reviews_sdf` dataframe and extract the `user_id` as `from_node` and `business_id` as `to_node` into a different dataframe called `review_graph_sdf`.  Finally, include the `stars` field but call it `score`.  Also make sure it is available as a table in SQL called `review_graph`.
# 
# Some of the values may be null; remove these for `user_id` or `business_id`.

# In[31]:


# Create review graph SDF from yelp_reviews
# Worth 5 points

# YOUR CODE HERE
review_graph_sdf = spark.sql('SELECT r.user_id, r.business_id, r.stars FROM yelp_reviews as r')
review_graph_sdf = review_graph_sdf.selectExpr('user_id as from_node', 'business_id as to_node', 'stars as score')
review_graph_sdf = review_graph_sdf.filter(review_graph_sdf.from_node.isNotNull() & review_graph_sdf.to_node.isNotNull())

review_graph_sdf.createOrReplaceTempView('review_graph')


# In[32]:


# Display the count of number of edges
review_graph_sdf.count()

### BEGIN HIDDEN TEST
if review_graph_sdf.count() != 5273700:
    raise ValueError('Unexpected graph size')
### END HIDDEN TEST


# ## 3.2 Businesses with highest indegree
# 
# Find, in decreasing order, the businesses with the most (highest number of) reviews, using either `review_graph_sdf` (or its SQL version) or `yelp_reviews` as well as `yelp_business`.  The dataframe should have the fields `name`,`count`, and `rating` (average `score`).  Assign these to a new dataframe `most_reviewed_sdf`.

# In[33]:


# TODO: most_reviewed_sdf
# Worth 5 points

# YOUR CODE HERE
import pyspark.sql.functions as func
most_reviewed_sdf = review_graph_sdf.groupBy('to_node').agg(func.mean('score').alias('rating'), func.count('to_node').alias('count'))
most_reviewed_sdf = most_reviewed_sdf.orderBy(func.desc('count'))
most_reviewed_sdf = most_reviewed_sdf.join(yelp_business_sdf, most_reviewed_sdf.to_node == yelp_business_sdf.business_id).drop('neighborhood', 'address', 'city', 'state', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'is_open', 'categories', 'business_id', 'to_node', 'review_count')
#yelp_reviews_sdf.groupBy("business_id").agg(func.mean('stars').alias('avg_rating'))
#best_average_sdf = best_average_sdf.join(yelp_business_sdf, best_average_sdf.business_id == yelp_business_sdf.business_id)
#best_average_sdf = best_average_sdf.drop('neighborhood', 'address', 'city', 'state', 'state', 'postal_code', 'latitude', 'longitude', 'stars', 'is_open', 'categories', 'business_id')


# In[34]:


most_reviewed_sdf.show(10)


# ## 3.2 Outdegree
# 
# Next get a list of users whose vertices have the highest outdegree, i.e., they write the most reviews.  Return this in a dataframe `prolific_reviewers_sdf` with fields `name` and `num_reviews`.

# In[35]:


# TODO: reviews who created most reviews (beware duplicate names)
# Worth 5 points

# YOUR CODE HERE
import pyspark.sql.functions as func
from pyspark.sql.functions import col
#prolific_reviewers_sdf = review_graph_sdf.groupBy('from_node').agg(func.count('from_node').alias('count'))
#cities = yelp_business_sdf.groupBy(yelp_business_sdf['city']).count()
prolific_reviewers_sdf = review_graph_sdf.groupBy(review_graph_sdf['from_node']).count()
prolific_reviewers_sdf = prolific_reviewers_sdf.orderBy(func.desc('count'))
prolific_reviewers_sdf = prolific_reviewers_sdf.join(yelp_users_sdf, prolific_reviewers_sdf.from_node == yelp_users_sdf.user_id)
prolific_reviewers_sdf.cache()
prolific_reviewers_sdf = prolific_reviewers_sdf.select('name', 'count').orderBy(func.desc('count'))
prolific_reviewers_sdf = prolific_reviewers_sdf.selectExpr('name as name', 'count as num_reviews')
prolific_reviewers_sdf.show(10)



# In[36]:


top10_reviewers = prolific_reviewers_sdf.take(10)
top10_reviewers


# Note that the indegree also gives you the most reviewed restaurants and the outdegree gives you the information about the users who write the most reviews.
# 
# For the advanced part of this Homework, we'll consider more complex measures than indegree / outdegree.  For now let's move on to the more general problem of graph traversal.

# # Step 4. “Traversing” a Graph
# 
# For our next tasks, we will be “walking” the graph and making connections.
# 
# 
# ## 4.1 Distributed Breadth-First Search
# A search algorithm typically starts at a node or set of nodes, and “explores” or “walks” for some number of steps to find a match or a set of matches.
# 
# Let’s implement a distributed version of a popular algorithm, breadth-first-search (BFS).  This algorithm is given a graph `G`, a set of origin nodes `N`, and a depth `d`.  In each iteration or round up to depth `d`, it explores the set of all new nodes directly connected to the nodes it already has seen, before going on to the nodes another “hop” away.  If we do this correctly, we will explore the graph in a way that (1) avoids getting caught in cycles or loops, and (2) visits each node in the fewest number of “hops” from the origin.  BFS is commonly used in tasks such as friend recommendation in social networks.
# 
# **How does distributed BFS in Spark work**?  Let’s start with a brief sketch of standard BFS.  During exploration “rounds”, we can divide the graph into three categories:
# 
# 1. *unexplored nodes*.  These are nodes we have not yet visited.  You don’t necessarily need to track these separately from the graph.
# 2. *visited nodes*.  We have already reached these nodes in a previous “round”.
# 3. *frontier nodes*.  These are nodes we have visited in this round.  We have not yet checked whether they have out-edges connecting to unexplored nodes.
# 
# We can illustrate these with a figure and an example.
# 
# ![Graph traversal](https://drive.google.com/uc?export=view&id=1I2Kc3uQcDlp7RsDqRQAfQAvS3F_VcJpA)

# Let’s look at the figure, which shows a digraph.  The green node A represents the origin.
# 
# * In the first round, the origin A is the sole frontier node.  We find all nodes reachable directly from A, namely B-F; then we remove all nodes we have already visited (there are none) or that are in the frontier (the node A itself).  This leaves the blue nodes B-F, which are all reachable in (at most) 1 hop from A.
# * In the second round, we move A to the visited set and B-F to the frontier.  Now we explore all nodes connected directly to frontier nodes, namely A (from B), F (from E), and the red nodes G-L.  We eliminate the nodes already contained in the frontier and visited sets from the next round’s frontier set, leaving the red nodes only.
# * In the third round, we will move B-F to the visited set, G-L to the frontier set, and explore the next round of neighbors N-V.  This process continues up to some maximum depth (or until there are no more unexplored nodes).
# 
# Assume we create data structures (we can make them DataFrames) for the visited and frontier nodes.  Consider (1) how to initialize the different sets at the start of computation (note: unexplored nodes are already in the graph), and (2) how to use the graph edges and the existing data structures to update state for the next iteration “round”.
# 
# You might possibly have seen how to create a breadth-first-search algorithm in a single-CPU programming language, using a queue to capture the frontier nodes. With Spark we don’t need a queue -- we just need the three sets above.
# 
# ### 4.1.1 Breadth-First Search Algorithm
# 
# Create a function `spark_bfs(G, origins, max_depth)` that takes a Spark DataFrame with a graph G (following the schema for `review_graph_sdf` described above, but to be treated as an **undirected graph**), a Python list-of-dictionaries `origins` of the form 
# 
# ```
# [{‘node’: nid1}, 
#  {‘node’: nid2}, 
#  …]
# ```
# 
# and a nonnegative integer “exploration depth” `max_depth` (to only run BFS on a tractable portion of the graph).  The `max_depth` will be the maximum number of edge traversals (e.g., the origin is at `max_depth=0`, one hop from the origin is `max_depth=1`, etc.  The function should return a DataFrame containing pairs of the form (node, distance), where the distance is depth at which $n$ was *first* encountered (i.e., the shortest-path distance from the origin nodes).  Note that the origin nodes should also be returned in this Spark DataFrame (with depth 0)!  
# 
# You can create a new Spark DataFrame with an integer `node` column from the above list of maps `origins`, as follows. This will give you a DataFrame of the nodes to start the BFS at
# 
# ```
# schema = StructType([
#             StructField("node", StringType(), True)
#         ])
# 
#     my_sdf = spark.createDataFrame(my_list_of_maps, schema)
# ```
# 
# In this algorithm, be careful in each iteration to keep only the nodes with their shortest distances (you may need to do aggregation or item removal).  You should accumulate all nodes at distances 0, 1, ..., `max_depth`.

# In[37]:


# TODO: iterative search over undirected graph
# Worth 5 points directly, but will be needed later

def spark_bfs(G, origins, max_depth):
    schema = StructType([
                StructField("node", StringType(), True),
                StructField("distance", IntegerType(), False)
            ])
    
    # YOUR CODE HERE
    for i in range(len(origins)):
        origins[i]['distance'] = 0
    
    frontier = spark.createDataFrame(origins, schema)
    visited = spark.createDataFrame(origins, schema)
    
    for j in range(0, max_depth):
        if j%2 == 0:
            frontier = frontier.join(G, frontier.node == G.from_node).select(G['to_node']).selectExpr('to_node as node')
            frontier = frontier.withColumn('distance', F.lit(j + 1))
            visited = visited.union(frontier)
            visited = visited.dropDuplicates(['node'])
        else:
            frontier = frontier.join(G, frontier.node == G.to_node).select(G['from_node']).selectExpr('from_node as node')
            frontier = frontier.withColumn('distance', F.lit(j + 1))
            visited = visited.union(frontier)
            visited = visited.dropDuplicates(['node'])
            
    return visited
            


# In[38]:


orig = [{'node': 'bv2nCi5Qv5vroFiqKGopiw'}]

count= spark_bfs(review_graph_sdf, orig, 3).count()
print(count)


# ## Step 4.2. Restaurant Recommendation
# 
# Now create a function `friend_rec` that takes in two arguments: the graph_sdf and the ID of a user, `me`.  Using `spark_bfs` it should recommend restaurants that are highly popular among the reviewers who reviewed the same restaurants as `me`.
# 
# Then, take the review_graph, filter it to only consider 3-star reviews or above, and run `friend_rec` over the results and the user with ID `bv2nCi5Qv5vroFiqKGopiw`.  In the Spark dataframe `rec_rest_sdf`, give us the `name`s of the most-highly recommended restaurants, sorted primarily in descending order of count, and then secondarily by lexicographic order of name.

# In[39]:


# TODO: restaurant recommendation using spark_bfs
# Worth 5 points

# YOUR CODE HERE
import pyspark.sql.functions as func
def friend_rec(graph_sdf, me):
    origins = [{'node': me}]
    restaurants = spark_bfs(graph_sdf, origins, 2)
    #extra = spark_bfs(graph_sdf, origins, 2)
    #restaurants = restaurants.subtract(extra)
    return restaurants

top_reviews = review_graph_sdf.filter(review_graph_sdf.score >= 3)

rec_rest_sdf = friend_rec(top_reviews, 'bv2nCi5Qv5vroFiqKGopiw')

rec_rest_sdf = rec_rest_sdf.join(review_graph_sdf, rec_rest_sdf.node == review_graph_sdf.from_node).select(review_graph_sdf['to_node']).selectExpr('to_node as node')
rec_rest_sdf = rec_rest_sdf.groupBy('node').count()
rec_rest_sdf = rec_rest_sdf.join(yelp_business_sdf, rec_rest_sdf.node == yelp_business_sdf.business_id).select('name', 'count')
rec_rest_sdf = rec_rest_sdf.orderBy(func.desc('count'), 'name')
rec_rest_sdf


# In[40]:


rec_rest_sdf.show()


# In[41]:


rec_rest = rec_rest_sdf.take(10)
rec_rest


# ## Step 4.3. Friend Visualization
# 
# #### 4.3.1: Loading data subsets
# A closer look at the `yelp_user` dataframe tells us that there is an attribute called `friends` that we can use in order to construct an undirected friend graph.  For this part of the assignment we'll go back to Pandas -- not Spark -- DataFrames.
# 
# We will work with the first 200 entries from the `yelp_user` data file and visualize these users' friends.
# 
# Read the first 200 entries of the `yelp_user.csv` file into a pandas dataframe called `user_200` (Remember: You can pass `nrows` as an option to the `pd.read_csv()`)
# 
# We’ll subsequently make use of the `networkx` graph visualization tool, which lets us see what the graph actually looks like.
# 

# In[43]:


# TODO: read 200 entries
# Worth 5 points

# YOUR CODE HERE
import csv
import pandas as pd

user_200 = pd.read_csv('/home/jovyan/work/hw2/data/yelp_user.csv', usecols=['user_id', 'name', 'friends'], nrows=200)


# In[44]:


user_200


# #### Step 4.3.2: Select users with at least one friend
# 
# In this part select the friends from `user_200` where the value is not `None`. The `friends` column is a string with comma-separated `user_id`s as values. We will make use of `lambda` functions in order to extract the different `user_id` from this comma separated string.
# 
# Select **only the users who have at least one friend**. That is, the `friends` column does not have the value "None".

# In[45]:


# TODO: find users with friends
# Worth 2 points

# YOUR CODE HERE
user_200 = user_200[user_200['friends'] != "None"]


# In[46]:


user_200


# #### Step 4.3.3: Extracting friend as list
# 
# Pandas dataframe supports the use of `df.apply()` which can take a function as a parameter.
# 
# Example use of `lambda` function with `.apply()`
# 
# `df['col_2'] = df['col_1'].apply(lambda x: x+10)`
# 
# The above statement will create a column 'col_2' in df with values of df['col_1'] + 10
# 
# For the next step, make use of `lambda` functions to `split` the value of the `friends` column by `,` and apply this to create a new column called `list_friends`.

# In[47]:


# TODO: friend lists
# Worth 5 points

# YOUR CODE HERE
user_200['list_friends'] = user_200['friends'].apply(lambda x: x.split(',') )


# In[48]:


user_200[['name','list_friends']]


# #### Step 4.3.4: Obtaining Friend lists
# 
# We are only interest in the `user_id` and `list_friends` columns.  Select these into a dataframe called `subset_users`.
# 
# The `.stack()` option allows you to "unnest" the items in the list, associating them with the corresponding value of the index.  
# 
# In our instance, we would like to `set_index()` on `user_id` so the user ID is the index.  Then if we `.apply()` the `stack()` operation on the `list_friends` column:
# 
# ```
# result_df = df.set_index(['col_1'])['col_2'].apply(pd.Series).stack()
# ```
# 
# We should get each element in the list associated with the appropriate `user_id`.  Performing `df.reset_index()` on `result_df` will give us `friend_data` in an edge table.  Rename the columns of `friend_data` to 'source', 'level_1', and 'target'.

# In[49]:


# TODO: projection to users and lists of friends, stacked
# Worth 3 points
# YOUR CODE HERE
subset_users = user_200[['user_id', 'list_friends']]
friend_data = subset_users.set_index(['user_id'])['list_friends'].apply(lambda x: pd.Series(x)).stack()
friend_data = friend_data.reset_index()
friend_data = friend_data.rename(index=str, columns={"user_id": "source", "level_1": "level_1", 0: "target"})


# In[50]:


friend_data.head()


# #### Step 4.3.5: Visualization using Networkx
# 
# In this step we will use the `networkx` library to visualize.
# 
# 
# You can create the graph from the networkx function `from_pandas_edgelist` and the `friend_data`. 
# 

# In[51]:


# TODO: networkx graph ready to draw
# Worth 3 points

import networkx as nx
# YOUR CODE HERE
graph = nx.from_pandas_edgelist(friend_data, 'source', 'target', ['source', 'target'])


# In[52]:


get_ipython().run_line_magic('matplotlib', 'inline')
nx.draw(graph)

