<h1>Description</h1>
This is a repository containing code that I developed for my dimploma thesis entitled <b>"Information Extraction from social media"</b>. The main goal of this project is to study node embedding and community detection algorithms and how effective they are in processing graphs. The algorithms we are going to focus on are: DeepWalk, node2vec and the Louvain method. The graphs we will examine are based on monthly data of the activity of the Greek politicians' accounts on Twitter.


<br/>


<h1>Layout</h1>

<h3><b>dataset/</b></h3>
<p>This directory contains the files that are used for loading the graphs of the Greek politicians dataset. There is also code for cleaning up the dataset, which involves picking the correct nodes and generating the ground truth.</p>

<h3><b>zachary_karate_club/</b></h3>
<p>This is code for testing various node embedding algorithms on Zachary's karate club network.</p>

<h3><b>gr_politicians/</b></h3>
<p>Here is all the code that I wrrote and the results that I got for the Greek politicians dataset.</p>


<br/>


<h1>Installation</h1>
<p>This project was created using Python 3.8. </p>
<p>Install all the necessary python modules by runing the following command:</p>

```
$ pip install -r requirements.txt
```