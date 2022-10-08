<h1>Notes</h1>

<ul>
    <li style="text-align: justify">All the files with the prefix <b>"single-"</b> refer to the sample graph in /datasets/gr_politicians/single/, while those with the prefix <b>"bacth-"</b> were used to implement the final experiment concerning the Greek politicians dataset. The same goes for the results as well.</li>
    <li style="text-align: justify">Each file is named using the correct prefix followed by the algorithm that is implemented.</li>
</ul>

<br/>

<h1>Instructions</h1>

<p style="text-align: justify">The files with the prefix "sigle-" can also be run using any of the graphs contained in /datasets/gr_politicians/batch/edge_lists/ directory. To do that you can comment all the lines inside the "IMPORT GRAPH FROM EDGE LIST FILE" section and uncomment the lines inside "IMPORT GRAPH FROM CSV FILE" section.</p>

<p style="text-align: justify">When a script is executed some new files will be created and they will be saved in the following directories:</p>
<ul>
<li style="text-align: justify"><b>results/</b> where you can find csv files containing the evaluation metrics for each algorithm</li>
<li style="text-align: justify"><b>clusters/</b> (in case of "single-" scripts) which contains text files that describe how the nodes of each graph are seperated</li>
<li style="text-align: justify"><b>embeddings/</b> (in case of "single-", node embedding scripts) which contains text files that include the vector representation of each node resulting from each node embedding algorithm</li>
<li style="text-align: justify"><b>metrics_plots/</b> (in case of "single-" scripts) which contains charts of each algorithm's evaluation metrics </li>
<li style="text-align: justify"><b>figures/</b> (in case of "batch-louvain-plot.py") which contains plots of the graph </li>
</ul>

<p>Now some instructions concerning each file individualy:</p>

<h3><b>single-node2vec.py</b></h3>
<p style="text-align: justify">When you run this script you will be prompted to choose between three diffferent test cases. These test cases were created in order to study how each node2vec parameter affects the performance of the algorithm. In case you want to manually input your own parameters, you can comment lines 48 and 57-67 and insert the desired values on lines 50-54.</p>

<h3><b>single-louvain.py</b></h3>
<p style="text-align: justify">You can change the resolution by editing line 43. The preset values are between 0.5 and 1.5 with step 0.1.</p>

<h3><b>batch-louvain-plot.py</b></h3>
<p style="text-align: justify">Running this script will generate plots of every snapshot of the Greek politicians dataset. The nodes are colored based on the Louvain method's results for every graph, while the positions are derived either from the spring_layout function of the networkx module, or from the latent vector representation of the node2vec algorithm (after applying a dimensions reduction algorithm). You will be prompted to choose one of the two options above.</p>

<h3><b>binary_classification.py</b></h3>
<p style="text-align: justify">For this script we take all common nodes of 11 consecutive Greek politician dataset snapshots, pair them up and check whether each pair is put in the same community by the Louvain method. We save the binary values for the first 10 graphs in an input matrix X and for the last one in the output vector y. Then, after splitting our new dataset in two parts (2/3 for training and 1/3 for testing), we train a simple Single Vector Machine Classifier. Finally, we evaluate how well our neural network can predict the joint classification of two nodes in the 11th snapshot if it knows how they were classified in the first 10. You can set how many and which snapshots will be used by modifying the constant values in lines 19-21.</p>