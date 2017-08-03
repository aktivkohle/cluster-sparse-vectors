# Test a clustering algorithm on my youtube captions dataset

So as I filled the database up myself, choosing the API queries, there was never any need to cluster anything as I know what is in there. These algorithms are more useful when you suspect but don't know what the clusters are. Nevertheless I thought it wouldn't hurt to see how well the algorithm(s) perform. The data is ready to go, there was not much processing or cleaning needed. 

Especially impressive was that **sklearn.cluster.KMeans** managed to work without me having to do anything to the SciPy COO sparse matrix! I checked the sklearn source code and you can see for example in [this line](https://github.com/scikit-learn/scikit-learn/blob/ab93d65/sklearn/cluster/k_means_.py#L1078) that they've built in some code to handle that so it wasn't magic.

So I tried 5,8,10 and 20 clusters, really 10 was about as good as it got. When you have a look at the results in the table at the bottom you will see it has not performed perfectly, but it has nevertheless seperated out some of the main groups. 

The number of videos found in each cluster is as follows:

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>videoTitle</th>
    </tr>
    <tr>
      <th>cluster_labels</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>548</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1855</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1035</td>
    </tr>
    <tr>
      <th>3</th>
      <td>850</td>
    </tr>
    <tr>
      <th>4</th>
      <td>154</td>
    </tr>
    <tr>
      <th>5</th>
      <td>420</td>
    </tr>
    <tr>
      <th>6</th>
      <td>277</td>
    </tr>
    <tr>
      <th>7</th>
      <td>287</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1213</td>
    </tr>
    <tr>
      <th>9</th>
      <td>725</td>
    </tr>
  </tbody>
</table>

The following interesting bit of code from [stackoverflow](https://stackoverflow.com/questions/22472213/python-random-selection-per-group) selects ten randomly from each group so we can read the titles or watch them if desired:

```python
size = 10        # sample size
replace = False  # with replacement
fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace),:]
clustered_sample = videos_df.groupby('cluster_labels', as_index=False).apply(fn)
clustered_sample[['videoTitle','cluster_labels', 'link']]
```

The ten clusters a numbered 0 to 9. 
* Cluster 0 seems to be a 'data science' cluster but mainly general topics. 
* Cluster 1 at the moment seems to be a mess of different topics. 
* Cluster 2 is mostly about cooking, possibly the more infotainment kind of videos. 
* Cluster 3 is about machine learning and very detailed technical topics. 
* Cluster 4 sounds like cooking family recipes and cakes. 
* Cluster 5 seems to have a chicken and curry, international foods influence.
* Cluster 6 is mainly cooking
* Cluster 6 seems to be entirely about dog training
* Cluster 8 might be 'healthy alternative' cooking
* Cluster 9 sounds like a high level videos about machine learning

Yes, there are other clustering algorithms than KMeans eg DBSCAN. In [the notebook](Cluster_Sparse_Vectors.ipynb) there is a bit more discussion about numbers of clusters, as well as the code which produced this.

One thing I observe is that no matter whether the current algorithm is run with 5, 8, 10 or 20 clusters it never seems to find the small cluster of cat videos. There are not many compared to dog training, data science, cooking etc. I think if you gave KMeans a fixed starting point rather than let it randomly choose its starting point it might find such a cluster. Will think about how to do that, at any rate, given there are 429429 dimensions in each of these sparse vectors, it's not something you can tell just by plotting it which you can do when there are 3 dimensions.

Without any further ado, here are the samples of ten from each cluster found:



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>videoTitle</th>
      <th>cluster_labels</th>
      <th>link</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="10" valign="top">0</th>
      <th>2436</th>
      <td>CSV Data 1 - Intro to Data Science</td>
      <td>0</td>
      <td>https://www.youtube.com/watch?v=eNEI8kdujCo</td>
    </tr>
    <tr>
      <th>3562</th>
      <td>Les serveurs dédiés OVH.com au cœur de l’infra...</td>
      <td>0</td>
      <td>https://www.youtube.com/watch?v=jEv7jISZ9tg</td>
    </tr>
    <tr>
      <th>6780</th>
      <td>All About Data Scientist Jobs</td>
      <td>0</td>
      <td>https://www.youtube.com/watch?v=XdduOJNZCQc</td>
    </tr>
    <tr>
      <th>3798</th>
      <td>What Data Is Good For Linear Regression - Intr...</td>
      <td>0</td>
      <td>https://www.youtube.com/watch?v=KFVdS328iC8</td>
    </tr>
    <tr>
      <th>4555</th>
      <td>End-to-end Data Science with Civis</td>
      <td>0</td>
      <td>https://www.youtube.com/watch?v=nMalICUv1UM</td>
    </tr>
    <tr>
      <th>1024</th>
      <td>Making data mean more through storytelling | B...</td>
      <td>0</td>
      <td>https://www.youtube.com/watch?v=6xsvGYIxJok</td>
    </tr>
    <tr>
      <th>4752</th>
      <td>Data Science Essentials | Microsoft on edX | C...</td>
      <td>0</td>
      <td>https://www.youtube.com/watch?v=OgONrGwHTyU</td>
    </tr>
    <tr>
      <th>5875</th>
      <td>Cornering the Market: Predicting the Price of ...</td>
      <td>0</td>
      <td>https://www.youtube.com/watch?v=TeSzDoNq8Wo</td>
    </tr>
    <tr>
      <th>7155</th>
      <td>What is Data Science? [Data Science 101]</td>
      <td>0</td>
      <td>https://www.youtube.com/watch?v=z1kPKBdYks4</td>
    </tr>
    <tr>
      <th>5519</th>
      <td>Data Wrangling, Normalization &amp; Preprocessing:...</td>
      <td>0</td>
      <td>https://www.youtube.com/watch?v=RWThOMYFsy8</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">1</th>
      <th>993</th>
      <td>Phoebe loves doing tricks</td>
      <td>1</td>
      <td>https://www.youtube.com/watch?v=6qTml4zjyKc</td>
    </tr>
    <tr>
      <th>6363</th>
      <td>Canadian School Teacher Suspended After Giving...</td>
      <td>1</td>
      <td>https://www.youtube.com/watch?v=VkKXiCllIyk</td>
    </tr>
    <tr>
      <th>5758</th>
      <td>Scenarios for MapReduce - Intro to Data Science</td>
      <td>1</td>
      <td>https://www.youtube.com/watch?v=svZh22xgOrk</td>
    </tr>
    <tr>
      <th>1494</th>
      <td>FutureGrid MOOC: IPOP Unit 10: IPOP Applicatio...</td>
      <td>1</td>
      <td>https://www.youtube.com/watch?v=amcddDW52NQ</td>
    </tr>
    <tr>
      <th>234</th>
      <td>Topcoder Data Science Marathon Match: Prostate...</td>
      <td>1</td>
      <td>https://www.youtube.com/watch?v=0-XuFpzDqcY</td>
    </tr>
    <tr>
      <th>4974</th>
      <td>Cats removed from filthy conditions in Barnste...</td>
      <td>1</td>
      <td>https://www.youtube.com/watch?v=PI79k8G6gJ8</td>
    </tr>
    <tr>
      <th>3422</th>
      <td>Grandma is not Cooking for Thanksgiving</td>
      <td>1</td>
      <td>https://www.youtube.com/watch?v=ITK7PkwEfJw</td>
    </tr>
    <tr>
      <th>4300</th>
      <td>Incredible Girl Cooking Water Snake Soup | How...</td>
      <td>1</td>
      <td>https://www.youtube.com/watch?v=mHelogxcLc0</td>
    </tr>
    <tr>
      <th>5961</th>
      <td>किचन टिप्स, Kitchen Tips and Tricks in Hindi -...</td>
      <td>1</td>
      <td>https://www.youtube.com/watch?v=tpRrlCyA7_g</td>
    </tr>
    <tr>
      <th>6098</th>
      <td>Normal Distribution - Intro to Data Science</td>
      <td>1</td>
      <td>https://www.youtube.com/watch?v=UEkOk4eNZdE</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">2</th>
      <th>1367</th>
      <td>Rooftop Cooking: The 'Turfuffle Burger'</td>
      <td>2</td>
      <td>https://www.youtube.com/watch?v=a_8ADiAU-u4</td>
    </tr>
    <tr>
      <th>7047</th>
      <td>Holiday Cooking Challenge: Full Meal Under 5 M...</td>
      <td>2</td>
      <td>https://www.youtube.com/watch?v=yKc1T2OaG64</td>
    </tr>
    <tr>
      <th>6316</th>
      <td>Catch n Cook over night fishing trip Andy's Fi...</td>
      <td>2</td>
      <td>https://www.youtube.com/watch?v=vCsx3f5fNDs</td>
    </tr>
    <tr>
      <th>7056</th>
      <td>Quick Trip to San Francisco and Marin County |...</td>
      <td>2</td>
      <td>https://www.youtube.com/watch?v=yLmx6UrsFM4</td>
    </tr>
    <tr>
      <th>173</th>
      <td>Train a text embedding model</td>
      <td>2</td>
      <td>https://www.youtube.com/watch?v=-jpIbc20V-k</td>
    </tr>
    <tr>
      <th>4739</th>
      <td>Personal Trainers Taste Test Junk Food</td>
      <td>2</td>
      <td>https://www.youtube.com/watch?v=oF-g2i0C1TI</td>
    </tr>
    <tr>
      <th>3617</th>
      <td>How to slice an onion with Francis Lam</td>
      <td>2</td>
      <td>https://www.youtube.com/watch?v=jMTL4-4MsT4</td>
    </tr>
    <tr>
      <th>1087</th>
      <td>Paul Gauguin's cooking lesson (+subs)</td>
      <td>2</td>
      <td>https://www.youtube.com/watch?v=7KYJwnzPMqU</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Game Tree - Georgia Tech - Machine Learning</td>
      <td>2</td>
      <td>https://www.youtube.com/watch?v=_dqKspIU7y4</td>
    </tr>
    <tr>
      <th>2619</th>
      <td>Part I: Vermont gubernatorial candidates debate</td>
      <td>2</td>
      <td>https://www.youtube.com/watch?v=fGo6use3_G0</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">3</th>
      <th>782</th>
      <td>8.3.1 Principal Component Analysis Problem For...</td>
      <td>3</td>
      <td>https://www.youtube.com/watch?v=4Z4-0OUFvN0</td>
    </tr>
    <tr>
      <th>2282</th>
      <td>Using Exponential Functions To Model Data Chap...</td>
      <td>3</td>
      <td>https://www.youtube.com/watch?v=E01E0KMyytE</td>
    </tr>
    <tr>
      <th>494</th>
      <td>1.2.3 Linear Regression with One Variable - Co...</td>
      <td>3</td>
      <td>https://www.youtube.com/watch?v=2i9LdgTbcg8</td>
    </tr>
    <tr>
      <th>2868</th>
      <td>Basics of MapReduce - Intro to Data Science</td>
      <td>3</td>
      <td>https://www.youtube.com/watch?v=gI4HN0JhPmo</td>
    </tr>
    <tr>
      <th>3673</th>
      <td>Plotting in Python - Intro to Data Science</td>
      <td>3</td>
      <td>https://www.youtube.com/watch?v=JUl0GpZKmaY</td>
    </tr>
    <tr>
      <th>7217</th>
      <td>Installing bCNC on Windows 10 64bit (Python 2....</td>
      <td>3</td>
      <td>https://www.youtube.com/watch?v=zedZmYIpGKM</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>Learning Classifier Systems in a Nutshell</td>
      <td>3</td>
      <td>https://www.youtube.com/watch?v=CRge_cZ2cJc</td>
    </tr>
    <tr>
      <th>3585</th>
      <td>[15][1]From PCA to autoencoders (5 mins)</td>
      <td>3</td>
      <td>https://www.youtube.com/watch?v=JIE30MFuWmo</td>
    </tr>
    <tr>
      <th>2211</th>
      <td>Machine Learning A Cappella - Overfitting Thri...</td>
      <td>3</td>
      <td>https://www.youtube.com/watch?v=DQWI1kvmwRg</td>
    </tr>
    <tr>
      <th>6216</th>
      <td>Unit 2 - Anaconda / Screencast tutorial</td>
      <td>3</td>
      <td>https://www.youtube.com/watch?v=uvG8RjOWeV0</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">4</th>
      <th>3423</th>
      <td>TruEat: Cooking With Family</td>
      <td>4</td>
      <td>https://www.youtube.com/watch?v=iTQvd7NpeT8</td>
    </tr>
    <tr>
      <th>2478</th>
      <td>Armenian Sweet Bread Bagharj - Բաղարջ - Heghin...</td>
      <td>4</td>
      <td>https://www.youtube.com/watch?v=ERZ90hO1ZJw</td>
    </tr>
    <tr>
      <th>5620</th>
      <td>Simple Vanilla Cupcakes - Vanilla Muffins by *...</td>
      <td>4</td>
      <td>https://www.youtube.com/watch?v=SdjNAaAiRWs</td>
    </tr>
    <tr>
      <th>4269</th>
      <td>✅Carrot Pickle Recipe- Tasty Gajar-ka-achar by...</td>
      <td>4</td>
      <td>https://www.youtube.com/watch?v=Md27SWm-gmg</td>
    </tr>
    <tr>
      <th>4721</th>
      <td>Peeled Wheat Pilaf with Mushrooms Recipe - Heg...</td>
      <td>4</td>
      <td>https://www.youtube.com/watch?v=Od4ImDurCyE</td>
    </tr>
    <tr>
      <th>4596</th>
      <td>Healthy Cooking Recipes - The Perfect Way To C...</td>
      <td>4</td>
      <td>https://www.youtube.com/watch?v=NSDk2e6iS44</td>
    </tr>
    <tr>
      <th>4132</th>
      <td>How to make #Marble Cake||Tutorial||Cooking wi...</td>
      <td>4</td>
      <td>https://www.youtube.com/watch?v=LPrku3w91m0</td>
    </tr>
    <tr>
      <th>7178</th>
      <td>Discover Method for Cooking Healthy Pork Ribs ...</td>
      <td>4</td>
      <td>https://www.youtube.com/watch?v=Z8tln8N5kuQ</td>
    </tr>
    <tr>
      <th>1996</th>
      <td>Cooking with the versatile corned beef: Reuben...</td>
      <td>4</td>
      <td>https://www.youtube.com/watch?v=cTIu_HI-dHk</td>
    </tr>
    <tr>
      <th>5299</th>
      <td>BBQ MEATBALLS || Cooking with Carly</td>
      <td>4</td>
      <td>https://www.youtube.com/watch?v=qXL7hkwKtrE</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">5</th>
      <th>2267</th>
      <td>BADAM KA SHARBAT *COOK WITH FAIZA*</td>
      <td>5</td>
      <td>https://www.youtube.com/watch?v=Dz48pLoug1k</td>
    </tr>
    <tr>
      <th>572</th>
      <td>Smothered Chicken</td>
      <td>5</td>
      <td>https://www.youtube.com/watch?v=39E0jHXNRmI</td>
    </tr>
    <tr>
      <th>4868</th>
      <td>Awesome Noodles - Cooking 5 KG Prawns with 50 ...</td>
      <td>5</td>
      <td>https://www.youtube.com/watch?v=OxEpDldr88A</td>
    </tr>
    <tr>
      <th>6342</th>
      <td>Dal | Maharashtrian Varan | Indian Recipe by A...</td>
      <td>5</td>
      <td>https://www.youtube.com/watch?v=vhdTkmY-RSs</td>
    </tr>
    <tr>
      <th>5568</th>
      <td>Spicy Masala Rice Recipe in Hindi by Cooking w...</td>
      <td>5</td>
      <td>https://www.youtube.com/watch?v=S0l1pRF2o3Y</td>
    </tr>
    <tr>
      <th>7098</th>
      <td>Whole 24 Healthy Cooking Series: Our Signature...</td>
      <td>5</td>
      <td>https://www.youtube.com/watch?v=Ys3NidnGqdY</td>
    </tr>
    <tr>
      <th>5930</th>
      <td>Pani Puri Recipe in Hindi By Cooking with Smit...</td>
      <td>5</td>
      <td>https://www.youtube.com/watch?v=tm4QidZeriw</td>
    </tr>
    <tr>
      <th>987</th>
      <td>Dylan Metz's ASL Cooking Project</td>
      <td>5</td>
      <td>https://www.youtube.com/watch?v=6PncczzM0SI</td>
    </tr>
    <tr>
      <th>667</th>
      <td>Cooking a Big Fish in My Village - Big Fish Ku...</td>
      <td>5</td>
      <td>https://www.youtube.com/watch?v=4_U1hvM2K90</td>
    </tr>
    <tr>
      <th>5476</th>
      <td>Cooking Goat Intestine in My Village - Our Lun...</td>
      <td>5</td>
      <td>https://www.youtube.com/watch?v=rrcjcCIMJcE</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">6</th>
      <th>7015</th>
      <td>Smoked Fish Dip | Simple Cooking | Carolina Ou...</td>
      <td>6</td>
      <td>https://www.youtube.com/watch?v=YdTFcBdy6zI</td>
    </tr>
    <tr>
      <th>6242</th>
      <td>I/O '17 Guide - Machine Learning</td>
      <td>6</td>
      <td>https://www.youtube.com/watch?v=UyU1k6eebp4</td>
    </tr>
    <tr>
      <th>6972</th>
      <td>How to Make - Roasted Pork Leg</td>
      <td>6</td>
      <td>https://www.youtube.com/watch?v=y79Ccq0e06Y</td>
    </tr>
    <tr>
      <th>2169</th>
      <td>Spinach Stuffed Mushrooms | Hilah Cooking</td>
      <td>6</td>
      <td>https://www.youtube.com/watch?v=dkhy4vn9HcY</td>
    </tr>
    <tr>
      <th>475</th>
      <td>Breakfast Burrito - You Suck at Cooking (episo...</td>
      <td>6</td>
      <td>https://www.youtube.com/watch?v=2DvQLV7niPM</td>
    </tr>
    <tr>
      <th>5223</th>
      <td>Raindrop Cake, Corinne VS Cooking #7</td>
      <td>6</td>
      <td>https://www.youtube.com/watch?v=QLjQcAQcSAQ</td>
    </tr>
    <tr>
      <th>488</th>
      <td>Healthy Cooking &amp; Healthy Living: Ep. 2 Segment 3</td>
      <td>6</td>
      <td>https://www.youtube.com/watch?v=2hEd6HjmsiE</td>
    </tr>
    <tr>
      <th>6485</th>
      <td>WSRE | Gourmet Cooking with Earl Peyroux | Epi...</td>
      <td>6</td>
      <td>https://www.youtube.com/watch?v=vzvOjGDlb0g</td>
    </tr>
    <tr>
      <th>3279</th>
      <td>Cooking Peach Cobbler " Fake Kitchen"</td>
      <td>6</td>
      <td>https://www.youtube.com/watch?v=IaisW2R_Xws</td>
    </tr>
    <tr>
      <th>1409</th>
      <td>Wild Food Cooking - Acorn Bread</td>
      <td>6</td>
      <td>https://www.youtube.com/watch?v=aA2jZSQFUfc</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">7</th>
      <th>5168</th>
      <td>PUPPY PAWS: HOW TO TRAIN A SLED DOG? TESTING T...</td>
      <td>7</td>
      <td>https://www.youtube.com/watch?v=qcyHgpHWsL8</td>
    </tr>
    <tr>
      <th>3137</th>
      <td>Senior dogs surrendered by sick owner to Balti...</td>
      <td>7</td>
      <td>https://www.youtube.com/watch?v=hnsbt_i3rsA</td>
    </tr>
    <tr>
      <th>3133</th>
      <td>How To adopt a pet from a shelter</td>
      <td>7</td>
      <td>https://www.youtube.com/watch?v=HNIlYwCpwJ0</td>
    </tr>
    <tr>
      <th>1995</th>
      <td>Dog Training Tips : How to Feed a Dog</td>
      <td>7</td>
      <td>https://www.youtube.com/watch?v=ctGYUoVcAcA</td>
    </tr>
    <tr>
      <th>1515</th>
      <td>Neo’s Story, the Inspiration Behind NEO-K9 [Do...</td>
      <td>7</td>
      <td>https://www.youtube.com/watch?v=apejntzn57o</td>
    </tr>
    <tr>
      <th>989</th>
      <td>Deaf Pup Learns Doggie Sign Language At Hopkin...</td>
      <td>7</td>
      <td>https://www.youtube.com/watch?v=6PyF3_97l64</td>
    </tr>
    <tr>
      <th>4129</th>
      <td>Dog Training Collars &amp; Harnesses : Using a Hal...</td>
      <td>7</td>
      <td>https://www.youtube.com/watch?v=LpE76_2EnfI</td>
    </tr>
    <tr>
      <th>1472</th>
      <td>Chicopee police dog outfitted with new protect...</td>
      <td>7</td>
      <td>https://www.youtube.com/watch?v=AJa5EthynpM</td>
    </tr>
    <tr>
      <th>6415</th>
      <td>Dog Training : How to Train Your Dog to Heel</td>
      <td>7</td>
      <td>https://www.youtube.com/watch?v=vQYlAkA8gDg</td>
    </tr>
    <tr>
      <th>3025</th>
      <td>How to Turn a Large Sock into a Tiny Dog Sweater</td>
      <td>7</td>
      <td>https://www.youtube.com/watch?v=H4adFCRoUQM</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">8</th>
      <th>5635</th>
      <td>Gnocchi di Zucca - Pumpkin Gnocchi | Cooking w...</td>
      <td>8</td>
      <td>https://www.youtube.com/watch?v=sFKmtk5Iwg0</td>
    </tr>
    <tr>
      <th>1754</th>
      <td>DIY VALENTINES DAY TREATS! Easy And Yummy! Coo...</td>
      <td>8</td>
      <td>https://www.youtube.com/watch?v=btsAVWt4K4I</td>
    </tr>
    <tr>
      <th>4169</th>
      <td>Cooking w/ Coconut w/ Swag SW and Michaela Huff</td>
      <td>8</td>
      <td>https://www.youtube.com/watch?v=lVEJybktsIs</td>
    </tr>
    <tr>
      <th>6749</th>
      <td>Easy No-Bake Cheesecake with Fresh Cheese | Co...</td>
      <td>8</td>
      <td>https://www.youtube.com/watch?v=x7UPnkbHKig</td>
    </tr>
    <tr>
      <th>5621</th>
      <td>Cooking On A Budget | Season 7 Ep. 8 | MASTERCHEF</td>
      <td>8</td>
      <td>https://www.youtube.com/watch?v=sDm1uhYNLTM</td>
    </tr>
    <tr>
      <th>6555</th>
      <td>Convection Cooking 101 | Whirlpool Corporation</td>
      <td>8</td>
      <td>https://www.youtube.com/watch?v=WD7dFwsywIU</td>
    </tr>
    <tr>
      <th>4819</th>
      <td>Get Cooking with dietitian Julie Gieseman</td>
      <td>8</td>
      <td>https://www.youtube.com/watch?v=OQ8bskZVmqc</td>
    </tr>
    <tr>
      <th>6314</th>
      <td>Infrared Cooking Tips</td>
      <td>8</td>
      <td>https://www.youtube.com/watch?v=vByGJGcGU8Y</td>
    </tr>
    <tr>
      <th>6714</th>
      <td>Cooking Breakfast with Gracie from Mommy and G...</td>
      <td>8</td>
      <td>https://www.youtube.com/watch?v=WzCSIDfvhls</td>
    </tr>
    <tr>
      <th>7076</th>
      <td>How to make Russian pelmeni - Cooking with Boris</td>
      <td>8</td>
      <td>https://www.youtube.com/watch?v=YO7AdLsUSec</td>
    </tr>
    <tr>
      <th rowspan="10" valign="top">9</th>
      <th>5655</th>
      <td>How to Do Sentiment Analysis - Intro to Deep L...</td>
      <td>9</td>
      <td>https://www.youtube.com/watch?v=si8zZHkufRY</td>
    </tr>
    <tr>
      <th>5117</th>
      <td>Lesson 7: Practical Deep Learning for Coders</td>
      <td>9</td>
      <td>https://www.youtube.com/watch?v=Q0z-l2KRYFY</td>
    </tr>
    <tr>
      <th>4280</th>
      <td>Diagnosing Concussions Based on High-Resolutio...</td>
      <td>9</td>
      <td>https://www.youtube.com/watch?v=mfEByUCn79s</td>
    </tr>
    <tr>
      <th>340</th>
      <td>DEF CON 24 - Clarence Chio - Machine Duping 10...</td>
      <td>9</td>
      <td>https://www.youtube.com/watch?v=0zAxuH0YsoM</td>
    </tr>
    <tr>
      <th>3539</th>
      <td>8. Autonomous Driving</td>
      <td>9</td>
      <td>https://www.youtube.com/watch?v=JcCzJfeEqoc</td>
    </tr>
    <tr>
      <th>67</th>
      <td>Learn Basics of Machine Learning using TensorF...</td>
      <td>9</td>
      <td>https://www.youtube.com/watch?v=_N9tspk7qbk</td>
    </tr>
    <tr>
      <th>1955</th>
      <td>Introduction to Google Cloud Machine Learning ...</td>
      <td>9</td>
      <td>https://www.youtube.com/watch?v=COSXg5HKaO4</td>
    </tr>
    <tr>
      <th>5038</th>
      <td>Induction and Deduction - Georgia Tech - Machi...</td>
      <td>9</td>
      <td>https://www.youtube.com/watch?v=pqXASFHUfhs</td>
    </tr>
    <tr>
      <th>1593</th>
      <td>Neural Programmer-Interpreters Learn To Write ...</td>
      <td>9</td>
      <td>https://www.youtube.com/watch?v=B70tT4WMyJk</td>
    </tr>
    <tr>
      <th>3835</th>
      <td>Top 3 Essentials for Growth with Google</td>
      <td>9</td>
      <td>https://www.youtube.com/watch?v=kl_O7qhtmNU</td>
    </tr>
  </tbody>
</table>
