# Clustering the Videos from their Caption Vectors

## Preprocessing was needed.

The sparse vectors pulled out of the youtube database are created only from the scripts or subtitles of the video, not the titles or anything else.

The first time I tried to cluster these vectors with KMeans, it found some clusters, but you could say that no matter whether you went for a small number or a large number of clusters, several of the clusters were quite impure and mixed, say mainly one type of video but with several other types thrown in. Experimenting with DBSCAN did not really make it any better. What *did* help in particular was normalizing the vectors before KMeans. This is exactly what I later realised the SKLearn documentation recommends. Principle component analysis - PCA was also tested, reducing the number of components from 429429 down to 500 or 1000. Actually, it was extremely memory intensive and could not go much above that many components. Running a pipeline with normalization and KMeans performed about the same with or without PCA but takes 4 seconds with PCA compared with 4 minutes without. That can be handy for experimenting with the number of clusters. 

The clustering with preprocessing will soon be in this repo for now at the bottom of [this notebook](clustering_without_preprocessing.ipynb) are the results of the original clustering with just KMeans. 

