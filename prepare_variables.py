import sys
sys.path.append('../')
import config
import pymysql.cursors
import pandas as pd
import numpy as np
from scipy import io as scipyio
from tempfile import SpooledTemporaryFile
from scipy.sparse import vstack as vstack_sparse_matrices

# Function to reassemble the p matrix from the vectors

def reconstitute_vector(bytesblob):
    f = SpooledTemporaryFile(max_size=1000000000)
    f.write(bytesblob)
    f.seek(0)
    return scipyio.mmread(f)

def youtubelink(vidid):
    return ('https://www.youtube.com/watch?v=' + vidid)    


connection = pymysql.connect(host='localhost',
                             user='root',
                             password=config.MYSQL_SERVER_PASSWORD,
                             db='youtubeProjectDB',
                             charset='utf8mb4', 
                             cursorclass=pymysql.cursors.DictCursor)

    
with connection.cursor() as cursor:                                 
            # https://stackoverflow.com/questions/612231/how-can-i-select-rows-with-maxcolumn-value-distinct-by-another-column-in-sql?rq=1
            # Note - this is a very interesting query! never seen it before..
            sql = """SELECT * FROM
            (SELECT DISTINCT(videoId) AS v, videoTitle FROM search_api) A
            INNER JOIN
            (SELECT * FROM captions c
            INNER JOIN(SELECT videoId AS InnerVideoId, 
            MAX(wordCount) AS MaxWordCount, 
            MAX(id) AS MaxId
            FROM captions 
            WHERE tfidfVector IS NOT NULL 
            GROUP BY videoId) grouped_c
            ON c.videoId = grouped_c.InnerVideoId
            AND c.wordCount = grouped_c.MaxWordCount
            AND c.id = grouped_c.MaxId) B
            ON A.v = B.videoId;"""
            cursor.execute(sql)
            manyCaptions = cursor.fetchall()
            videos_df = pd.read_sql(sql, connection)                        
connection.close()

# note that the other program which put the vectors there only did it on captions WHERE language like '%en%'
# for that reason this query does not contain language. It has instead WHERE tfidfVector IS NOT NULL

videos_df = videos_df.drop('v', 1)

videos_df['tfidfVector_NP'] = videos_df['tfidfVector'].apply(reconstitute_vector)
listOfSparseVectors = list(videos_df['tfidfVector_NP'].values.flatten())
videos_df['link'] = videos_df['videoId'].apply(youtubelink)
p = vstack_sparse_matrices(listOfSparseVectors)