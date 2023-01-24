from google.colab import drive
drive.mount('/content/drive')

!pip install pyspark

from pyspark.sql import SparkSession
from pyspark.sql.functions import count, desc , col, max, struct
import matplotlib.pyplot as plts

spark = SparkSession.builder.appName('spark_app').getOrCreate()

listening_csv_path = '/content/drive/MyDrive/dataset/listenings.csv'
listening_df = spark.read.format('csv').option('inferSchema', True).option('header', True).load(listening_csv_path)

listening_df.show()

listening_df = listening_df.drop('date')

listening_df.printSchema()

shape = (listening_df.count(), len(listening_df.columns))
print(shape)


#0: select two columns: track and artist
q0 = listening_df.select('artist', 'track')
q0.show()

#1:Let's find all of the records of those users who have listened to Rihanna
q1 = listening_df.select('*').filter(listening_df.artist == 'Rihanna')
q1.show()

#2:Let's find top 10 users who are fan of Rihanna
q2 = listening_df.select('user_id').filter(listening_df.artist =='Rihanna').groupby('user_id').agg(count('user_id').alias('count')).orderBy(desc('count')).limit(10)
q2.show()

#3:find top 10 famous tracks
q3 = listening_df.select('artist', 'track').groupBy('artist','track').agg(count('*').alias('count')).orderBy(desc('count')).limit(10)
q3.show()

#4:find top 10 famous tracks of Rihanna
q4 = listening_df.select('artist', 'track').filter(listening_df.artist == 'Rihanna').groupBy('artist','track').agg(count('*').alias('count')).orderBy(desc('count')).limit(10)
q4.show()

#5:find top 10 famous albums
q5 = listening_df.select('artist', 'album').groupBy('artist','album').agg(count('*').alias('count')).orderBy(desc('count')).limit(10)
q5.show()

#importing the genre.csv file:
genre_csv_path = '/content/drive/MyDrive/dataset/genre.csv'
genre_df = spark.read.format('csv').option('inferSchema', True).option('header', True).load(genre_csv_path)

genre_df.show()

#inner join these two data frames
data = listening_df.join(genre_df, how ='inner', on=['artist'])
data.show()

#6find top 10 users who are fan of pop music

q6 = data.select('user_id').filter(data.genre =='pop').groupBy('user_id').agg(count('*').alias('count')).orderBy(desc('count')).limit(10)
q6.show()

#7find top 10 famous genres
q7 = data.select('genre').groupBy('genre').agg(count('*').alias('count')).orderBy(desc('count')).limit(10)
q7.show()

#8find out each user favourite genre
q8_1 = data.select('user_id','genre').groupBy('user_id','genre').agg(count('*').alias('count')).orderBy('user_id')
q8_1.show()

q8_2 = q8_1.groupBy('user_id').agg(max(struct(col('count'),col('genre'))).alias('max')).select(col('user_id'),col('max.genre'))
q8_2.show()

##9find out how many pop,rock,metal and hip hop singers we haveand then visulize it using bar chart
q9 = genre_df.select('genre').filter( (col('genre')=='pop') | (col('genre')=='rock') | (col('genre')=='metal') | (col('genre')=='hip hop')).groupBy('genre').agg(count('genre').alias('count'))
q9.show()

q9_list=q9.collect()

labels = [row['genre'] for row in q9_list]
counts = [row['count'] for row in q9_list]

print(labels)
print(counts)

plts.bar(labels,counts)


          
     

