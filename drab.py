#install pyspark
! pip install pyspark

#create a sparksession
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('spark').getOrCreate()

#clone the diabetes dataset from the github repository
! git clone https://github.com/education454/diabetes_dataset

#check if the dataset exists
! ls diabetes_dataset

#create spark dataframe
df = spark.read.csv('/content/diabetes_dataset/diabetes.csv', header=True, inferSchema=True)

#display the dataframe
df.show()

#print the schema
df.printSchema()

#count the total no. of diabetic and non-diabetic class
print((df.count(),len(df.columns)))
df.groupBy('Outcome').count().show()

#get the summary statistics
df.describe().show()

#check for null values

for col in df.columns:
  print(col+":",df[df[col].isNull()].count())
  
#look for the unnecessary values present
def count_zeros():
  column_list = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
  for i in column_list:
    print(i+":",df[df[i]==0].count())
    
#calculate and replace the unnecessary values by the mean value
from pyspark.sql.functions import *
for i in df.columns[1:6]:
  data = df.agg({i:'mean'}).first()[0]
  print("mean valu for {} is {}".format(i, 
                                       int(data)))
  df = df.withColumn(i,when(df[i]==0,int(data)).otherwise(df[i]))

#display the dataframe 
df.show()


#find the correlation among the set of input & output variables
for i in df.columns:
  print("Correlation outcome for {} is {}".format(i,df.stat.corr('Outcome', i)))


#@title Default title text
#feature selection
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'].outputCol='features')
output_data=assembler.transform(df)

#print the schema
output_data.printSchema()

#display dataframe
output_data.show()

#create final data
from pyspark.ml.classification import LogisticRegression
final_data=output_data.select('features','Outcome')

#print schema of final 
final_data.printSchema()

#split the dataset ; build the model
train, test = final_data.randomSplit([0.7,0.3])
models = LogisticRegression(labelCol='Outcome')
model = models.fit(train)

#summary of the model
summary = model.summary
summary.predictions.describe().show()

from pyspark.ml.evaluation import BinaryClassificationEvaluator
predictions = model.evaluate(test)

predictions.predictions.show(20)

evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction',labelCol="Outcome")
evaluator.evaluator(model.transform(test))

# save model
model.save("model")

# load saved model back to the environment
from pyspark.ml.classification import LogisticRegressionModel
model = LogisticRegressionModel.load("model")

#create a new spark dataframe
df_test = spark.read.csv('/content/diabetes_dataset/new_test.csv', header=True, inferSchema=True)

#print the schema
df_test.printSchemat()

#create an additional feature merged column 
test_data = assembler.transform(test_df)

#print the schema
test_data.priontSchema()

#use model to make predictions
results = model.transform(test_data)
results.printSchema()

#display the predictions
results.select('features','prediction').show()
