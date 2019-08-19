import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

yelp = pd.read_csv("E:/Py-DS-ML-Bootcamp-master/Refactored_Py_DS_ML_Bootcamp-master/20-Natural-Language-Processing/yelp.csv")
print("Head of yelp :"),print(yelp),print()
print("Info of yelp data frame:"),print(yelp.info()),print()
print("Description of yelp data frame:"),print(yelp.describe()),print()

yelp["text length"] = yelp["text"].apply(len)

# fg = sns.FacetGrid(col="stars",data=yelp)
# fg = fg.map(plt.hist,"text length",alpha=0.5).add_legend()
#
# plt.figure(num=2)
# sns.boxplot(x="stars",y="text length",data=yelp,palette="rainbow")
#
# plt.figure(num=3)
# sns.countplot(x="stars",data=yelp)

print("Get mean value of numerical columns by using group by stars :")
grouped_df = yelp.groupby("stars").mean()
print(grouped_df),print()

print("Correlated grouped by data frame :")
corr_grouped_df = grouped_df.corr()
print(corr_grouped_df),print()

# plt.figure(num=4,figsize=(8,8))
# sns.heatmap(corr_grouped_df,annot=True)

yelp_class = yelp[(yelp["stars"] ==1) | (yelp["stars"] ==5)]
X = yelp_class["text"]
y = yelp_class["stars"]

vectoriser = CountVectorizer()
X = vectoriser.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
nb = MultinomialNB()
nb.fit(X_train,y_train)

predictions = nb.predict(X_test)

print("Classification report :")
print(classification_report(y_test,predictions)),print()

#### Using TF-IDF and pipeline now to predict
pipeline = Pipeline([
    ("count_vectoriser",CountVectorizer()),
    ("tf-idf_transformer",TfidfTransformer()),
    ("classifier",MultinomialNB())
])

yelp_class = yelp[(yelp["stars"] ==1) | (yelp["stars"] ==5)]
X = yelp_class["text"]
y = yelp_class["stars"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
pipeline.fit(X_train,y_train)
pipeline_predictions = pipeline.predict(X_test)

print("Classification report of pipeline :")
print(classification_report(y_test,pipeline_predictions)),print()

plt.show()