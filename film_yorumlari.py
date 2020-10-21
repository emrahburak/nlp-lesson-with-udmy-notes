
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
#dataframe

df = pd.read_csv('IMDB Dataset.csv')
y = df.sentiment.replace({"positive":1,"negative":0})
x = df.review

bag = CountVectorizer()
X = bag.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = RandomForestClassifier(n_jobs=-1)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)

cm = confusion_matrix(y_test, preds)

test1 = "I really did not enjoy watching this , Very disappointed"
test2 = "What a wanderful movie, I enjoyed wathcing this witch my kids"

pre_1 = clf.predict(bag.transform([test1]))
pre_2 = clf.predict(bag.transform([test2]))

print(pre_1)
print(pre_2)




