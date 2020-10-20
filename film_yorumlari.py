
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup



#dataframe

df = pd.read_csv('IMDB Dataset.csv')
text = df.head()


def removeHTML(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


maxObs = 20
observation = ''

for n in range(maxObs):
    observation += removeHTML(df.iloc[n][0])


observationClean = ','.join(i.lower() for i in observation.split() if i.isalnum())

setOfWords = set(observationClean.split(','))

dictList = []
for n in tqdm(range(len(df))):
    observation = removeHTML(df.iloc[n][0])
    clean = ','.join(i.lower() for i in observation.split() if i.isalnum())
    dictOfWords = dict.fromkeys(setOfWords, 0)
    for word in clean.split(','):
        if word in dictOfWords:
            dictOfWords[word] += 1
    dictList.append(dictOfWords)


pd.DataFrame(dictList).sample(10)

    
