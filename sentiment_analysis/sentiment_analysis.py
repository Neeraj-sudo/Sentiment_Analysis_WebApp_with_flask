#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# In[16]:


traindf = pd.read_csv('/home/neeraj/python_programs/kaggle_practice/tweet-sentiment-extraction/train.csv')
traindf.head()


# In[17]:


traindf['sentiment'].value_counts().plot(kind = 'bar')


# In[18]:


traindf = traindf.dropna()


# In[19]:


traindf['text'] = traindf['text'].astype('U')


# In[20]:


traindf['length'] = traindf['text'].apply(len)
traindf.head()


# In[21]:


import string

mess = 'Sample message! Notice: it has punctuation.'
from nltk.corpus import stopwords
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[22]:


from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(traindf['text'])
messages_bow = bow_transformer.transform(traindf['text'])


# In[23]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[24]:


tfidf_transformer = TfidfTransformer().fit(messages_bow)
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# In[26]:


print(type(messages_tfidf[10]))


# In[27]:


from sklearn.naive_bayes import MultinomialNB
sentiment_model = MultinomialNB().fit(messages_tfidf, traindf['sentiment'])


# In[28]:


predictions = sentiment_model.predict(messages_tfidf)
print(predictions)


# In[29]:


import pickle 
  
pickle.dump(sentiment_model, open('model.pkl', 'wb')) 


# In[30]:


print("Model is ready")


# In[32]:


from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[33]:


pipeline.fit(traindf['text'],traindf['sentiment'])


# In[34]:


testdf = pd.read_csv('/home/neeraj/python_programs/kaggle_practice/tweet-sentiment-extraction/test.csv')
testdf.head()


# In[35]:


text1 = "i am happy, it was a good day"


# In[37]:


print(pipeline.predict([text1]))


# In[39]:


pickle.dump(pipeline, open('pipe_model.pkl', 'wb'))


# In[40]:


pipe_model = pickle.load(open('pipe_model.pkl', 'rb'))


# In[43]:


type(pipe_model.predict([text1]))


# In[ ]:




