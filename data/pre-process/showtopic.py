import codecs
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import re
import gensim
vocab_size = 120000

model = gensim.models.LdaModel.load("STC-2-rm-single.lda")
dictionary = corpora.Dictionary.load("STC-2-rm-single.lda.id2word") 
while True:
    query = raw_input(">>")
    tokens = re.split(" +", query)
    tokens = [each for each in tokens if len(each.decode("utf-8")) > 1]
    bow = dictionary.doc2bow(tokens)
    topics = model[bow]
    for t in topics: 
        print t, model.print_topic(topicno=t[0], topn=10)
