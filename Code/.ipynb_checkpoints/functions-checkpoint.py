# MSFT Computer Vision API
import base64
import http.client, urllib.request, urllib.parse, urllib.error, base64
import io
import json
import requests
import urllib

from gensim import corpora, models, similarities
import re
from gensim.utils import tokenize
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string
from spellchecker import SpellChecker


def preprocess(document, spellchecker):
    clean = remove_stopwords(document)
    clean = preprocess_string(document) 

    # find those words that may be misspelled
    #misspelled = spellchecker.unknown(clean)
    #for word in range(len(clean)):
    #    if clean[word] in misspelled:
    #        clean[word] = spellchecker.correction(clean[word])# Get the one `most likely` answer
    
    return(clean)

    

    
    
def image_to_text(imglist, ndocs=10):
    '''
    Take in a list of PIL images and return a list of extracted text
    '''
    headers = {
        # Request headers
        'Content-Type': 'application/octet-stream',
        'Ocp-Apim-Subscription-Key': '89279deb653049078dd18b1b116777ea',
    }

    params = urllib.parse.urlencode({
        # Request parameters
        'language': 'en',
        'detectOrientation ': 'true',
    })
    
    outtext = []
    docnum = 0
    
    for cropped_image in imglist[0:ndocs]:
        print("Processing document -- ", str(docnum))
        # Cropped image must have both height and width > 50 px to run Computer Vision API
        #if (cropped_image.height or cropped_image.width) < 50:
        #    cropped_images_ocr.append("N/A")
        #    continue
        ocr_image = cropped_image
        imgByteArr = io.BytesIO()
        ocr_image.save(imgByteArr, format='PNG')
        imgByteArr = imgByteArr.getvalue()

        try:
            conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
            conn.request("POST", "/vision/v1.0/ocr?%s" % params, imgByteArr, headers)
            response = conn.getresponse()
            data = json.loads(response.read().decode("utf-8"))
            
            curr_text = []
            for r in data['regions']:
                for l in r['lines']:
                    for w in l['words']:
                        curr_text.append(str(w['text']))
            conn.close()
        except Exception as e:
            print("[Errno {0}] {1}".format(e.errno, e.strerror))
        
        outtext.append(' '.join(curr_text))
        docnum += 1

    return(outtext)




def run_lda(textlist, 
            num_topics=10,
            return_model=False,
            preprocess_docs=True):
    
    '''
    Train and return an LDA model against a list of documents
    '''
    spellchecker = SpellChecker()
    if preprocess_docs:
        doc_text  = [preprocess(d, spellchecker) for d in textlist]
    dictionary = corpora.Dictionary(doc_text)
    
    corpus = [dictionary.doc2bow(text) for text in doc_text]
    tfidf = models.tfidfmodel.TfidfModel(corpus)
    transformed_tfidf = tfidf[corpus]
    
    lda = models.ldamulticore.LdaMulticore(transformed_tfidf, num_topics=num_topics, id2word=dictionary)
    
    input_doc_topics = lda.get_document_topics(corpus)
    
    return(lda, dictionary)
    
    
    
    
def get_topic(lda_result, num_topics=1):
    '''
    Tag a list of documents with num_topics topics, using the LDA model trained on that list of documents
    '''
    document_topics = lda_result.show_topics(formatted=False, num_words=50)
    document_topics = [t[1] for t in document_topics]
    document_topics = [list(dict(x).keys()) for x in document_topics]
    document_topics = [item for sublist in document_topics for item in sublist]
    
    topK = Counter(document_topics).most_common(num_topics)
    
    return(topK)


def find_topic(testObj, dictionary, lda):
    text_corpus = []
    '''
    https://stackoverflow.com/questions/16262016/how-to-predict-the-topic-of-a-new-query-using-a-trained-lda-model-using-gensim
    
     For each query ( document in the test file) , tokenize the 
     query, create a feature vector just like how it was done while training
     and create text_corpus
    '''
    for query in testObj:
        temp_doc = tokenize(query.strip())
        current_doc = []
        temp_doc = list(temp_doc)
        for word in range(len(temp_doc)):
            current_doc.append(temp_doc[word])

        text_corpus.append(current_doc)
    '''
     For each feature vector text, lda[doc_bow] gives the topic
     distribution, which can be sorted in descending order to print the 
     very first topic
    ''' 
    tops = []
    for text in text_corpus:
        doc_bow = dictionary.doc2bow(text)
        topics = sorted(lda[doc_bow],key=lambda x:x[1],reverse=True)[0]
        tops.append(topics)
    return(tops)


def topic_label(ldamodel, topicnum):
    alltopics = ldamodel.show_topics(formatted=False)
    topic = alltopics[topicnum]
    topic = dict(topic[1])
    import operator
    return(max(topic, key=lambda key: topic[key]))
