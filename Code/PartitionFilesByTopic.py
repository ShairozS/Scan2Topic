import os

from functions import preprocess, image_to_text, find_topic, run_lda, topic_label
from PIL import Image
import os
from spellchecker import SpellChecker
import matplotlib.pyplot as plt
import shutil


OUTPUT_FOLDER = r'F:\Research\OCR\Outputs\AllDocuments'

def read_and_return(foldername, fileext='.txt'):
    allfiles = os.listdir(foldername)
    allfiles = [os.path.join(foldername, f) for f in allfiles if f.endswith(fileext)]
    alltext = []
    for filename in allfiles:
        with open(filename, 'r') as f:
            alltext.append((filename, f.read()))
        f.close()
    return(alltext) # Returns list of tuples [(filename, text), ... (filename,text)]

if __name__ == '__main__':

    OUTPUT_FOLDER = r'F:/Research/OCR/Outputs/AllDocumentsByTopic/'

    # First, read all the output .txt files
    print("Reading files")
    texts = read_and_return(r'F:\Research\OCR\Code')


    # Second, train the LDA model (pre-processing is internally done)
    print("Preprocessing Text")
    textlist = [t[1] for t in texts]
    ldamodel, dictionary = run_lda(textlist)


    # Third, extract the top topic for each document
    print("Extracting Topics")
    topics = []
    for t in texts:
        topics.append((t[0], find_topic([t[1]], dictionary, ldamodel)))


    # Convert topics to topic names
    for i in range(len(topics)):
        topnum = topics[i][1][0][0]
        #print(topnum)
        topics[i][1][0] = topic_label(ldamodel, topnum)
        # [(filename, topic), ..., (filename, topic)]


    # Create folders for the topics
    print("Copying Documents into Topic Folders")
    foundtopics = []
    for t in topics:
        foundtopics+= t[1]
    foundtopics = set(foundtopics)
    topicfolders = [os.path.join(OUTPUT_FOLDER, f) for f in foundtopics]
    [os.makedirs(m) for m in topicfolders]

    # Copy files into appropriate topic folders
    for t in topics:
        filename, topic = t
        src = filename
        filename = filename.split("\\")
        dest = os.path.join(OUTPUT_FOLDER, topic[0])
        dest = dest + "/" + filename[-1]
        copystr = "copy " + src + " " + dest
        shutil.copyfile(src, dest)

    print("Done")