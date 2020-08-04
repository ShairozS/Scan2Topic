
'''
Read in a list of scanned images (as .png files > 50x50px) and output a set of .txt files containing the text content of these scans
'''


from functions import preprocess, image_to_text
from PIL import Image
import os
from spellchecker import SpellChecker
import matplotlib.pyplot as plt
# MSFT Computer Vision API
import base64
import http.client, urllib.request, urllib.parse, urllib.error, base64
import io
import json
import requests
import urllib

INPUT_FOLDER = r'F:\Data\Imagery\OCR2\Images'
OUTPUT_FOLDER = r'F:\Research\OCR\Outputs\AllDocuments'


## First, read in all the scanned document images into PIL images
scanned_docs_path = os.listdir(INPUT_FOLDER)
scanned_docs_path = [x for x in scanned_docs_path if x.endswith('.png')]
scanned_docs = [Image.open(os.path.join(INPUT_FOLDER, path)) for path in scanned_docs_path]
#scanned_docs = [Image.fromarray(x) for x in scanned_docs]


## Second, utilize Microsoft CV API to extract text from these images using OCR
scanned_docs_text = image_to_text(scanned_docs)


## Third, remove mis-spellings that might have occured from bad OCR readings
spell = SpellChecker()
for i in range(len(scanned_docs_text)):
    clean = scanned_docs_text[i]
    misspelled = spell.unknown(clean)
    clean = clean.split(" ")
    for word in range(len(clean)):
        if clean[word] in misspelled:
            clean[word] = spell.correction(clean[word])# Get the one `most likely` answer
    clean = ' '.join(clean)
    scanned_docs_text[i] = clean
    
    
    

## Fourth, write the extracted text to individual .txt files with the same name as input files
for k in range(len(scanned_docs_text)): # For each scanned document

    text = scanned_docs_text[k]
    path = scanned_docs_path[k] # Get the corresponding input filename
    
    text_file_path = path[:-4] + ".txt" # Create the output text file
    text_file = open(text_file_path, "wt")
    
    n = text_file.write(text) # Write the text to the ouput text file
    
    text_file.close()

print("Done")
