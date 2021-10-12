import tensorflow as tf
import nltk
#nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import random
import json

bot_name = "SmartBot"
with open('teamcenter_data.json', 'r') as f:
    intents = json.load(f)


words = []
parent_classes = []
classes = []
documents = []
ignore = ['?','!','*']
# loop through each sentence in the intent's patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each and every word in the sentence
        w = nltk.word_tokenize(pattern)
        # add word to the words list
        words.extend(w)
        # add word(s) to documents
        documents.append((w, intent['tag']))
        # add tags to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
        if intent['parent_tag'] not in parent_classes:
            parent_classes.append((intent['parent_tag']))


# Perform stemming and lower each word as well as remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

# remove duplicate classes
classes = sorted(list(set(classes)))
parent_classes = sorted(list(set(parent_classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)
print (len(parent_classes), "parent classes", parent_classes)

training = []
output = []
# create an empty array for output
output_empty = [0] * len(classes)
output_empty_1 = [0] * len(parent_classes)

# create training set, bag of words for each sentence
for doc in documents:
    # initialize bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stemming each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is '1' for current tag and '0' for rest of other tags
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    output_row_1 = list(output_empty_1)
   # output_row_1[parent_classes.index((doc[2]))] = 1
    training.append([bag, output_row])

# shuffling features and turning it into np.array
random.shuffle(training)
training = np.array(training)

# creating training lists
train_x = list(training[:,0])
train_y = list(training[:,1])


from tensorflow.python.framework import ops

ops.reset_default_graph()

#tf.reset_default_graph()

# Building neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Defining model and setting up tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Start training
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')


import pickle
pickle.dump( {'words':words, 'classes':classes,'parent_classes':parent_classes , 'train_x':train_x, 'train_y':train_y}, open( "training_data_old", "wb" ) )


#test.py


import socket
import time
import threading
import requests
import ast
from console_log import ConsoleLog

import logging

logger = logging.getLogger('console')



from flask import Flask, render_template, request
from test import response
app = Flask(__name__)


app.static_folder = './static'
app.temp_dict = {}

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/get")
def get_bot_response():
    logger.info("Error logged from Python")
    userText = request.args.get('msg')
    msg5 = response(userText)
    #print(type(msg5))
    temp1 = {}
    if userText.isnumeric():
        if int(userText) in app.temp_dict.keys():
            userText = app.temp_dict[int(userText)]
            userText = ''.join(userText)

    if 'form' in msg5:
        import webbrowser

        webbrowser.open("D:/PLM Nordic/UOM_request_form.docx")
    if '/' in msg5:
        if 'open' in userText:
            import webbrowser
            webbrowser.open(msg5)

            return "File opened successfully!"

            # self.text_widget.insert(END,"\n")

        if 'delete' in userText:
            import os
            if os.path.exists(msg5):
                os.remove(msg5)
                msg5 = "File Deleted Successfully!"
                return msg5

            else:
                msg5 = "File Dose Not Exist!"
                return msg5
    if (type(msg5) is list) == True:
            #print("yes")

            listToStr = ''.join([str(elem) for elem in msg5])
            #print(type(listToStr))
            #print(listToStr)
            #return listToStr
            res = [''.join(ele) for ele in msg5]
            return render_template('index.html')



            temp_num = 0
            for i in res[1:]:
                temp_num = temp_num + 1
                app.temp_dict[temp_num] = []

                app.temp_dict[temp_num].append(str(i))


            return  app.temp_dict
       


    

    return str(response(userText))
conversation=[] # Our all conversation

# threading the recv function




    




if __name__ == "__main__":
    app.run()
