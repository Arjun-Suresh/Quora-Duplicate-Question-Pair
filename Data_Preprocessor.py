import nltk
from nltk.tag.stanford import StanfordNERTagger
import re
import os
from string import punctuation
import sys

def removeCamelCasing(text):
    match = text.group(0)
    return match[:-1] + match[-1].lower()

def cleanup(text):

    text = ' ' + text[0].lower() + text[1:] + ' ' #Convert first character to lower

    text = re.sub("(?<=[\.\?\)\!\'\"])[\s]*.", removeCamelCasing, text) #convert first character after .!?)"' characters to lower

    #Substituting characters from other locales to en-us
    text = re.sub("’", "'", text)
    text = re.sub("`", "'", text)
    text = re.sub("“", '"', text)
    text = re.sub("？", "?", text)
    text = re.sub("…", " ", text)
    text = re.sub("é", "e", text)
    text = re.sub("‘", "'", text)
    text = re.sub("’", "'", text)

    text = re.sub("\'s", " ", text) #Changing "Daniel's" to "Daniel"
    text = re.sub(" whats ", " what is ", text, flags=re.IGNORECASE)
    text = re.sub("\'ve", " have ", text) #Changing "I've" to "I have"
    text = re.sub("can't", "can not", text)
    text = re.sub("n't", " not ", text) #Changing "amn't" to "am not"
    text = re.sub("i'm", "i am", text, flags=re.IGNORECASE)
    text = re.sub("\'re", " are ", text) #Changing "you're" to "you are"
    text = re.sub("\'d", " would ", text)  #Changing "I'd" to "I would"
    text = re.sub("\'ll", " will ", text) #Changing "I'll" to "I will"
    text = re.sub("e-mail", " email ", text, flags=re.IGNORECASE)

    text = re.sub("\(s\)", " ", text, flags=re.IGNORECASE)
    #standardizing the use of popular symbols
    text = re.sub('\$', " dollar ", text)
    text = re.sub('\%', " percent ", text)
    text = re.sub('\£', " pound ", text)
    text = re.sub('\&', " and ", text)
    text = re.sub('\@', " at ", text)


    #Standardizing country names to camel casing
    text = re.sub("(the[\s]+|The[\s]+)?U\.S\.A\.", " America ", text, flags=re.IGNORECASE)
    text = re.sub("(the[\s]+|The[\s]+)?United State(s)?", " America ", text, flags=re.IGNORECASE)
    text = re.sub(r" (the[\s]+|The[\s]+)?US(A)? ", " America ", text)
    text = re.sub(r" UK ", " England ", text, flags=re.IGNORECASE)
    text = re.sub(r" india ", " India ", text)
    text = re.sub(r" switzerland ", " Switzerland ", text)
    text = re.sub(r" china ", " China ", text)
    text = re.sub(r" chinese ", " Chinese ", text)

    #Correcting a few common typos in the data
    text = re.sub(r" imrovement ", " improvement ", text, flags=re.IGNORECASE)
    text = re.sub(r" quikly ", " quickly ", text)
    text = re.sub(r" unseccessful ", " unsuccessful ", text)
    text = re.sub(r" demoniti[\S]+ ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" demoneti[\S]+ ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" addmision ", " admission ", text)
    text = re.sub(r" insititute ", " institute ", text)
    text = re.sub(r" connectionn ", " connection ", text)
    text = re.sub(r" permantley ", " permanently ", text)
    text = re.sub(r" sylabus ", " syllabus ", text)
    text = re.sub(r" sequrity ", " security ", text)
    text = re.sub(r" undergraduation ", " undergraduate ", text)  # not typo, but GloVe can't find it
    text = re.sub(r"(?=[a-zA-Z])ig ", "ing ", text)
    text = re.sub(r" latop", " laptop", text)
    text = re.sub(r" programmning ", " programming ", text)
    text = re.sub(r" begineer ", " beginner ", text)
    text = re.sub(r" qoura ", " Quora ", text)
    text = re.sub(r" wtiter ", " writer ", text)
    text = re.sub(r" litrate ", " literate ", text)
    text = re.sub(r" intially ", " initially ", text, flags=re.IGNORECASE)

    #expanding abbreviations and other miscellaneous transformations
    text = re.sub(r" quora ", " Quora ", text, flags=re.IGNORECASE)
    text = re.sub(r" dms ", " direct messages ", text, flags=re.IGNORECASE)
    text = re.sub(r" demonitization ", " demonetization ", text, flags=re.IGNORECASE)
    text = re.sub(r" actived ", " active ", text, flags=re.IGNORECASE)
    text = re.sub(r" kms ", " kilometers ", text, flags=re.IGNORECASE)
    text = re.sub(r" cs ", " computer science ", text, flags=re.IGNORECASE)
    text = re.sub(r" upvote", " up vote", text, flags=re.IGNORECASE)
    text = re.sub(r" iPhone ", " phone ", text, flags=re.IGNORECASE)
    text = re.sub(r" \0rs ", " rs ", text, flags=re.IGNORECASE)
    text = re.sub(r" calender ", " calendar ", text, flags=re.IGNORECASE)
    text = re.sub(r" ios ", " operating system ", text, flags=re.IGNORECASE)
    text = re.sub(r" gps ", " GPS ", text, flags=re.IGNORECASE)
    text = re.sub(r" gst ", " GST ", text, flags=re.IGNORECASE)
    text = re.sub(r" programing ", " programming ", text, flags=re.IGNORECASE)
    text = re.sub(r" bestfriend ", " best friend ", text, flags=re.IGNORECASE)
    text = re.sub(r" dna ", " DNA ", text, flags=re.IGNORECASE)
    text = re.sub(r" III ", " 3 ", text)
    text = re.sub(r" banglore ", " Banglore ", text, flags=re.IGNORECASE)
    text = re.sub(r" J K ", " JK ", text, flags=re.IGNORECASE)
    text = re.sub(r" J\.K\. ", " JK ", text, flags=re.IGNORECASE)

    # reduce extra spaces into single spaces
    text = re.sub('[\s]+', " ", text)
    text = ''.join([c for c in text if c not in punctuation])  # Remove any punctuations
    text = re.sub('[^\x00-\x7F]+', "", text)
    return text

def namedEntityMatch(row):
    #os.environ["JAVA_HOME"]="C:\Program Files\Java\jdk1.8.0_151"
    if "JAVA_HOME" not in os.environ:
        print ("Please set the value of JAVA_HOME environment variable, or install java in your machine")
        sys.exit(-1)

    ner = StanfordNERTagger(r"stanford-ner-2014-06-16\classifiers\english.all.3class.distsim.crf.ser.gz",
                                      r"stanford-ner-2014-06-16\stanford-ner.jar")
    
    ques1Entities = ner.tag(str(row['question1']).lower().split())
    ques2Entities = ner.tag(str(row['question2']).lower().split())
    entityDict1 = {}
    entityDict2 = {}
    for entity in entityDict1:
        if entity[1] != "0":
            if entity[1] in entityDict1:
                entityDict1[entity[1]].append(entity[0])
            else:
                nameList=[]
                nameList.append(entity[0])
                entityDict1[entity[1]]=nameList

    for entity in entityDict2:
        if entity[1] != "0":
            if entity[1] in entityDict2:
                entityDict2[entity[1]].append(entity[0])
            else:
                nameList=[]
                nameList.append(entity[0])
                entityDict2[entity[1]]=nameList


    if len(entityDict1) == 0 or len(entityDict2) == 0:
        return 0

    totalCount=0
    matchCount=0
    for key in entityDict1:
        entityList1=entityDict1[key]
        if key in entityDict2:
            entityList2=entityDict2[key]
            for item in entityList1:
                if item in entityList2:
                    matchCount+=1
                totalCount+=1
    for key in entityDict2:
        entityList2=entityDict2[key]
        if key in entityDict1:
            entityList1=entityDict1[key]
            for item in entityList2:
                if item in entityList1:
                    matchCount+=1
                totalCount+=1

    return float(matchCount)/float(totalCount)


