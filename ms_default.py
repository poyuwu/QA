# coding=utf-8
import json
import os
import numpy as np
import re
import string
regex = re.compile('[%s]' % re.escape('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\“”'))
from nltk import word_tokenize,sent_tokenize
import nltk
import sys
from pythonrouge import pythonrouge
def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|-|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    vTEXT = re.sub(ur'—|–', ' ', vTEXT, flags=re.MULTILINE)
    return(vTEXT)
#from nltk.stem.porter import *
#stemmer = PorterStemmer().stem
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
id_mapping = {"PAD_ID": 0,"EOS_ID":1,"UNK_ID":2,"GO":3}
id2word = ["PAD_ID","EOS_ID","UNK_ID","GO"]
count = 4
#read data
def pad_array(arr,seq_len=None): #2D
    if seq_len is None:
        M = max(len(a) for a in arr)
        return np.array([a + [0] * (M - len(a)) for a in arr])
    else:
        output = np.zeros((len(arr),seq_len))
        for i in range(len(arr)):
            for j in range(seq_len):
                if j < len(arr[i]):
                    output[i][j] = arr[i][j]
        return output.astype(int)
facts = []
question = []
answer = []
max_len = 0
q_max_len = 0
max_decode_length = 27
decode_len = 0.
encode_length = []
q_length = []
decode_length = []
counter = 0
with open('train_v1.1.json') as f:
    for line in f:
        line = json.loads(line)
        if len(line['answers']) == 0:
            continue
        ans_temp = []
        for token in word_tokenize(remove_urls(line['answers'][0].lower())):
            token = regex.sub(u'',token)
            if token == u'':
                continue
            if token in id_mapping:
                ans_temp.append(id_mapping.get(token))
            else:
                ans_temp.append(count)
                id_mapping.update({token:count})
                id2word.append(token)
                count += 1
        ans_temp.append(1)
        temp_facts = []
        is_selected = False
        for passages in line['passages']:
            if passages['is_selected']  == 1:
                temp = []
                for token in word_tokenize(remove_urls(passages['passage_text'].lower())):
                    token = regex.sub(u'',token)
                    if token == u'':
                        continue
                    if token in id_mapping:
                        temp.append(id_mapping.get(token))
                    else:
                        temp.append(count)
                        id_mapping.update({token:count})
                        id2word.append(token)
                        count += 1

                temp_facts += temp
                is_selected = True
                #temp_facts.append(temp)
        temp_facts.append(1)
        if is_selected == False:
            continue
        facts.append(temp_facts)
        encode_length.append(len(temp_facts))
        max_len = max(max_len,len(temp_facts))
        #ans
        if len(ans_temp) > max_decode_length:
            continue
        else:
            decode_length.append(len(ans_temp))
        answer.append(ans_temp)

        temp = []
        for token in word_tokenize(remove_urls(line['query'].lower())):
            token = regex.sub(u'',token)
            if token == u'':
                continue
            if token in id_mapping:
                temp.append(id_mapping.get(token))
            else:
                temp.append(count)
                id_mapping.update({token:count})
                id2word.append(token)
                count += 1
        temp.append(1)
        question.append(temp)
        q_length.append(len(temp))
        q_max_len = max(q_max_len,len(temp))
#print decode_len/82326
facts = pad_array(facts,max_len)
question = pad_array(question,q_max_len)
answer = pad_array(answer,max_decode_length)#max_decode_length = 40
encode_length = np.array(encode_length)
q_length = np.array(q_length)
decode_length = np.array(decode_length)

num_symbol = len(id_mapping)
#max_len = 1199#700##
import qaseq
model = qaseq.Model(d_max_length=max_len,q_max_length=q_max_len,a_max_length=27,num_symbol=num_symbol,rnn_size=64)
model.build_model()

batch_size = 256
left = len(answer)%batch_size
index_list = np.array(range(len(answer)-left))
nb_batch = len(index_list)/batch_size
print left

ROUGE = "/home/poyuwu/github/pythonrouge/pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl" #ROUGE-1.5.5.pl
data_path = "/home/poyuwu/github/pythonrouge/pythonrouge/RELEASE-1.5.5/data" #data folder in RELEASE-1.5.5
for epoch in range(1000):
    avg = 0.
    print model.sess.run(model.learning_rate)
    for num in range(nb_batch):
        #if np.random.rand() > 0.1 + 0.6*num/nb_batch:
        opti = model.opti                        
        loss = model.output_net['loss']          
        #else:                                        
        #    opti = model.sch                         
        #    loss = model.output_net['test_loss']     
        _, cost = model.sess.run([opti,loss],
                feed_dict={
                    model.input_net['d']:facts[num*batch_size:num*batch_size+batch_size],
                    model.input_net['d_mask']:encode_length[num*batch_size:num*batch_size+batch_size],
                    model.input_net['q']:question[num*batch_size:num*batch_size+batch_size],
                    model.input_net['q_mask']:q_length[num*batch_size:num*batch_size+batch_size],
                    model.input_net['a']:answer[num*batch_size:num*batch_size+batch_size],
                    model.input_net['a_mask']:decode_length[num*batch_size:num*batch_size+batch_size]})

        avg += cost
        sys.stdout.write(str(epoch)+"\t"+str(avg/(num+1))+"\r")
        sys.stdout.flush()
    print model.sess.run(model.learning_rate_decay_op)
    loss = model.sess.run(model.output_net['test_loss'],
                feed_dict={
                model.input_net['d']:facts[-left:],
                model.input_net['d_mask']:encode_length[-left:],
                model.input_net['q']:question[-left:],
                model.input_net['q_mask']:q_length[-left:],
                model.input_net['a']:answer[-left:],
                model.input_net['a_mask']:decode_length[-left:]})
    sys.stdout.write(str(epoch)+"\t"+str(avg/nb_batch)+"\t"+str(loss)+"\n")
    """
    R1 = 0.
    R2 = 0.
    R3 = 0.
    RSU4 = 0.
    RL = 0.
    bleu = 0.
    for i in range(1,54):
        peer = ""
        for j in range(26):
            if answer[len(answer)-i][j+1] == 1:
                break
            peer += id2word[answer[len(answer)-i][j+1]].decode("utf8") +" "
        pred = ""
        predictor = sess.run(predict,feed_dict={encode_input: [question[len(answer)-i]],decode_input: [[3]+[0]*26],encode_length_tensor: [q_length[len(answer)-i]]})
        for j in range(27):# position of word
            index = np.argmax(model[j][0])
            if  index == 1:
                break
            pred += id2word[index].encode('utf8')+" "
        score = pythonrouge.pythonrouge(peer,pred,ROUGE,data_path)
        R1 += score["ROUGE-1"]
        R2 += score["ROUGE-2"]
        R3 += score["ROUGE-3"]
        RSU4 += score["ROUGE-SU4"]
        RL += score["ROUGE-L"]
        sys.stdout.write("1: {0:.6f}\t2: {1:.6f}\t3: {2:.6f}\tSU4: {3:.6f}\tL: {4:.6f}\r".format(
        R1/i,R2/i,R3/i,RSU4/i,RL/i))
        try:
            bleu += float(nltk.translate.bleu_score.modified_precision([peer.split()],pred.split(),n=2))
        except:
            bleu += 0
        sys.stdout.write("{0}\r".format(bleu/i))
        sys.stdout.flush()
    print
    #print "\nepoch ",epoch,": of avg: ",avg#/batch_size
    """
