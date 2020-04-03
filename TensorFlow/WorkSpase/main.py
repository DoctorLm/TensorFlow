#coding:utf-8
def getText():
    txt = open("TensorFlow/深度学习应用开发/第03讲（根据基础选修） 工欲善其事必先利其器：简明Python基础/Data/THE TRAGEDY OF ROMEO AND JULIET.txt", "r").read()
    txt = txt.lower()
    for ch in '!"#$%&()*+,-./:;<=>?@[\\]^_‘{|}~':
        txt = txt.replace(ch, " ")
    return txt
rjTxt = getText()
words  = rjTxt.split()
counts = {}
for word in words:
    counts[word] = counts.get(word,0) + 1
items = list(counts.items())
items.sort(key=lambda x:x[1], reverse=True)
for i in range(10):
    word, count = items[i]
    print "({},{})".format(count, word)
