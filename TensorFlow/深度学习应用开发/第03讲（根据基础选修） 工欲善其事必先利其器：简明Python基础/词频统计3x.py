#coding:utf-8
def getText():
    txt = open("THE TRAGEDY OF ROMEO AND JULIET.txt", "r", encoding='unicode_escape').read()
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
    print ("({},{})".format(count, word))

