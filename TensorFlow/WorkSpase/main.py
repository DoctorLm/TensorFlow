#coding:utf-8
#import matplotlib.pyplot as plt
#import os
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#允许使用TF2执行模式
#import tensorflow.compat.v2 as tf
#tf.enable_v2_behavior()
# 禁用TF2执行模式
#import tensorflow.v1 as tf
#tf.disable_eager_execution()
#import tensorflow as tf
#print("TensorFlow版本:", tf.__version__)

import tkinter.filedialog
import tkinter


def selectPath():
    path_ =tkinter.filedialog.askopenfilename()
    path.set(path_)

def getuser():
    user=roadEntry.get()
    return user

def showPicture():
    imgpath = getuser()

def showPicture():
    imgpath = getuser()
    global photopath
    photopath = 'ins.gif'
    photo = plt.imread(imgpath)
    photo = transform.resize(photo, (300, 580), mode='constant')
    photo=plt.imsave(photopath, photo)
    img_open = Image.open(photopath)
    photo = ImageTk.PhotoImage(img_open)
    label.config(image=photo,compound = tkinter.CENTER)
    label.image=photo

root = tkinter.Tk()
root.geometry('940x640')
root.wm_title('毕业设计识别演示')
root.config(bg='#8080c0')

roadLabel = tkinter.Label(root, font=('黑体', 18), text='文件路径：',width=10,bg='#8080c0')
roadLabel.grid(row=7, column=0)

Selectbutton = tkinter.Button(root, text="Select√", font=('黑体', '16'),bg='#ffa042',height=1, command=selectPath)
Selectbutton.grid(row=7, column=2)

path = tkinter.StringVar()
roadEntry = tkinter.Entry(root,font=('黑体','18') ,textvariable=path, width=47)
roadEntry.grid(row=7, column=1)


label = tkinter.Label(root,bg='#8080c0')
label.grid(row=3, column=1)
OkButton = tkinter.Button(root,text='确认并显示',relief='raise',font=('楷体','14'),command=showPicture,bg='#ffa042').grid(row=6,column=1)


root.mainloop()

