#!/usr/bin/env python
# coding: utf-8

# ### 用TensorFlow说Hello World!

# In[1]:


import tensorflow as tf

# 创建一个常值运算，将作为一个节点加入到默认计算图中
hello = tf.constant("Hello World!")

# 创建一个TF对话
sess = tf.Session()

# 运行并获得结果
print(sess.run(hello))


# 输出前面'b'表示Bytes literals(字节文字)

# ### 计算图(数据流图)
# TensorFlow程序的执行创建流图、执行对话两个部分。
#
# TensorFlow内部将上述计算过程表征为数据流图，也称为计算图。
#
# 创建流图（或计算图）就是建立计算模型，执行对话则是提供数据并获得结果。
#
# 计算图是一个有向图，由以下内容构成：
# * 一组节点，每个节点都代表一个操作，是一种运算
# * 一组有向边，每条边代表节点之间的关系（数据传递和控制依赖）
#
# TenforFlow有两种边：
#
# * 常规边（实线）：代表数据依赖关系。一个节点的运算输出成为另一个节点的输入，两个节之间有tensor流动（值传递)
# * 特殊边（虚线）：不携带值，表示两个节点之间的控制相关性。比如，happens-before关系，源节点必须在目的节点执行前完成执行

# In[2]:


# 一个简单计算图
node1 = tf.constant(3.0,tf.float32,name="node1")
node2 = tf.constant(4.0,tf.float32,name="node2")
node3 = tf.add(node1,node2)


# ### 计算图 -- png-1.png
# <img src= "png-1.png">

# In[3]:


print(node3)


# ### 创建流图（或计算图）就是建立计算模型，执行对话才能提供数据并获得结果

# In[4]:


# 建立对话并显示运行结果
sess = tf.Session()
print("运行sess.run(node1)的结果：",sess.run(node1))


# In[5]:


# 更新变量并返回计算结果
print("运行sess.run(node3)的结果：",sess.run(node3))
# 关闭session
sess.close()


# ### 张量的属性
#
# Tensor("add:0",shape=(2,),dtype=float32)
#
# 名字（name）
#
#     "node:src_output"：node 节点名称，src_output 来自节点的第几个输出
#
# 形状（shape）
#
#     张量的维度信息，shape=（），表示是标量
#
# 类型（type）
#
#     每一个张量会有一个唯一的类型。
#
#     TensorFlow会对参与运算的所有张量进行类型的检查，发现类型不匹配时会报错

# In[6]:


import tensorflow as tf
tens1 = tf.constant([[[1,2,3],[2,2,3]],
                     [[3,5,6],[5,4,3]],
                     [[7,0,1],[9,1,9]],
                     [[11,12,7],[1,3,14]]],name="tens1")
# 语句中包含[],{}或（）括号中间换行的就不需要使用多行连接符

print(tens1)


# In[7]:


import tensorflow as tf

scalar = tf.constant(100)
vector = tf.constant([1,2,3,4,5])
matrix = tf.constant([[1,2,3],[4,5,6]])
cube_matrix = tf.constant([[[1],[2],[3],[4],[5],[6],[7],[8],[9]]])

print(scalar.get_shape())
print(vector.get_shape())
print(matrix.get_shape())
print(cube_matrix.get_shape())


# ### 获取张量的元素
# 阶为1的张量等价于向量；
#
# 阶为2的张量等价于矩阵，通过t[i,j]获取元素；
#
# 阶为3的张量，通过t[i,j,k]获取元素；
#
# 例:

# In[8]:


import tensorflow as tf
tens1 = tf.constant([[[1,2],[2,3]],[[3,4],[5,6]]])

sess = tf.Session()
print(sess.run(tens1)[1,1,0])
sess.close()


# #### 下标从0开始

# ### 张量的类型
# * TensorFlow支持14种不同的类型
# * 实数 tf.float32, tf.float64
# * 整数 tf.int8, tf.int16, tf.int32, tf.int64, tf.uint8
# * 布尔 tf.bool
# * 复数 tf.complex64, tf.complex128
# * 默认类型：
# * 不带小数点的数会被默认为int32
# * 带小数点的会被默认为float32
#

# In[9]:


import tensorflow as tf
a = tf.constant([1.0,2.0],name="a")
b = tf.constant([2.0,3.0],name="b")
result = tf.add(a,b)


# ### 操作 Operation
# #### 计算图中的节点就是操作（Operation）。比如，一次加法是一个操作，一次乘法也是一个操作，构建一些变量的初始值也是一个操作。
# #### 每个运算操作都有属性，它在构建图的时候需要确定下来。操作可以和计算设备绑定，指定操作在某个设备上执行。
# #### 操作之间存在顺序关系，这些操作之间的依赖就是“边”。如果操作A的输入是操作B执行的结果，那么这个操作A就依赖于操作B

# In[10]:


import tensorflow as tf

# 本例用到了TensorBoard,具体使用后面讲解

tf.reset_default_graph() #清除default_graph和不断增加的节点

# 定义变量 a
a = tf.Variable(1,name="a")
# 定义操作b为a+1
b = tf.add(a,1,name="b")
# 定义操作c为b*4
c = tf.multiply(b,4,name="c")
# 定义d为c-b
d = tf.subtract(c,b,name="d")

# logdir改为机器上的合适路径
logdir = 'log'

# 生成一个写日志的writer,并将当前的TensorFlow计算图写入日志。
writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
writer.close()

# 执行tensorboard命令: tensorboard --logdir=log


# ### 计算图 -- png-2.png
# <img src= "png-2.png">

# # 会话的模式 1

# In[11]:


import tensorflow as tf

#定义计算图
tens1 = tf.constant([1,2,3])

#创建一个会话
sess = tf.Session()
try:
#使用这个创建好的会话来得到关心的运算的结果。比如可以调用 sess.run(result)
#来得到张量result的取值
    print(sess.run(tens1))
except:
    print("Exception!")
finally:
#关闭会话使得本次运行中使用到的资源可以被释放
    sess.close()


# ### 会话的模式2

# In[12]:


node1 = tf.constant(3.0, tf.float32, name="node1")
node2 = tf.constant(4.0, tf.float32, name="node2")
result = tf.add(node1, node2)

# 创建一个会话，并通过python中的上下文管理器来管理这个会话
with tf.Session() as sess:
    # 使用创建好的会话计算结果
    print(sess.run(result))
    
# 不需要再调用Session.close() 函数来关闭会话
# 当上下文退出时会话关闭和资源释放也自动完成


# ### 指定默认的会话
# TensorFlow不会自动生成默认会话，需要手动指定
#
# 当默认的会话被指定之后可以通过tf.Tensor.eval函数来计算一个张量取值

# In[13]:


node1 = tf.constant(3.0, tf.float32, name="node1")
node2 = tf.constant(4.0, tf.float32, name="node2")
result = tf.add(node1, node2)

sess = tf.Session()
with sess.as_default():
    print(result.eval())


# 下面代码也可以完成相同的功能

# In[14]:


sess = tf.Session()

#下面两个命令功能相同
print(sess.run(result))

print(result.eval(session=sess))


# ### 交互式环境下设置默认会话
# 在交互环境下，Python脚本或jupyter编辑器下，通过设置默认会话来获取张量的取值便加方便
#
# tf.InteractiveSession 使用这个函数会自动将生成的会话注册为默认会话

# In[15]:


node1 = tf.constant(3.0, tf.float32, name="node1")
node2 = tf.constant(4.0, tf.float32, name="node2")
result = tf.add(node1, node2)
sess = tf.InteractiveSession()
print(result.eval())
sess.close()


# ### 常量
# 在运行过程中值不会改变的单元，在Tensorflow中无须进行初始化操作
#
# ### 创建语句：
#
#     constant_name = tf.constant(value)

# In[16]:


a = tf.constant(1.0,name='a')
b = tf.constant(2.5,name='b')
c = tf.add(a,b,name='c')

sess = tf.Session()
c_value = sess.run(c)
print(c_value)
sess.close()


# ### 变量 Variable
# #### 在运行过程中值不会改变的单元，在Tensorflow中必须进行初始化操作
# - 创建语句：name_variable = tf.Variable(value,name)
# - 个别变量初始化：init_op = name_variable.initializer()
# - 所有变量初始化：init_op = tf.global_variables_initializer()

# In[17]:


node1 = tf.Variable(3.0,tf.float32,name="node1")
node2 = tf.Variable(4.0,tf.float32,name="node2")
result = tf.add(node1,node2,name="add")

sess = tf.Session()

#变量初始化
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(result))


# 以上代码在Session对话变量后，增加了一个init初始化变量，并调用会话的run命令对参数进行始初化

# # 变量赋值
# - 与传统编程语言不同，TensorFlow中的变量定义后，一般无需人工赋值，系统会根据算法模型训练优化过程中自动调整变量对应的数值
# - 后面在将机器学习模型训练时会更能体会，比如权重Weight变量w,经过多次迭代，会自动调 epoch = tf.Variable(0,name='epoch',trainable=False)
# - 特殊情况需要人工更新的，可用变量赋值语句
# 变量更新语句:update_op = tf.assign(variable_to_be_updated,new_value)

# In[18]:


# 通过变量赋值输出1、2、3...10

import tensorflow as tf

tf.reset_default_graph()  #清除和不断增加的节点

value = tf.Variable(0,name="value")
one = tf.constant(1)
new_value = tf.add(value,one)
update_value = tf.assign(value,new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(10):
        sess.run(update_value)
        print(sess.run(value))
        
#logdir改为机器上的合适路径
logdir = 'log'

#生成一个写日志的writer,并将当前的TensorFlow计算图写入日志。
writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
writer.close()

#执行tensorboard命令: tensorboard --logdir=log


# ### 思考题：如何通过变量赋值计算：1+2+3+...+10 ?
# ### 计算图 -- png-3.png
# <img src= "png-3.png">

# ### 占位符 placeholder
# TensorFlow占位符Placeholder,先定义一种数据，其参数为数据的Type和Shape
#
#     占位符Placeholder的函数接口如下：tf.placeholder(dtype, shape=None, name=None)
#
# ### Feed提交数据
# 如果构建一个包含placeholder操作的计算图，当在session中调用run方法时，placeholder占用的变量必须通过feed_dict参数传递进去，否则报错

# In[19]:


import tensorflow as tf

a = tf.placeholder(tf.float32,name='a')
b = tf.placeholder(tf.float32,name='b')
c = tf.multiply(a,b,name='c')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 通过feed_dict的参数传值，按字典格式
    result = sess.run(c,feed_dict={a:10.0,b:3.5})
    
    print(result)


# 多个操作可以通过一次Feed完成执行

# In[20]:


import tensorflow as tf

a = tf.placeholder(tf.float32,name='a')
b = tf.placeholder(tf.float32,name='b')
c = tf.multiply(a,b,name='c')
d = tf.subtract(a,b,name='d')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    result = sess.run([c,d],feed_dict={a:[8.0,2.0,3.5],b:[1.5,2.0,4]})
    
    print(result)
    #取结果中的第一个
    print(result[0])


# #### 一次返回多个值分别赋给多个变量

# In[21]:


import tensorflow as tf

a = tf.placeholder(tf.float32,name='a')
b = tf.placeholder(tf.float32,name='b')
c = tf.multiply(a,b,name='c')
d = tf.subtract(a,b,name='d')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    #返回的两个值分别赋给两个变量
    rc,rd = sess.run([c,d],feed_dict={a:[8.0,2.0,3.5],b:[1.5,2.0,4]})
    
    print("value of c=",rc, "value of d=",rd)


# ### TensorBoard 可视化
# #### 在TensorBoard中查看计算图结构

# In[22]:


import tensorflow as tf

#清除和不断增加的节点
tf.reset_default_graph()

#logdir改为机器上的合适路径
logdir = 'log'

#定义一个简单的计算图，实现向量加法的操作
input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")

#生成一个写日志的writer,并将当前的TensorFlow计算图写入日志。
writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
writer.close()
#执行tensorboard命令: tensorboard --logdir=log


# ### 计算图 -- png-4.png
# <img src= "png-4.png">
