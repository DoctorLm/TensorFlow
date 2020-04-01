#!/usr/bin/env python
# coding: utf-8

# 简明Python基础
# 1 简单 print 用法
# print 函数 - 在终端中输出，Python3.x需要加（），Python2.x没有（）
# Python中没有强制的语句终止字符
print("Hello world!")

# print默认输出是换行的
print("Hello")
print("world")

# 如果要实现print输出不换行，需要指定结尾符end=''
print("Hello", end=' ')
print("world!")
print("Hello","world!")

# 量与基本数据类型
# 每个变量在内存中创建，都包括变量的标识，名称和数据这些信息
# 每个变量在使用前都必须赋值
# 赋值号是“=”
int_var = 3
float_var = 3.1415926
str_var = "Hello"
print(int_var, float_var, str_var)

# Python 中的变量赋值不需要数据类型声明，数据类型根据具体的赋值确定
print(int_var, type(int_var))
print(float_var, type(float_var))
print(str_var, type(str_var))
