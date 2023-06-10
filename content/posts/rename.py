import os
import re
import sys

currentpath = os.getcwd()		#得到进程当前工作目录
fileList = os.listdir(currentpath)		#待修改文件夹
print("修改前：")
for filename in fileList:
    print(filename)		#输出文件夹中包含的文件
#os.chdir(r"C:\Users\Administrator\Desktop\stars")		#将当前工作目录修改为待修改文件夹的位置
for fileName in fileList:		#遍历文件夹中所有文件
	#pat="NOTE-.*)"		#匹配文件名正则表达式
	#pattern = re.findall(pat,fileName)		#进行匹配
	os.rename(fileName, (fileName.replace("NOTE-", "", 1)))		#文件重新命名

print("---------------------------------------------------")
#os.chdir(currentpath)		#改回程序运行前的工作目录
sys.stdin.flush()		#刷新
print("修改后：")
fileList = os.listdir(currentpath)
for filename in fileList:
    print(filename)		#输出修改后文件夹中包含的文件
