title: Java 01 | 安装
date: 2016-12-18
categories:
- CS
tags:
- Java
- cs61b
---
## Hello World
参考了伯克利 Josh Hug 的 [cs61b spring 2017](datastructur.es/sp17/) 和 [cs61b spring 2018](http://sp18.datastructur.es). Lab, homework 和 project 代码实现参考 https://github.com/ShootingSpace/cs61b-data-structures.
## Java安装与配置
安装Java，前往[Oracle](http://www.oracle.com/technetwork/java/javase/downloads/index.html)下载java sdk，我用的是Java SE 8u151/ 8u152 版本。安装sdk时会同时安装sdr。

Windows系统配置:
* 推荐安装[git bash](http://git-scm.com/download/), 一切按照默认安装就好.
* 更新系统环境变量: 直接在`运行`中搜索`Environment Variables`, 选择`编辑系统环境变量`, 在弹出的框中选择`高级->环境变量`, 在弹出的框中`系统变量`里面
    * 新建变量: 变量名 = `JAVA_HOME`, 变量值 = 你的jdk路径,如`C:\Program Files\Java\jdk1.8.0_151`
    * 编辑Path: 在前面加入`%JAVA_HOME%\bin;%PYTHON_HOME%;`(请注意，不能有空格.)

OS X系统配置:
* 安装Homebrew，一个非常好用的包管理工具。要安装，请在terminal终端输入`ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"`(注意：在此过程中，可能会提示输入密码。当输入密码时，终端上不会显示任何内容，但计算机还是会记录你的密码的。这是一个安全措施, 让其他人在屏幕上看不到你的密码。只需输入您的密码，然后按回车。)
* 然后，通过输入以下命令来检查brew系统是否正常工作`brew doctor`. 如果遇到警告，要求下载命令行工具，则需要执行此操作。请参考这个[StackOverflow](http://stackoverflow.com/questions/9329243/xcode-4-4-and-later-install-%20%20command-line-tools)。
* 安装git：输入`brew install git`

安装并配置好java后，测试是否成功:
随便在你喜欢的文件夹里新建一个java文件`HelloWorld.java`
```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello world!");
    }
}
```
你可以选择用sublime来快速新建文件, 直接在你选择的文件里右键 git bash, 在git bash 里面键入`subl HelloWorld.java`, 还自动启动sublime并新建一个空白的`HelloWorld.java`文件, 把上面的代码复制进去并保存即可. (若出现类似提示: 找不到subl command, 解决办法请参考博文[在Gitbash中直接启动sublime或atom等编辑器以打开或新建文件](/Launch-editor-in-Gitbash) )
开始真正的测试。直接在之前打开的git bash中输入:
1. `ls`, 会看到`HelloWorld.java`这个文件, `ls`会列出这个目录中的文件/文件夹
2. `javac HelloWorld.java`, 理论上这一步不会有任何输出，有的话可能是设置有问题。现在，如果你继续`ls`，会看到多了一个`HelloWorld.class`文件， 这是javac创建的。
3. `java HelloWorld` (注意没有`.java`), 会看到输出`Hello World`, 表明你的Java设置没有问题
