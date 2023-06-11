---
title: Bash 直接启动 sublime 或 atom 等编辑器以打开或新建文件
date: 2018-01-01
author: "Cong Chan"
tags: ['Software Engineer']
---

程序员或者其他需要码字多的人，经常要使用编辑器如sublime、atom 和 Typora等。如果每次都要用鼠标点击才能用sublime打开文件，或者在编辑器中新建文件，那么就会有点麻烦！但你可以用一句命令解决！

配置在Git Bash中用各种文本编辑器打开文件或者直接新建文件。这里以atom为例。

## 常规步骤
1. 打开Git Bash并`cd`到你的目标文件夹, 或者直接在目标文件中右键打开Git Bash.
2. `atom xxx.md` 就会在弹出的atom窗口中打开名为`xxx.md`的markdown文件, 如果没有这个文件, 会自动创建一个.
3. 适用于其他类型文件, 如`.java`等.
4. 如果想用sublime, 可以用`subl xxx.java`, 同理notepad++ 可以用 `notepad++ xxx.java`等。 (若出现错误,看下面)

## 若系统无法识别命令
一般使用sublime或者notepad++的用户, 可能会出现`error: 系统无法识别命令...`之类的, 可以这么解决:
### 方法1
新建一个文件命名为`subl`（注意不能有后缀名），内容：
```
#!/bin/sh
"D:\Sublime Text 3\sublime_text.exe" $1 &
```
第一行指明这是个 shell 脚本.
第二行的字符串是sublime的安装目录, 示例只是我电脑的目录, 注意这里要改为你自己的目录,
第二行的$1 是取的命令之后输入的参数
第二行的&是此命令在后台打开，这样sublime打开之后，就不会阻塞你的git bash

文件保存到 `C:\Program Files (x86)\Git\mingW32\bin` 目录下(你的git目录可能与我的不一样，注意改成你自己的)

同理适用于其他编辑器，比如用`chrome`打开`.html`文件等。如果不想每次都新建一个文件，可以用下面的方法2。

### 方法2
1. 找到 `C:\Users\你的计算机名`目录，如果你的计算机名是Administrator，那么你就要去`C:\Users\Administrator`目录下, 这里一般存放着windows系统的`我的文档, 桌面`等文件夹.
2. 在该目录下用Git Bash输入`notepad .bashrc`, 这会用windows记事本新建并打开一个文件`.bashrc`，这个文件没有名称只有后缀名。`.bashrc`里面可以给Git Bash设置命令的别名, 设置路径等。
3. 在.bashrc文件加入下面一行文本`alias notepad++="/D/Notepad++/notepad++.exe"`, 这里你需要修改为你电脑的安装路径。`alias`就是别名的意思，当我们执行`notepad++`的时候，实际执行的是`=`后面的语句.
4. 重新打开Git Bash, 设置才能生效，如果不想关掉在打开的话，可以直接在bash下输入`source ~/.bashrc`就可以立刻加载修改后的设置，设置立即生效。
现在在bash下输入`notepad++ test.py`, 就直接打开了notepad++并创建了这个叫test的Python文件。这里的别名不一定非要取`notepad++`，随你想叫什么都行。

同理也可以扩展到别的文本编辑器，`alias atom="atom的路径"`, `alias sublime="sublime的路径"`等. 最后还要注意一点，上面所说的路径最好不要有空格，括号等，否则会造成命令无效.

`.bashrc`还有很多有用的配置,可以根据需要进行扩展. 比如很多程序猿会选择修改删除命令`rm`(此命令不加任何参数的话，会直接删除文件, 可能会造成误删的后果)。这个时候可以给rm加个参数-i，意为在删除的时候给出提示。在文件`.bashrc`里添加这行代码`alias rm="rm -i"`。但这里不建议这么做，因为`rm="rm -i"`是一个定时炸弹，在使用它之后，习惯了之后, 你会本能地期望`rm`在删除文件之前会提示你。但是，总有一天你可能会用一个没有rm alias 别名的系统, 这时若你也直接随手一甩`rm`, 本以为会有提示, 结果发现数据真的被删除了。

在任何情况下，预防文件丢失或损坏的好方法就是进行备份。

所以如果你想个性化删除命令, 最好不要动`rm`，而是创建属于你的命令，比如`trash, myrm, delete`等, 用`alias trash='/bin/rm -irv'`会创建一条把文件放入垃圾回收站的命令.
