---
title: Java 格式
date: 2017-05-29
author: "Cong Chan"
tags: ['Java']
---
## Java格式化指令"
跟在`%`后面的都是格式化指令.
`% [argument number] [flags] [width] [.precision] type`
<!-- more -->
`[]`内都是选择性的参数, 且必须按照顺序.
1. `argument number` 当要格式化的参数超过一个, 这里指定是哪一个
2. `flags` 特定类型的特定选项, 如数字是否要加逗号或正负号
3. `width` 最小的字符数. 但不是总数, 输出可以超过此宽度, 若不足则会主动补零
4. `.precision` 精确度
5. `type` 类型标识
    * `d` decimal, 十进制整数
    * `f` 浮点数
    * `x` hexadecimal, 16进制
    * `c` character

如`format("%, 6.1f", 33.000);`

如果有多个待输出的参数, 可以把新的参数加到后面, 并对应两个不同的格式化设定, 也就是两个`%`格式指令 `String s = String.format("%,.2f out of %,d", 999, 1000)`

## 可变参数列表
可以看到格式化的参数似乎可以不断添加，如果用重载来实现会显得不现实。为了应对格式化的API，Java支持可变参数列表 varable argument list (vararg).

## 日期
日期格式化的类型是用`t`开头的两个字符表示
* `%tc` 完整的日期与时间 `Sun Nov 03 14:52:41 2018`
* `%tr` 只有时间 `03:01:47 PM`

周月日
`Sunday, November 03`, 通过组合而来
```java
Date today = new Date();
String.format('%tA, %tB %td', today, today, today);
```
如果不想重复给参数
```java
String.format('%tA, %<tB %<td', today);
```
`<`符号指示格式化程序重复利用参数

但除了要去的当前日期用到`Date`之外, 其余的时间功能几乎都在`Calendar`上面.
因`Calendar`是抽象类, 所以不能取得它的实例, 但可调用它的静态方法, 对`Calendar`静态方法的调用以取得一个具体子类的实例, `Calendar cal = Calendar.getInstance();` 
