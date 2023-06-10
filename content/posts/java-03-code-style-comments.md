title: Java 03 | 代码风格 注释 Javadoc
date: 2016-12-21
categories:
- CS
tags:
- Java
---
## 代码风格与注释
努力保持代码可读性。良好的编码风格的一些最重要的特点是：
* 一致的风格（间距，变量命名，缩进风格等）
* 大小（线不太宽，源文件不要太长）
* 描述性命名（变量，函数，类），例如变量或函数名称为年份或getUserName而不是x或f。让代码本身提供可解读性。
* 避免重复的代码：若有两个重要的代码块及其相似，应该想办法合并。
* 适当的评论, 使其他读者也能轻松理解你的代码
    * 行注释: `//`分隔符开头行被当做注释。
    * Block（又名多行注释）注释: `/*`,  `*/ `, 但我们更推荐javadoc形式的注释。

### Javadoc
Javadoc: `/ **`，`*/`, 可以（但不总是）包含描述性标签。 借助[javadoc工具](http://docs.oracle.com/javase/8/docs/technotes/tools/windows/javadoc.html)可以生成HTML格式的API文档。
第一段是方法的描述。描述下面是不同的[描述性标签](https://en.wikipedia.org/wiki/Javadoc), 比如参数 `@param`， 返回值 `@return`， 可能抛出的任何异常 `@throws`
```java
/**
 * @author   名字，邮箱<address @ example.com>
 * @version     1.6 版本
 * @param
 * @return
 */
public class Test {
    // class body
}
```
