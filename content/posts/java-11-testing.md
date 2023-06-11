---
title: Java 11 | 测试 Testing
date: 2017-01-29
author: "Cong Chan"
tags: ['Software Engineer', 'Java']
---
## 测试
如何知道自己的程序是否真的在工作？在现实世界中，程序员相信他们的代码，因为代码通过了他们自己编写的测试。常用的测试有 Ad Hoc Testing， Unit test 和 Integration Testing。

Ad Hoc Testing，是指没有计划和记录的软件测试，除非发现缺陷，不然一般只运行一次。

### Unit test
程序可分解为单元（或程序中可测试的最小部分），Unit test 严格测试代码的每个单元，最终确保项目正确运行。
Unit test 好处：
1. Unit test 保证良好的代码结构（每个 method “只打一份工”），帮助我们较好地解析任务，
2. 允许我们考虑每个方法的所有边界情况，并单独测试它们。
3. 让我们每次只专注于一个单元，进行测试，debug，对准确度有信心后，再进行下一个单元的开发。相比于一次性写完所有代码，再测试debug，Unit test 减少了 debugging 时间。
<!-- more -->
坏处：
1. 测试也要花时间
2. 测试本身也是有可能出错的，测试可能不全面，不规范，或者有bug
3. 有些单元是依赖于其他单元的
4. Unit testing 无法保证各个模块的交互，无法保证整个系统作为一个整体是否正常工作。

### JUnit
JUnit是一个给Java做测试的框架，由Erich Gamma（Design Patterns）和Kent Beck（eXtreme Programming）编写。
JUnit使用Java的 reflection 功能（Java程序可以检查自己的代码）和注释。
JUnit允许我们：
- 定义并执行测试和测试套件
- 使用测试作为规范的有效手段
- 使用测试来支持重构
- 将修改的代码集成到构建中
JUnit可用于多个IDE，例如BlueJ，JBuilder和Eclipse在一定程度上具有JUnit集成。

```java
import org.junit.Test;
import static org.junit.Assert.*;

@Test
public void testMethod() {
    assertEquals(<expected>, <actual>);
}
```
`assertEquals`测试一个变量的实际值是否等于它的期望值。
JUnit test 各个测试方法，必须是非静态的（JUnit的设计人员设计规定的）。

JUnit的术语
* Test runner：测试运行器， 运行测试和报告结果的软件。实现方式：集成到IDE中，独立GUI，命令行等
* Test suite：测试套件是一组测试用例。
* Test case：测试用例用于测试单个方法对特定输入集的响应。
* Unit test：单元测试的单元，是代码中我们能够相对合理地测试的最小的元素，通常是单个类。

常用的JUnit接口和方法
`@Before`: Creates a test fixture by creating and initialising objects and values.

`@After`: Releases any system resources used by the test fixture. Java usually does this for free, but files, network connections etc. might not get tidied up automatically.

`@Test`：tests cases.

`static void assertTrue(boolean test)`, `static void assertTrue(String message, boolean test)`, `static void assertFalse(boolean test)`, `static void assertFalse(String message, boolean test)`

### Integration Testing
鉴于 Unit testing 无法保证，有交互的多个模块，作为一个整体是否正常工作。
我们可能需要 integration testing，把各个模块合并，作为一个组合，进行测试（也可以把 Unit test 组合起来变成 integration testing）。

Integration testing 一般都比较麻烦，也不容易自动化，而且一般是在比较高的抽象层进行测试，可能会漏掉微小的错误。

当把所有模块都作为一个整体，也就是整个系统作为测试对象时，就是 system testing。

### Test driven development
TDD开发步骤：
1. 明确一项新功能需求。
2. 为该功能编写 Unit test。
3. 运行测试，<font color="red">按理应该无法通过测试</font>（因为还没写功能程序）。
4. 编写通过实现该功能的代码，<font color="green">通过测试</font>。
5. 可选：重构代码，使其更快，更整洁等等。

![source from http://ryantablada.com/post/red-green-refactor---a-tdd-fairytale ](http://www.pathfindersolns.com/wp-content/uploads/2012/05/red-green-refactorFINAL2.png "image from: http://ryantablada.com/post/red-green-refactor---a-tdd-fairytale")
