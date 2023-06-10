title: Java Exceptions
date: 2017-05-29
categories:
- CS
tags:
- Java
---
## Exception-handling
假如调用了一个不是自己写的方法, 该方法执行某些有风险的任务(当然,自己写的也可能会有风险),可能会在运行期间出状况,那么就必须认识到该方法是有风险的, 要写出可以在发生状况时做出应对的代码.

当程序出现错误时，假如继续运行下去已经没有意义（或者根本不可能继续），那么我们就想要中断正常的控制流程 - throws an exception。
<!-- more -->

### Throwing Exceptions
比如当想从某`ArrayMap`中提取某个不存在的键值时, java自动抛出一个`implicit exception`
```java
$ java ExceptionDemo
Exception in thread "main" java.lang.ArrayIndexOutOfBoundsException: -1
at ArrayMap.get(ArrayMap.java:38)
at ExceptionDemo.main(ExceptionDemo.java:6)
```
如果想让自己的程序抛出更详细的信息, 可以在程序中加入`explicit exception`
```java
public V get(K key) {
    intlocation = findKey(key);
    if(location < 0) {
        throw newIllegalArgumentException("Key " + key + " does not exist in map."\);
    }
    return values[findKey(key)];
}
```
```java
$java ExceptionDemo
Exception in thread "main" java.lang.IllegalArgumentException: Key yolp does not exist in map.
at ArrayMap.get(ArrayMap.java:40)
at ExceptionDemo.main(ExceptionDemo.java:6)
```

### Catch Exceptions
单纯 throw exception 会导致代码崩溃。但是通过 `try - catch` “捕捉”异常(`RuntimeException` 是 Java object), 可以防止程序崩溃。

比如通过捕捉异常, 来引入修正措施: 下面这个代码通过
```java
/** 当狗在生气时, 如果尝试拍拍它，会抛出一个 RuntimeException，
捕捉到 exception 后, 用香蕉来抚慰它. */
Dog d = new Dog("Lucy", "Retriever", 80);
d.becomeAngry();

try { // 把有风险的代码块放在try块
    d.receivePat();
} catch (Exception e) {
    System.out.println(
    "Tried to pat: " + e);
    d.eatTreat("banana");
}
d.receivePat();
System.out.println(d);
```
```java
$ java ExceptionDemo
Tried to pat: java.lang.RuntimeException: grrr... snarl snarl
Lucy is a displeased Retriever weighing 80.0 standard lb units.
```
```java
$ java ExceptionDemo
Tried to pat: java.lang.RuntimeException: grrr... snarl snarl
Lucy munches the banana

Lucy enjoys the pat.

Lucy is a happy Retriever weighing 80.0 standard lb units.
```

Exception 继承自 `Throwable`, 异常也是多态的。
```
public class Throwable extends Object implements Serializable
```
> The Throwable class is the superclass of all errors and exceptions in the Java language. Direct Known Subclasses: Error, Exception

如果使用`if else`来管理异常会让代码变得很乱而难以阅读. 而使用`try catch`可以为每种类型的 exception 提供不同的应对。使代码像清晰的记述文般铺展开来: 首先，尝试执行所需的操作。然后，捕捉任何错误。
```java
public class Laundry {
    public void doLaundry() throws PantsException, LingerieException {}
}

public class Foo {
    public void go() {
        Laundry laundry = new Laundry();
        try {
            laundry.doLaundry();
        } catch (PantsException pex) {
            // doSomething
        } catch (LingerieException lex) {
            //doSomething
        }
    }
}
```
这种清晰度使代码的长期维护变得更容易。
以异常的父类来声明会抛出的异常, 这样就不必声明每个子类异常了, 在`catch`块中只需要捕捉异常的父类.
```java
public void doLaundry() throws ClothingException {}
try {
    laundry.doLaundry();
} catch (ClothingException cex) {
}
```
但如果需要对每个不同的子类异常采取不同的措施, 那么还是需要声明各个子类异常, 并分别捕捉.
```java
try {
    laundry.doLaundry();
} catch (LingerieException lex) {
    //doSomething
} catch (ClothingException cex) {
    //doSomething
} catch (Exception ex) {
    //doSomething
}
```
此时多个`catch`块要从小排到大(在继承树上看, 就是先捕捉子类), **不能把父类异常放在子类上面,否则无法通过编译**. 因为JVM只会从上开始往下找第一个符合范围的catch块, 如果第一个catch就是`catch (Exception ex)`, 那么剩下的捕捉都没用了.

### Finally
在`try-catch`后面, 使用`finally`指示无论如何都要执行的部分(不管有没有异常).
```java
try {
    turnOvenOn();
    x.bake();
} catch (BakingException ex) {
    ex.printStackTrace();
} finally { // 不管怎样最后都要关火!
    turnOvenOff();
}
```
* 如果`try`块失败了, 抛出异常, 流程马上转移到`catch`块. 当`catch`块完成时, 继续执行`finally`块. 当`finally`块完成时, 继续执行其他代码
* 如果`try`块成功, 流程会跳过`catch`块并移动到`finally`. `finally`块完成时, 继续执行其他代码
* **即使try或者catch块中有return指令，finally还是会执行！**， 流程会先暂时保存`return`的值，去执行`finally`, 然后再`return`.
* 但是, **如果finally中也有return指令，那么会直接执行该return指令，结束整个流程**，这意味着`try/catch`中的`return`得不到执行, 且它的值会丢失.

### 如果不想处理异常
有些情况下, 比如某个异常是在你的程序调用的其他程序中抛出的, 你可以选择不处理这个异常, 把它`duck`掉, 让那些调用你方法的程序来处理这个异常. 也就说你的程序只是把异常转个手, 给个出路.
```java
// 只有抛出, 没有try/catch 异常
public void foo() throws ReallyBadException {}
```
方法抛出异常时, 方法会从栈上立即被取出, 而异常会再度丢给栈上的方法, 也就说下一个调用方, 这种过程可以一直循环下去. 但ducking只是在击鼓传花, 最后总得有个方法接盘, 如果连`main()`也duck掉, 那么就是`Uncaught Exceptions Stack Trace`, 异常到达堆栈底部后仍未被捕获，JVM崩溃，Java 打印出堆栈的跟踪:
```java
java.lang.RuntimeException in thread “main”:
at ArrayRingBuffer.peek:63
at GuitarString.sample:48
at GuitarHeroLite.java:110
```
![](https://joshhug.gitbooks.io/hug61b/content/assets/callstack.png "image from: https://joshhug.gitbooks.io/hug61b")
程序猿可以据此追踪错误路径。

### Checked vs Unchecked Exceptions
有时候，某些抛出的 exception 无法通过编译，可以理解为这些异常在编译器看来是非常恶心的存在，需要程序猿必须给这些 exception 提供明确的应对处理方案 - 这种叫 checked exception （"must be checked"）。这种异常是以人无法预测或防止的方法出现的执行期失败状况, 比如我们无法保证文件一直都在, 无法保证服务器不会死机.
```java
public class Eagle {
    public static void gulgate() {
        if (today == “Thursday”) {
            throw new IOException("hi"); }
    }
}
```
```java
$ javac Eagle
Eagle.java:4: error: unreported exception IOException; must be caught or declared to be thrown
throw new IOException("hi"); }
^
```
很明显，Java对此`IOException`并不满意, 因为`IOExceptions`是 checked exception, 而这里没有提供应对处理方案。但假如换做`RuntimeException`就可以编译通过 (虽然在 runtime 时会崩溃).![](https://joshhug.gitbooks.io/hug61b/content/assets/checked_exceptions.png "image from: https://joshhug.gitbooks.io/hug61b/") `InterruptException`也是要检查的异常. 大多数`check exception`都有修正的可能性。例如遇到`FileNotFound`，可以考虑要求用户重新指定他们想要的文件 (可能是因为错误输入导致的)。

`Errors` 和 `Runtime Exceptions`, 以及它们的子类都是`unchecked`. 大部分`RuntimeException`都是因为程序逻辑的问题. 虽然`check exception`是人力无法保证, 但我们可以确保程序的逻辑不出错, 例如对只有长度N的数组中取第`N+1`个元素, 这种是逻辑错误了, 不存在什么补救, 而是应该在写代码时就要避免。`try/catch`是用来处理真正的异常, 而不是程序的逻辑错误.

Java在尽最大努力确保每个程序运行时不会崩溃，所以它不会允许程序留下任何明明可以应对修正却没有被明确地修正的错误。

两种方法来处理 checked error:
1. **Catch**
```java
public static void gulgate() {
    try {
        if (today == “Thursday”) {
            throw new IOException("hi");
        }
    } catch (Exception e) {
        System.out.println("psych!");
    }
}
```
    假如能够应对，尽量用 catch 锁定异常防止其逃逸。
2. **Specify**: 如果实在不想在该方法中处理这种异常，可以将责任推迟到别的地方。我们可以指定该方法是危险的
```java
public static void gulgate() throws IOException {
    ... throw new IOException("hi"); ...
}
```
    然后任何其他调用`gulgate()`的方法也变成危险的了, 它们也需要被处理(同样使用两种方法之一)
    ```java
    // catch
    public static void main(String[] args) {
        try {
            gulgate();
        } catch(IOException e) {
            System.out.println("Averted!");
        }
    }
    // 或 specify
    public static void main(String[] args) throws IOException {
        gulgate();
    }
    ```
    需要明确异常处理责任人。同时确保调用者知道该方法是危险的！

### 异常处理规则
1. catch和finally不能没有try
2. try块和catch块之间不能有其他代码
3. try一定要有catch或finally
4. 只带有finally而没有catch的try必须要声明异常, 也就是明确抛出`void go() throws FooException {try {} finally {} }`
