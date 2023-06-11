---
title: Java 多线程
date: 2017-02-26
author: "Cong Chan"
tags: ['Java']
---
## 多线程
线程是独立的执行空间，Java语言内置多线程功能，用类`Thread`来表达。每个Java应用程序会启动一个主线程 -- 将`main()`放在自己的执行空间的最开始处. JVM会负责主线程的启动(以及其他比如GC的系统线程). 程序员负责启动自己的建立的线程.
<!-- more -->

启动新线程
1. 建立`Runnable`对象作为线程任务`Runnable job = new MyRunnable()`, `Runnable`接口只有一个方法`run()`
2. 建立Thread对象并赋值Runnable `Thread thread1 = new Thread(job); Thread thread2 = new Thread(job)`
3. 启动`thread.start();`

另一种创建线程的方法是用Thread的子类覆盖掉`run()`, 构造新线程`Thread t = new Thread();`. 从OO的角度看待, 此时Thread 与线程任务是不同概念的. 让子类继承Thread的目的通常是需要更特殊的Thread, 需要特殊的行为, 如果没有这种需求, 就没必要继承Thread.

### 线程调度器
多线程间的切换由调度器scheduler来管理, 线程有多种状态:
1. 执行中,
2. `sleep(2000)`: 睡眠2000ms, 时间到之前不会被执行, 但时间到了并不保证一定会被执行. 可能会抛出InterruptedException, 所以对它的调用要包含在try/catch中.
3. locked.

线程的`run`完成执行后, 将无法重新启动.

调度器在不同的JVM有不同的做法. 测试多线程时需要在不同机器上测试.

### 并发
Concurrency并发环境中, 为了避免冲突, 需要上锁, 使用`synchronized`来修饰方法使之每次只能被单一线程读写. 同步化是有代价的, 查询钥匙有性能上的损耗, 同步化也会强制线程排队执行, 还可能出现死锁.

### 死锁
因为两个线程互相持有对方正在等待的东西, 导致没有一方可以脱离等待. 数据库有事务回滚机制来复原死锁的事务, 但Java没有处理死锁的机制.

### Volatile
Java为了提高程序运行效率, 编译器自动会优化, 把经常被访问的变量混存起来, 程序在读取这个变量时有可能会直接从缓存(例如寄存器)中读取这个值, 而不会去内存中读取. 但在多线程环境中, 变量的值可能因为别的线程而改变了, 而该缓存的值不会相应改变, 从而造成应用程序读取的值和实际的变量值不一致.

使用`volatile`修饰被不同线程访问和修改的变量, 使得其每次被用到时, 都是直接从对应的内存中提取, 而不会利用缓存了.
```java
public class MyThread implements Runnable {
    private volatile Boolean flag;
    public void stop() { flag = false; }
    public void run() { while(flag) ; }
}
```
如果`flag`没有被声明为`volatile`, 那么当这个县城的`run()`判断`flag`时, 使用的有可能是缓存中的值, 此时就不能及时地获取其他线程对flag所做的操作了.

但`volatile`不能保证操作的原子性, 因此一般情况下不能代替`sychronized`. 此外, 使用`volatile`会阻止编译器对代码的优化, 因此会降低
