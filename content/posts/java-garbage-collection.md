---
title: Java 垃圾回收机制
date: 2017-06-19
author: "Cong Chan"
tags: ['Java']
---
在Java中，JVM负责内存动态分配和垃圾回收的问题。Java的对象必须要有引用才能被使用，也就是说如果要操作对象，必须通过引用来进行。如果一个对象唯一的引用变量死了（随着堆栈块一起解散），对象就会被认定为可被垃圾回收（Garbage Collection）的。没有被引用的对象，是没有存在意义的，因为没有人知道它的地址，无法调用它，它的存在只会浪费空间。
<!-- more -->
目前内存的动态分配和内存回收技术已经相当成熟，但还是需要了解GC和内存分配，这样当需要排查各种内存溢出、内存泄漏问题时，当GC成为系统达到更高并发量的瓶颈时，需要对这些自动化的技术实施必要的监控和调节。![](/images/jvm.png)

### 判断对象是否可回收
最简单的方法是通过引用计数（reference counting）来判断一个对象是否可以被回收。不失一般性，如果一个对象没有任何引用与之关联，那么这个对象就被判定为可被回收的对象了。这种方式实现简单，而且效率较高，但是它无法解决对象间循环引用的问题，因此在Java中并没有采用这种方式（Python采用的是引用计数法）。

主流的商用程序语言（Java，C#，Lisp）的主流实现中，使用可达性分析（reachability analysis）来判定对象是否存活。
* 使用一些列GC Root对象作为起始点，从这些节点开始往下沿着引用链搜索，如果GC Root到某个对象无法通过任何引用链项链，则该对象会被标记一次, 并且进行一次筛选, 筛选的条件是此对象是否有必要执行`finalize()`方法.
* 当对象没有覆盖`finalize()`方法, 或者该方法已经被JVM调用过, JVM都会视之为没有必要执行`finalize()`。
* 如果该对象被判定为有必要执行`finalize()`方法, 那么这个对象会被放置在F-Queue队列中, 并在稍后由一个由JVM自动建立的, 低优先级的Finalizer线程去执行. 执行只是触发该方法, 但不会等待它结束, 因为可能会有执行缓慢或者死循环等特殊情况
* 稍后GC会动F-Queue里的对象进行第二次标记, 如果对象要在`finalize()`中避免被消灭, 只需要重新与引用链上的任何一个对象建立关联即可(`finalize()`的优先级较低), 这样第二次标记时它将不会被考虑. 否则就只能被回收.

要注意, 任何一个对象的`finalize()`只会被系统自动调用一次, 下次再GC时不会再执行, 也就是只有一次自救机会.

在Java中，可作为GC Roots的对象包括：
* 虚拟机栈（栈帧中的本地变量表）中引用的对象。
* 方法区中类静态属性引用的对象
* 方法区中常量引用的对象
* 本地方法栈中native方法引用的对象

### 引用
引用分为强引用，软引用，弱引用，虚引用。根据不同引用，有不同的GC回收策略。

强引用：类似`Object obj = new Object();`这种, 垃圾回收器永远不会回收他们

软引用：非必须引用，内存溢出之前进行回收. 如果这次回收后内存还不足,才会抛出内存溢出异常
```java
Object obj = new Object();
SoftReference<Object> sf = new SoftReference<Object>(obj);
obj = null;
sf.get();//有时候会返回null
```
软引用主要用户实现类似缓存的功能，在内存足够的情况下直接通过软引用取值，无需从繁忙的真实来源查询数据，提升速度；当内存不足时，自动删除这部分缓存数据，从真正的来源查询这些数据。

弱引用：非必须引用, 强度比软引用更弱. 当GC工作时, 无论当前内存是否足够, 都会回收掉只有弱引用关联的对象.
```java
Object obj = new Object();
WeakReference<Object> wf = new WeakReference<Object>(obj);
obj = null;
wf.get();//有时候会返回null
wf.isEnQueued();//返回是否被垃圾回收器标记为即将回收的垃圾
```

虚引用：幽灵引用， 最弱的引用关系。无法通过虚引用取到实例，为一个对象设置虚引用关联的唯一目的是在这个对象被收集器回收时收到一个系统通知。
```java
Object obj = new Object();
PhantomReference<Object> pf = new PhantomReference<Object>(obj);
obj=null;
pf.get();//永远返回null
pf.isEnQueued();//返回是否从内存中已经删除
```

### GC收集算法
#### 标记／清除算法
Mask-Sweep 分为两个阶段：标记阶段和清除阶段。标记阶段的任务是标记出所有需要被回收的对象，清除阶段就是回收被标记的对象所占用的空间。![](/images/mask_sweep.png)

缺点：效率低；释放空间不连续容易导致内存碎片；会停止整个程序运行；

#### 标记-整理算法
Mark-Compact 该算法标记阶段和Mark-Sweep一样，但是在完成标记之后，它不是直接清理可回收对象，而是将存活对象都向一端移动，然后清理掉端边界以外的内存。

#### 复制算法
将可用内存按容量划分为大小相等的两块，每次只使用其中的一块。当这一块的内存用完了，就将还存活着的对象复制到另外一块上面，然后再把已使用的内存空间一次清理掉，这样一来就不容易出现内存碎片的问题。![](/images/mask_sweep_copy.png)这种算法对内存空间的使用做了一半的牺牲，效率跟存活对象的数目多少有很大的关系，如果存活对象很多，那么Copying算法的效率将会大大降低。

现在的商业虚拟机，基于IBM的研究假设：新生代中的对象98%是朝生暮死的，所以并不需要按照1:1划分内存，而是按照一定的比例把内存空间划分为一块较大的Eden空间（80%）和两块较小的Survivor空间（10%）， 每次使用Eden和其中一个Survivor。所有对象创建在新生代的Eden区，当Eden区满后触发新生代的Minor GC，将Eden区和非空闲Survivor区存活的对象复制到另外一个空闲的Survivor区中，然后再清理掉原先的空间。这样保证其中一个Survivor区是空的，新生代Minor GC就是在两个Survivor区之间相互复制存活对象，直到Survivor区满为止。

#### 分代收集算法
Generational Collection 是目前大部分JVM的垃圾收集器采用的算法。根据对象存活的生命周期将内存划分为若干个不同的区域。一般情况下将堆区划分为老年代（Tenured Generation）和新生代（Young Generation）。老年代的特点是每次垃圾收集时只有少量对象需要被回收，而新生代的特点是每次垃圾回收时都有大量的对象死亡，那么就可以根据不同代的特点采取最适合的收集算法。

目前大部分垃圾收集器对于新生代都采取Copying算法，因为新生代中每次垃圾回收都要回收大部分对象，也就是说需要复制的操作次数较少。

老年代的特点是每次回收都只回收少量对象，一般使用的是Mark-Compact或者Mask-Sweep算法。当新生代Survivor区也满了之后就通过Minor GC将对象复制到老年代。老年代也满了的话，就将触发Full GC，针对整个堆（包括新生代、老年代、持久代）进行垃圾回收。

堆区之外还有一个永久代（Permanet Generation），它用来存储class类、常量、方法描述等。对永久代的回收主要回收两部分内容：废弃常量和无用的类。持久代如果满了，将触发Full GC。

#### 垃圾收集器
垃圾收集算法是内存回收的理论基础，而垃圾收集器就是内存回收的具体实现。JDK 7的HotSpot虚拟机提供多种垃圾收集器，可以需求组合出各个年代使用的收集器。
1. Serial/Serial Old：最基本最古老的收集器，它是一个单线程收集器，并且在它进行垃圾收集时，必须暂停所有用户线程。Serial收集器是针对新生代的收集器，采用的是Copying算法，Serial Old收集器是针对老年代的收集器，采用的是Mark-Compact算法。它的优点是实现简单高效，但是缺点是会给用户带来停顿。
2. ParNew收集器是Serial收集器的多线程版本，使用多个线程进行垃圾收集。
3. Parallel Scavenge收集器是一个新生代的多线程收集器（并行收集器），它在回收期间不需要暂停其他用户线程，其采用的是Copying算法，该收集器与前两个收集器有所不同，它主要是为了达到一个可控的吞吐量。
4. Parallel Old是Parallel Scavenge收集器的老年代版本（并行收集器），使用多线程和Mark-Compact算法。
5. Current Mark Sweep（CMS）收集器是一种以获取最短回收停顿时间为目标的收集器，它是一种并发收集器，采用的是Mark-Sweep算法。
6. G1收集器是当今收集器技术发展最前沿的成果，它是一款面向服务端应用的收集器，它能充分利用多CPU、多核环境。因此它是一款并行与并发收集器，并且它能建立可预测的停顿时间模型。

### 内存分配
对象的内存分配，总的来说就是在堆上分配，对象主要分配在新生代的Eden Space和From Space，少数情况下会直接分配在老年代。![](/images/java_memory_alloc.png)

对象优先在Eden分配。
* 如果新生代的Eden Space和From Space的空间不足，则会发起一次Minor GC。
* 在GC的过程中，会将Eden Space和From Space中的存活对象移动到To Space，然后将Eden Space和From Space进行清理。
* 如果在清理的过程中，To Space无法足够来存储某个对象，就会将该对象移动到老年代中。
* 如果进行了GC之后，Eden Space和From Space能够容纳该对象就放在Eden Space和From Space。下次GC时会将存活对象复制到From Space，如此反复循环。


一般来说，大对象会被直接分配到老年代，所谓的大对象是指需要大量连续存储空间的对象，最常见的一种大对象就是大数组，比如：`byte[] data = new byte[4*1024*1024];`这种一般会直接在老年代分配存储空间。

长期存活的对象进入老年代。虚拟机给每个对象定义了一个对象年龄计数器，如对象在Survivor区躲过一次GC的话，其对象年龄便会加1，默认情况下，如果对象年龄达到15岁，就会移动到老年代中。阈值可以通过`-XX : MaxTenuringThreshold`设置.

如果在Survivor空间中相同年龄的所有对象大小的总和大于Survivor空间的一半，年龄大于或等于该年龄的对象就直接进入老年代，不必等到`MaxTenuringThreshold`

当然分配的规则并不是百分之百固定的，这要取决于当前使用的是哪种垃圾收集器组合和JVM的相关参数。

### 参考资料
* 深入理解Java虚拟机
* [Java垃圾回收机制](http://www.cnblogs.com/dolphin0520/p/3783345.html)
