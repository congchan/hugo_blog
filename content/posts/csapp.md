---
title: Computer Systems - A Programmer's Perspective (CSAPP) - CMU 15213
date: 2018-01-29
author: "Cong Chan"
tags: ['csapp', 'C']
---

> CSAPP 非常巧妙的把程序设计及优化、数字电路基础、指令集体系、汇编语言、存储器体系结构、链接与装载、进程、虚存等来自不同学科的核心知识点和在一起，并以程序员的视角呈现; 告诉我们作为一个程序员，究竟需要对计算机的硬件了解到什么程度？

本笔记是 CMU CSAPP 的学习笔记, 使用 CMU 15-213, UW CSE351 的课程视频, lab, 作业, project 辅助练习.
1. [Computer Systems: A Programmer's Perspective (csapp)](http://csapp.cs.cmu.edu/), 豆瓣-[深入理解计算机系统](https://book.douban.com/subject/26912767/)
2. [卡内基梅隆大学 CMU 15-213 Introduction to Computer Systems (ICS)](https://www.cs.cmu.edu/~213/)
3. [华盛顿大学 UW CSE351: The Hardware/Software Interface](https://courses.cs.washington.edu/courses/cse351/)
<!-- more -->

## 信息的表达与操作
Information is Bits + Context. Study systems by tracing the lifetime of the hello program, from the time it is created by a programmer, until it runs on a system, prints its simple message, and terminates.
```c
#include <stdio.h>
int main()
{ printf("hello, world\n"); }
```
The source program is a sequence of bits, each with a value of 0 or 1, organized in 8-bit chunks(bytes). Each byte represents some text character in the program.

All information in a system — including disk files, programs stored in memory, user data stored in memory, and data transferred across a network—is represented as a bunch of bits.

### 整数加减乘位移
Most machines shift and add faster than multiply, compiler translate multiply to shift and add automatically.

Power-of-2 Multiply with Shift: `u << k` gives $u \times 2^k$, `u * 24 = u * 32 - u * 8 = (u << 5) - (u << 3)`

Signed Power-of-2 Divide with Shift: `x >> k` using arithmetic shift(补1) gives $x / 2^k$, when `u < 0`, say `y = -15213 = b[11000100 10010011]`, `y >> 1 = b[11100010 01001001] = -7607`, the rounding is downward, which is not the same as the convention toward zero.

Correct Power-of-2 Divide by adding bias: $x / 2^k$, computed as $(x + 2^k - 1) / 2^k$, in C `(x<0 ? x+(1<<k)-1 : x) >> k`

### 浮点数
IEEE floating-point standard represents a number in a form $V = (−1)^s × M × 2^E$
1. sign `s`: determines whether the number is negative (s = 1) or positive (s = 0)
2. exponent `E` weights the value by a (possibly negative) power of 2, encoded by the `k`-bit exponent field `exp`.
3. significand `M`: a fractional binary number that ranges either [1, 2 − ϵ) or [0, 1 − ϵ), encoded by the `n`-bit fraction field `frac`

![](/images/floating_point.png "image from http://www.cs.cmu.edu/~213/")
Case 1: Normalized Values
`E` is interpreted as representing a signed integer in biased form, `E = e − Bias`, where `e` is the **unsigned** number, `Bias` equals to $2^{k−1}−1$.
The significand is defined to be `M = 1 + f`, where `f` is the fraction field, `0 ≤ f < 1`, `M` =  $1.f_{n−1}f_{n−2}...f_0$

Case 2: Denormalized Values
Exponent field is all zeros, the exponent value is `E = 1 − Bias`, and the significand value is `M = f`, `M` =  $0.f_{n−1}f_{n−2}...f_0$
Denormalized numbers serve two purposes.
1. provide a way to represent numeric value `0`,
  * `+0.0`, bit pattern all zeros, `s = M = f = 0`
  * `−0.0`, bit pattern all zeros except `s = 1`.
2. represent numbers that are very close to `0.0`: possible numeric values are spaced evenly near `0.0`

Case 3: Special Values
Exponent field is all ones.
1. When the fraction field is all zeros, the resulting values represent **infinity**, either `+∞` when `s = 0`, or `−∞` when `s = 1`. Infinity can represent results that overflow
2. When the fraction field is nonzero, the resulting value is `NaN`

![](/images/floating_decoding.png "image from http://www.cs.cmu.edu/~213/")
![](/images/floating_point_range.png "image from http://www.cs.cmu.edu/~213/")

### Programs are traslated by other programs into different forms
The hello program begins as a high-level C program because it can be read and understood by human beings in that form. However, in order to run hello.c on the system, the individual C statements must be translated by other programs into a sequence of low-level machine-language instructions.

These instructions are then packaged in a form called an executable object program and stored as a binary **disk** file. Object programs are also referred to as executable object files.

The programs that perform the four phases (preprocessor, compiler, assembler, and linker) are known collectively as the compilation system.
* Preprocessing phase.The preprocessor (cpp) modifies the original C program according to directives that begin with the # character.
* Compilation phase. The compiler (cc1) translates the text file hello.i into the text file hello.s, which contains an assembly-language program. Assembly language is useful because it provides a common output language for different compilers for different high-level languages.
* Assembly phase. Next, the assembler (as) translates hello.s into machinelanguage instructions, packages them in a form known as a relocatable object program, and stores the result in the object file hello.o.
    * The hello.o file is a binary file whose bytes encode machine language instructions rather than characters.
* Linking phase. The printf function resides in a separate precompiled object file called printf.o, which must somehow be merged with our hello.o program. The linker (ld) handles this merging.

大多数 JVM 将内存区域划分为 **Method Area（Non-Heap）（方法区）**, **Heap（堆）**, **Program Counter Register（程序计数器）**, **VM Stack（虚拟机栈/JAVA方法栈）**, **Native Method Stack（ 本地方法栈 ）**，其中Method Area和Heap是线程共享的，VM Stack，Native Method Stack和Program Counter Register是非线程共享的。![](/images/JVM内存模型图.png)
* 程序计数器是一个比较小的内存区域，用于指示当前线程所执行的字节码执行到了第几行，是线程隔离的
* 虚拟机栈描述的是Java方法执行的内存模型，用于存储局部变量，操作数栈，动态链接，方法出口等信息，是线程隔离的
* 原则上讲，所有的对象都在堆区上分配内存，是线程之间共享的
* 方法区域存放了所加载的类的信息（名称、修饰符等）、类中的静态变量、类中定义为final类型的常量、类中的Field信息、类中的方法信息，当开发人员在程序中通过Class对象中的getName、isInterface等方法来获取信息时，这些数据都来源于方法区域，同时方法区域也是全局共享的，在一定的条件下它也会被GC，当方法区域需要使用的内存超过其允许的大小时，会抛出OutOfMemory的错误信息

一个一般性的 Java 程序工作过程：
1. 一个 Java 源程序文件，会被编译为字节码文件（以 class 为扩展名），每个java程序都需要运行在自己的JVM上，然后告知 JVM 程序的运行入口，再被 JVM 通过字节码解释器加载运行。
2. 程序开始运行后，开始涉及各内存区域：
    1. JVM初始运行的时候都会分配好 Method Area（方法区）和Heap（堆），
    2. 而JVM每遇到一个线程，就为其分配一个Program Counter Register（程序计数器）, VM Stack（虚拟机栈）和Native Method Stack（本地方法栈），当线程终止时，三者所占用的内存空间也会被释放掉。

这也是为什么把内存区域分为线程共享和非线程共享的原因，非线程共享的那三个区域的生命周期与所属线程相同，而线程共享的区域与JAVA程序运行的生命周期相同，所以这也是系统垃圾回收的场所只发生在线程共享的区域（实际上对大部分虚拟机来说知发生在Heap上）的原因。![](/images/Java内存区域.png)

### Processors read and interpret instructions stored in memory
The hello.c source program has been translated by the compilation system into an executable object file called hello that is stored on disk, to run the executable file on Unix:
```
unix> ./hello
hello, world
unix>
```
The shell is a command-line interpreter that prints a prompt, waits for you to type a command line, and then performs the command.

### Hardware organization of a systems
Hardware organization of a typical system.
#### Buses
Electrical conduits that carry bytes of information back and forth between the components. Buses are typically designed to transfer fixed-sized chunks of bytes known as words. USB: Universal Serial bus.
#### Input/output (I/O) devices
The system’s connection to the external world. Each I/O deviceisconnected to the I/O bus by either a controller or an adapter：
* Controllers are chip sets in the device itself or on the system’s main printed circuit board (often called the motherboard).
* An adapter is a card that plugs into a slot on the motherboard.

#### Main Memory
A temporary storage device that holds both a program and the data it manipulates while the processor is executing the program.
* Physically, main memory consists of a collection of dynamic random access memory (DRAM) chips.
* Logically, memory is organized as a linear array of bytes, each with its own unique address (array index) starting at zero
*
#### Processor: Central Processing Unit (CPU)
* PC: Program counter, a word-sized storage device (or register) at CPU core. At any point in time, the PC points at (contains the address of) some machine-language instruction in main memory.
* Register: a quickly accessible location available to CPU,
* Register file: an array of registers, each with its own unique name.
* Arithmetic/logic unit: ALU computes new data and address values.

A processor repeatedly executes the instruction pointed at by the program counter and updates the program counter to point to the next instruction. The processor reads the instruction from memory pointed at by the PC, interprets the bits in the instruction, performs some simple operation dictated by the instruction, and then updates the PC to point to the next instruction.

#### CPU operations examples
Load: Copy a byte or a word from main memory into a register, overwriting the previous contents of the register.

Store(write): Copy a byte or a word from a register to a location in main memory, overwriting the previous contents of that location.

Operate: Copy the contents of two registers to the ALU, perform an arithmetic operation on the two words, and store the result in a register, overwriting the previous contents of that register.

Jump: Extract a word from the instruction itself and copy that word into the program counter (PC), overwriting the previous value of the PC.

Branch greater than (BGT): compares two registers and decides whether to branch (target would be the address to branch to), i.e. it is implementing the "if" decision.

### Running a programs
1. Initially, the shell program is waiting for user types a command. As we type the characters “./hello” at the keyboard, the shell program reads each one into a register, and then stores it in memory.
2. When we hit the *enter* key on the keyboard, the shell knows that we have finished typing the command. The shell then loads the executable hello file by executing a sequence of instructions that copies the code and data in the hello object file **from disk to main memory**. The data include the string of characters “hello, world\n” that will eventually be printed out. Using a technique known as direct memory access (DMA), the data travels directly from disk to main memory, without passing through the processor.
3. Once the code and data in the hello object file are loaded into memory, the processor begins executing the machine-language instructions in the hello program’s main routine. These instructions copy the bytes in the `hello, world\n` string from memory to the register file, and from there to the display device, where they are displayed on the screen.

### Caches
An important lesson from this simple example is that a system spends a lot of time moving information from one place to another. From a programmer’s perspective, much
of this copying is overhead that slows down the “real work” of the program. Because of physical laws, larger storage devices are slower than smaller storage devices. Speed that processor read from: register > memory > disk.

It is easier and cheaper to make processors run faster than it is to make main memory run faster. To deal with the processor-memory gap, system designers include smaller
faster storage devices called cache memories (or simply caches) that serve as temporary staging areas for information that the processor is likely to need in the near future.

The L1 and L2 caches are implemented with a hardware technology known as static random access memory (SRAM). Newer and more powerful systems even have three levels of cache: L1, L2, and L3.

By setting up caches to hold data that is likely to be accessed often, we can perform most memory operations using the fast caches.

### Storage Devices Form a Hierarchy
![](/images/memory_hierarchy.png "source from:http://csapp.cs.cmu.edu/")

### Operating system
The operating system has two primary purposes: (1) to protect the hardware from misuse by runaway applications, and (2) to provide applications with simple and uniform mechanisms for manipulating complicated and often wildly different low-level hardware devices.

Think of the operating system as a layer of software interposed between the application program and the hardware, with fundamental abstractions: processes, virtual memory, and files.
![](/images/operating_system_abs.png "Abstractions provided by an operating system. Source from:http://csapp.cs.cmu.edu/")

#### Process进程
A process is the operating system’s abstraction for a running program. Multiple processes can run concurrently on the same system by having the processor switch (**context switching**) among them, and each process appears to have exclusive use of the hardware.

The os keeps track of all the state information that the process needs in order to run. This state, i.e. the context, includes information such as the current values of the PC, the register file, and the contents of main memory.

When the operating system decides to transfer control from the current process to some new process, it performs a context switch by saving the context of the current process, restoring the context of the new process, and then passing control to the new process. The new process picks up exactly where it left off. ![](/images/Process_context.png "Process context switching. Source from:http://csapp.cs.cmu.edu/")

#### Virtual Memory
Virtual memory is an abstraction that provides each process with the illusion that it has exclusive use of the main memory. Each process has the same uniform view of memory, which is known as its virtual address space.

In Linux, the topmost region of the address space is reserved for code and data in the operating system that is common to all processes. The lower region of the address space holds the code and data defined by the user’s process.
![](/images/Process_virtual_address.png "Process virtual address space. Source from:http://csapp.cs.cmu.edu/")
Starting with the lowest addresses and working our way up:
1. Program code and data: Fixed in size once the process begins running. The code and data areas are initialized directly from the contents of an executable object file, in our case the hello executable.
2. Run-time heap: expands and contracts dynamically at run time as a result of calls to C standard library routines such as `malloc` and `free`.
3. Shared libraries: holds the code and data for shared libraries such as the C standard library and the math library.
4. User stack: the compiler uses to implement function calls. Each time we call a function, the stack grows. Each time we return from a function, it contracts.
5. Kernel virtual memory: The kernel is the part of the operating system that is always resident in memory. Application programs are not allowed to read or write the contents of the top region of the address space (which is reserved for the kernel) or to directly call functions defined in the kernel code.

#### Thread线程
In computer science, a thread of execution is the smallest sequence of programmed instructions that can be managed independently by a scheduler, which is typically a part of the operating system.

In most cases a thread is a component of a process. Multiple threads can exist within one process, executing concurrently and sharing resources such as memory, while different processes do not share these resources.

Threads are an increasingly important programming model because of the requirement for concurrency in network servers, because it is easier to share data between multiple threads than between multiple processes, and because threads are typically more efficient than processes.

#### Files
A file is a sequence of bytes. Every I/O device, including disks, keyboards, displays, and even networks, is modeled as a file. All input and output in the system is performed by reading and writing files, using a small set of system calls known as *Unix I/O*.



## Concurrency and Parallelism
Concurrency: general concept of a system with multiple, simultaneous activities.
Parallelism: the use of concurrency to make a system run faster.

Parallelism could be achieved in different levels of abstraction in computer system. There are three common levels (from the highest to the lowest level in the system hierarchy):

### Thread-Level Concurrency
Building on the process abstraction, we are able to devise systems where multiple programs execute at the same time, leading to concurrency. With threads, we can even have multiple control flows executing within a single process.

When we construct a system consisting of multiple processors all under the control of a single operating system kernel, we have a multiprocessor system

**Multi-core processors**: Several CPUs (referred to as “cores”) integrated onto a single integrated-circuit chip.

**Hyperthreading**: Sometimes called simultaneous multi-threading, is a technique that allows a single CPU to execute multiple flows of control.

### instruction-level parallelism
At a much lower level of abstraction, modern processors can execute multiple instructions at one time.

### Single-Instruction, Multiple-Data (SIMD) Parallelism
At the lowest level, special hardware that allows a single instruction to cause multiple operations to be performed in parallel.

## Memory, Data, & Addressing
十进制，2进制，16进制:
* A single byte consists of 8 bits.
* 二进制 value ranges from 00000000<sub>2</sub> to 11111111<sub>2</sub>,
* 十进制 value ranges from 0<sub>10</sub> to 255<sub>10</sub>
* 二进制表示法过于冗长，而使用十进制表示法，与bits进行模式转换非常繁琐。
* 十六进制，hexadecimal numbers: Hexadecimal (or simply “hex”) uses digits ‘0’ through ‘9’ along with characters ‘A’ through ‘F’ to represent 16 possible values. Values range from 00<sub>16</sub> to FF<sub>16</sub>.

内存：
* A machine-level program views memory as a very large array of bytes, referred to as virtual memory.
* Every byte of memory is identified by a unique number, known as its address.
* The set of all possible addresses is known as the virtual address space - 进程可用的虚拟地址范围称为该进程的“虚拟地址空间”。

这个虚拟地址空间只是一个呈现给机器级程序的虚拟概念。实际的实现需要用到随机访问存储器（RAM），磁盘存储，特殊的硬件和操作系统软件的组合来构建相对于程序而言的单片字节数组。

### Address and Pointers
地址是内存的位置，指针是一种包含地址的数据对象。

Byte ordering: Endianness
* **little endian** - where the least significant byte comes first, followed by most Intel-compatible machines.
* **big endian** - where the most significant byte comes first, followed by most machines from IBM and Sun Microsystems
* Many recent microprocessors are bi-endian, meaning that they can be configured to operate as either little- or big-endian machines.

### Integer and floating point numbers
把多个bits组合起来，通过解码，可以表达有限集合内的所有元素。比如二进制数字系统可以表示正整数。

Three most important representations of numbers.
1. Unsigned encodings：based on traditional binary notation, representing numbers greater than or equal to 0.
2. Two’s-complement encodings: the most common way to represent signed integers, that is, numbers that may be either positive or negative.
3. Floating-point encodings: base-two version of scientific notation for
representing real numbers.

## C
### Methods
* Naming data types with `typedef`: C的`typedef`声明用于给数据类型命名。这对提高代码可读性有很大的帮助，因为深层嵌套类型声明可能难以解读。
    ```c
    typedef int *int_pointer;
    int_pointer ip;
    ```
    等同于`int *ip;`
* Formatted printing with `printf`(`fprintf` and `sprintf`): provides a way to print information
with considerable control over the formatting details.
    * The first argument is a format string. Each character sequence starting with ‘%’ indicates how to format the next argument. `%d` - 输出十进制整数, `%f` - 浮点数, `%c` - 字符
    * while any remaining arguments are values to be printed.
* `sizeof(T)` returns the number of bytes required to store an object of type T
* `void *malloc(size_t size)`分配请求的内存(size in bytes)并返回一个指向它的指针(如果请求失败，则返回`NULL`)。

### Addresses and pointer in C
指针是C的核心功能，可以引用数据结构元素（包括数组）。就像变量一样，指针有两个组成部分：值和类型。该值指示某个对象的位置，而其类型指示该位置处存储什么类型的对象（例如，整数或浮点数）。
* `&` - “address of", return a pointer;
* Variable declarations： `int x`, find location in memory in which to store integer.
* Pointer declarations use `*`: `int *pointer`, declares a variable `pointer` that is a pointer pointing to an object of type integer.
* Assignment to a pointer: `pointer = &x`, assigns `pointer` to point to the address where `x` is stored.
* To use the value pointed to by a pointer, use `*`:
    * if `pointer = &x`, then `x = *pointer +1` is the same as `x = x + 1`
    * 假如x是一个对象, 那么`*(&x)`=`*&x` = `x`

### Pointers and arrays
C通过数组将标量数据聚合为更大的数据类型。In C, we can dereference a pointer with array notation, and we can reference array elements with pointer notation.
C有一个不常见的特性, 就是我们可以生成指向数组内的元素的指针，并使用这些指针来执行算术运算。
```c
T A[N];
```
首先，它在内存中分配一个L*N大小的连续区域, 其中L是数据类型T的大小（以bytes为单位）. 数组的元素可以使用 0 ~ N-1 之间的整数索引来访问 `A[i]`;

其次，它引入了一个标识符A，可以作为指向数组开头的指针;

在指针上进行算术运算时，其实际的索引值会根据指针引用的数据类型的大小进行缩放,  即假设A的值是xa, 那么`A+i`的值就是`xa + L * i`, `A[i]` = `*(A+i)`;

### 其他
`#define` 指令允许在源代码中定义宏 macro。这些宏定义允许在整个代码中声明常量值。 宏定义不是变量，不能像变量那样通过程序代码进行更改。创建表示数字，字符串或表达式的常量时，通常使用此语法。

定义常数：`#define CNAME value` or `#define CNAME (expression)`。`CNAME`是常数的名称。大多数C程序员用大写字母来定义常量名，但这不是C语言的要求。`expression`就是被分配给常量的表达式。如果表达式包含运算符，则该表达式必须括在括号内。
