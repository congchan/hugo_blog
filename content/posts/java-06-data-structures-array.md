title: Java 06 | 数据结构 - array 数组
date: 2016-12-27
categories:
- CS
tags:
- Software Engineer
- Java
---
## 数组（Array）
数组是一种特殊的对象，有一个固定的数组长度参数N，由一连串（N个）连续的带编号的内存块组成，每个都是相同的类型(不像Python可以包含不同类型)，索引从0到N-1编号。A[i]获得数组A的第i个元素。这与普通的类实例不同，类实例有具体变量名命名的内存块。
### 数组实例化，包含对象的数组
Array Instantiation, Arrays of Objects

要创建最简单的整数数组, 有三种方式:
```java
x = new int [3]; //创建一个指定长度的数组，并用默认值（0）填充每个内存块。
y = new int [] {1，2，3，4，5}; //创建一个合适大小的数组，以容纳指定的初始值
int [] z = {9，10，11，12，13}; //省略了new，只能结合变量声明使用。
```
<!-- more -->
创建一组实例化对象:
```java
public class DogArrayDemo {
    public static void main(String[] args) {
        /* Create an array of two dogs. */
        Dog[] dogs = new Dog[2];
        dogs[0] = new Dog(8);
        dogs[1] = new Dog(20);
        /* Yipping will result, since dogs[0] has weight 8. */
        dogs[0].makeNoise();
    }
}
```
注意到new有两种不同的使用方式：一种是创建一个可以容纳两个Dog对象的数组，另外两个创建各个实际的Dog实例。

数组复制
```java
x = new int[]{-1, 2, 5, 4, 99};
int[] b = {9, 10, 11};
System.arraycopy(b, 0, x, 3, 2); //效果类似于Python的`x[3:5] = b[0:2]`
```
`System.arraycopy`的五个参数分别代表：
1. 待复制的数组(源)
2. 源数组复制起点
3. 目标数组
4. 目标数组粘贴起点
5. 有多少项要复制


2D数组
Java的二维数组实质上是一数组的数组, 即每一个数组元素里面也是一个数组。
```java
int[][] matrix; //声明一个引用数组的数组
matrix = new int[4][]; //创建四个内存块, 用默认null值填充, 之后用于储存对整数数组的引用, 即地址,
int[] rowZero = matrix[0];

/** 实例化整数数组, 把其地址/引用分别赋值给/储存到 matrix 的第N个内存块*/
matrix[0] = new int[]{1};
matrix[1] = new int[]{1, 1};
matrix[2] = new int[]{1, 2, 1};
matrix[3] = new int[]{1, 3, 3, 1};

int[] rowTwo = matrix[2];
rowTwo[1] = -5;

/** 创建四个内存块, 其中每个被引用的整数数组长度为4,每个元素都是0.*/
matrix = new int[4][4];

int[][] matrixAgain = new int[][]{{1}, {1, 1},{1, 2, 1}, {1, 3, 3, 1}};
```
