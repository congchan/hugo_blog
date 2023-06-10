title: Dynamic Programming 05 - 跳台阶
date: 2017-09-05
mathjax: true
categories:
- CS
tags:
- Java
- Algorithms
- Dynamic Programming
---
## 跳台阶
跳上一个n级的台阶总共有多少种跳法，先后次序不同算不同的结果，限制条件是每次只能跳1级或者2级。
<!-- more -->
抽象出来的模型是：给定正整数`N`，有多少种累加方案，不同顺序当做不同方案，限制条件可以是给定的整数$n_0, n_1, ..., n_k$作为可选累加元素.

对于限制条件为只有两种跳法, 即1阶或者2阶的, 问题可以分解为:
* 假定第一次跳的是`1`阶，那么就剩下`n-1`个台阶，剩余跳法是`f(n-1)`;
* 假定第一次跳的是`2`阶，则剩下`n-2`个台阶，剩余跳法是`f(n-2)`

可以归纳出通用的公式: `f(n) = f(n-1) + f(n-2)`, 只有一阶的时候`f(1) = 1`, 只有两阶的时候可以有`f(2) = 2`, 刚好就是斐波那契数列. 所以这个简单的跳台阶问题就是计算斐波那契数列的问题。

反过来思考, 比如对于`8`个台阶, 有多少种回滚方案? 只有两种: 回滚1个台阶, 就到了`7`; 回滚2个台阶, 就到了`6`. 等于说: 假如有`f(7)`种方案跳到`7`, 有`f(6)`种方案跳到`6`，那么就有`f(7) + f(6)`种方案到达`8`

从树结构来理解: 如果节点代表台阶数`n`对应的跳法`f(n)`, 节点与节点间的枝代表单次可以跳的阶数, 父节点的值就是其所有子节点的值和. 对于只有两种跳法限制问题, 父节点`f(n)`就只有两个子节点, 分别为`f(n-1)`和`f(n-2)`.

### 斐波那契数列
举例：Fibonacci sequence: ${\displaystyle 0,\;1,\;1,\;2,\;3,\;5,\;8,\;13,\;21,\;34,\;55,\;89,\;144,\;\ldots }$
$$F_0 = 0, F_1 = 1, F_2 = 1, F_n = F_{n-1} + F_{n-2} (n>2) $$

Fibonacci numbers grow almost as fast as the powers of 2.

Recursive solution is exponential algorithm
```
fib1(n):
    if n = 0:  return 0
    if n = 1:  return 1
    return fib1(n - 1) + fib1(n - 2)
```
因为每一个`fib1()`都会生成指数级数量的子分支计算, 所以这个算法复杂度是$O(2^n)$.

但是注意到斐波那契数列公式是$F_n = F_{n-1} + F_{n-2}$, 也就是只要知道n前面两个值, 就能计算出$f_n$. 又因为斐波那契数列天然的是从低往高算, 那么每次迭代只需要用到前两次的值$F_{n-1}, F_{n-2}$, 计算后更新它们即可. 用这个思路来计算斐波那契数列, 复杂度就是$O(n)$.
```java
public int JumpFloor(int target) {
    if (target <= 1) { return target; }
    int n = 2;
    int n0 = 1;
    int n1 = 1;
    int ways = 0;
    while (n <= target) {
        ways = n0 + n1;
        n0 = n1;
        n1 = ways;
        n++;
    }
    return ways;
}
```

### 变态跳台阶
变态跳台阶就是是用来更复杂的限制条件, 比如可选单次跳阶数为`[1, ... n]`, 也就是无限制的情况, 也可以按照上面的思路推导.

比如从树结构的考虑, 就变成每个父节点`f(n)`可以有`n`个子节点, 就是`f(n-1), f(n-2), ..., f(n-n)`, 所以`f(n)`就是所有这些子节点的和. `f(n-n)`也就是`f(0)`意味着一次跳完所有阶数`n`, 所以`f(0) = 1`. 进一步归纳, `f(n-2) + ... + f(n-n) = f(n-1)`, 所以`f(n) = f(n-1) + f(n-1)`, 可以用递归或者动态规划来计算.

当然进一步归纳会发现$f(n) = 2^{n-1}$, 可以用位移来操作:
```java
public int JumpFloorII(int target) {
    int a=1;
    return a << (target - 1);
}
```
只是要注意`int`是有范围的.

### 大变态跳台阶
再举一个更复杂的限制条件, 可选单次跳阶数为$2^0, 2^1, ..., 2^k$, $2^k$要小于`n`. 那么相应的,
$$f(n) = f(n - 2^0) + f(n - 2^1)... + f(n - 2^k), \quad s.t. \quad 2^k <= n,$$
这样就意味着对于每个`f(n)`, 需要用到的`f(k)`值数量是不同的, 就不能简单地用固定数量的变量来保留较小值了.

对于不同的`f(n)`, 它们的很多子分支计算是共享的, 比如`f(6)`和`f(5)`都用到了`f(4)`. 那么在递归的过程中，只要把每次计算出来的较小的`f(k)`储存到数组中, 后续其他`f(n)`要用到`f(n - 2^k)`时, 直接从内存中取值即可. 初始值取`f(0) = f(1) = 1`:
```java
public int JumpFloorIII(int target) {
    int[] f = new int[target];
    f[0] = f[1] = 1;
    return jump(f, target);
}

private static int jump(int[] f, int target) {
    if (f[target] == 0) {
        int ways = 0;
        for (int i = 0; (1 << i) <= target; i++) {
            ways += jump(f, target - (1 << i));
        }
        f[target] = ways;
    }
    return f[target];
}
```
这个代码适用于`n <= 1024`的情况. 否则要改为循环。
