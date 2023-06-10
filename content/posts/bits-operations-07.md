title: 位操作 - 快速幂
date: 2017-09-25
mathjax: true
categories:
- CS
tags:
- Bitwise Operation
- Software Engineer
- Computer Science
- Algorithms
---
如何实现快速的幂运算？
<!-- more -->

要求$c = a^b$, 按照朴素算法把`a`连乘`b`次的时间复杂度是$O(n)$. 而快速幂能做到$O(\log n)$。把`b`转换为二进制, 二进制数第`i`位的权为$2^{i-1}$，就可以把二进制拆分为若干个以`2`为底的真数, 然后利用幂数的性质，例如用朴素算法求$a^{11}$要求乘`11`次. 考虑到`11`的二进制为`1011`, 如果把$a^{11}$拆分为:
$$a^{11} = a^{a_0 2^0 + a_1 2^1 + a_2 2^2 + a_3 2^3} = a^1 a^2 a^8$$
可以看到每一个因子都是上一个因子的平方，利用$a^2 a^2$求出$a^4$, 同样利用$a^4$的平方求出$a^8$, 每次计算只需要用到上一次计算出来的结果, 所以总的运算次数是`4`次. 任何一个数`b`最多能写成长度为$O(\log b)$的二进制, 因此这个算法就是$O(\log n)$.

在程序设计中是根据`b`的二进制中是否为`1`来控制是否乘以上一次翻倍的积
* 不断右移`b`, 直到`b`不再有`1`：
    * 根据当前位的权重（当前`b`最后一位）是否为`1`来决定`c`是否乘以最新的`a`
    * 把`a`平方，用于下一位计算

在Java中要考虑极端值INT_MIN
```java
// 递归
public double myPow(double x, int n) {
    if(n==0) return 1;

    double temp = myPow(x, n/2);
    if (n % 2 ==0) return temp * temp;
    else
    {
        if(n > 0) return x*temp*temp;
        else return (temp*temp) / x;
    }
}
```
```java
// 循环
public double myPow(double x, int n) {
    double ans = 1;
    if(n < 0){
        n = -(n+1); // 处理极端值
        x = 1/x;
        ans *= x;
    }

    System.out.println(n);

    while (n > 0) {
        if ((n & 1) == 1) ans *= x;
        x *= x;
        n >>= 1;
    }
    return ans;
}
```

### 快速幂取余
求`a^b mod c`.
如果`b`是偶数, `a^b mod c` = $(a^2)^{b/2} \% c$
如果`b`是奇数, `a^b mod c` = $((a^2)^{b/2} \times a) \% c$

又因为取余有性质:`a^b mod c = (a mod c)^b`

引理：`(a * b) mod c = [( a mod c ) * (b mod c) ] mod c`

证明：
```
设 a mod c =d，b mod c= e;
       则：a=t*c + d ;  b=k*c + e ;
       (a*b)mod c = (t*c+d)(t*c+e)
                 = (tk c^2 + ( te+dk ) *c + d*e) mod c
                 = de mod c
```
即积的取余等于取余的积的取余.

利用快速幂的思想, 令`k = (a * a) mod c`，所要求的最终结果即为 `k^(b/2) mod c`, 这个过程可以迭代下去, 如果b是奇数, 或多出一项`a mod c`. 当`b = 0`时, 所有因子已经相乘, 迭代结束, 复杂度为`O(log b)`
```java
long long  PowerMod(int a, int b, int c)
{
    int  ans = 1;
    a = a % c;
    while(b>0) {
        if(b % 2 = = 1)
            ans = (ans * a) % c;
        b = b/2;       //   b>>=1;
        a = (a * a) % c;
    }
    return ans;
}
```
