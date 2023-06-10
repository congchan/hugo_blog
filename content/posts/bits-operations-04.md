title: 位操作 - 找数问题
date: 2017-09-24
mathjax: true
categories:
- CS
tags:
- Bitwise Operation
- Software Engineer
- Computer Science
- Algorithms
---
“找出只出现一次的数”， “找出唯二的只出现M次的数”， “找出缺失的数”等等这类问题，都可以利用异或操作的特性， 即一个整数和自己进行异或运算会归0的性质。
<!-- more -->

## 找出缺失的数字
问题1：给定一个包含n个不同数字的数组，取自`0,1,2,...,n`，找到数组中缺少的数字。

最直觉的解法是利用等差数列的性质直接数学求解。但这个方法限制于等差数列.

问题2: 在一个长度为`n`的数组里的所有数字都在`0 ~ n-1`之间, 数组中有些数字是重复的, 找出任意一个重复的数字. 这也是<<剑指offer>>的一道题.

但是如果利用数组大小和元素范围的特性, 就可以发现, 这里的数组的大小和数字的范围是有限定关系的. 对于第二个问题, 假如没有重复, 那么重新排列的话数组每一个位置都可以放上自己对应的数字. 对于第一个问题, 假如没有缺失, 那么除了每一个index都可以重新放上自己对应的数字外, 还会多出一个最大的数字没地方放. 这样就可以把数组包含的数字解读为index, 然后在遍历检查数组时, 同时检查以各个数字为index的其他位置的数字.

使用这种思路可以同时解决两个问题, 这里以问题1解法为例:
```java
public int missingNumber(int[] nums) {
    int n = nums.length;
    int misP = n; // points to the position where misssing.
    for (int i = 0; i < n; i++) 
    {
        while (i != nums[i] && nums[i] != misP)
        {
            int j = nums[i];
            nums[i] = nums[j];
            nums[j] = j;
        }   
        if (nums[i] == misP) 
            misP = i;
    }
    return misP;
}
```

## 找出只出现一次的数
问题3：在一个非空整数数组，找出那个只出现了一次的元素，已知其余每个元素均出现两次。

要达到`O(n)`复杂度需要利用**位异或**. 位异或运算能够把二进制相同的数化为0. 把数组所有的数都异或, 出现两次的数就会互相抵消为0, 剩余的就是那个只出现了一次的数:
```java
public int singleNumber(int[] nums) {
    int output = 0;
    for (int i : nums)
        output ^= i;
    return output;
}
```

问题3其实等价于问题1, 如果把问题1再加上另外一个完整连续不重复`0,1,2,...,n-1`数组(可以直接循环数组索引). 所以问题1也可以用同样思路解决:
```java
public int missingNumber(int[] nums) {
    int miss = nums.length;
    for (int i = 0; i < nums.length; i++)
        miss ^= (nums[i] ^ i);
    return miss;
}
```

## 找出唯一一个仅出现M次的数
但如果把问题3扩展为“其余每个元素均出现三次”， 也就是leetcode 137. Single Number II, 这样就无法直接利用异或抵消的性质了。剑指Offer的解法是用一个长度`32`的数组`bitSum`, 把原数组所有整数的二进制每一位分别累加到`bitSum`里面, 这样就可以通过判断`bitSum`哪些位不可以被3整除来找出那个数:
```java
public int singleNumber(int[] nums) {
    int[] bitSum = new int[32];
    for (int i = 0; i < nums.length; i++)
    {
        int bitMask = 1;
        for (int j = 31; j >= 0; j--)
        {
            int b = nums[i] & bitMask;
            if (b != 0)
                bitSum[j] += 1;
            bitMask <<= 1;
        }
    }
    int res = 0;
    for (int i = 0; i < 32; i++)
    {
        res <<= 1;
        res += bitSum[i] % 3;
    }
    return res;
}
```
因为`bitSum`的长度是常数, 所以该方法复杂度还是`O(N)`. 该方法可以进一步扩展问题为求唯一一个元素出现`M`次，其他所有元素出现`K`次的问题。

### 构造状态转移表
方法来自[An General Way to Handle All this sort of questions](https://leetcode.com/problems/single-number-ii/discuss/43296/An-General-Way-to-Handle-All-this-sort-of-questions.), 这个方法核心思想是建立一个记录状态的变量, 该变量代表某个数字出现一次后的状态. 目标就是使得一个数字重复出现K次后状态刚好归0.

对于`K=2`, 就要使两次叠加后归0, 需要两种状态, 从信息论的角度看待, 只需要一个位(`0`,`1`)来表达，状态`0`对应着两种等价的情况: 一个数字完全没出现过, 或者出现了2次后一起抵消重置. 状态`1`对应着仅仅出现一次的情况. 在这里数字和状态概念等价，构建一个状态转移表（真值表）：
```
状态 输入 输出
a    c    a
0    0    0
1    0    1
0    1    1
1    1    0
```
可以看到，不管是状态1还是0，如果输入相同数字，就会变为0；如果输入不同的数字，就会变为1. 根据表写出逻辑表达式为异或运算.

根据真值表写出逻辑式的基本套路是: 只看输出结果为1的转移, 凡取值为1的变量写成原变量，取值为0的变量写成反变量, 得出对应的表达式, 再把所有转移方程的表达式加起来. 如输出为1的是`0 & 1 = 1, 1 & 0 = 1`, 表达式就是`(~a & c) | (a & ~c)`, 这个本质上就是`a ^ c`

对于`K = 3, M = 1(or 2)`, 需要三种状态, 那么至少需要两个位(`00`, `01`, `10`)来表达. 让状态`00`对应"假"输出, 对应两种等价的情况: 一个数字完全没出现过, 或者出现了3次后一起抵消重置. 再定义`01`为出现了一次的状态, `10`为出现了2次, 这两种状态都对应着"真"输出, 也就是我们想要的答案, 得出状态转移为:
```
状态      输入      输出
(a, b)    (c)      (a,b)
0, 0	   0	   0, 0
0, 1	   0	   0, 1
1, 0	   0	   1, 0
0, 0	   1	   0, 1
0, 1	   1	   1, 0
1, 0	   1	   0, 0
```
得出`a = (a & ~b & ~c) | (~a & b & c)`, `b = (~a & b & ~c) | (~a & ~b & c)`.
只要把数组所有数按照这个逻辑分别叠加到`a`和`b`上面, 最后答案就是`a | b`.
```java
public int singleNumber(int[] nums) {
    int a = 0, b = 0;
    for (int c : nums)
    {
        int temp = (a & ~b & ~c) | (~a & b & c);
        b = (~a & b & ~c) | (~a & ~b & c);
        a = temp;
    }
    return (a | b);
}
```
以上只是一种通用的套路，对于每一种特定的`K, M`组合, 可能会有不同的特殊最优方案.

### 通过不同集合收录不同数字
同上面的问题，LeetCode某大神给出一个[目前为止最优的方案](https://leetcode.com/problems/single-number-ii/discuss/43294/Challenge-me-thx), 并放言"Challenge me", 草鸡们看了瑟瑟发抖：
```java
public int singleNumber(int[] nums) {
    int ones = 0, twos = 0;
    for (int c : nums)
    {
        ones = (ones ^ c) & ~twos;
        twos = (twos ^ c) & ~ones;
    }
    return ones;
}
```
原理是利用两个数`ones`和`twos`作为一种概念上的集合`set`，通过异或操作来收录分别出现了1次和2次的数, `set ^ val`有两种结果:
* 如果`set`里面没有`val`, 把`val`异或进去, 如`a ^ 0 = a`
* 如果`set`之前已经收录了`val`, 那么亦或操作就会在`set`中移除这个`val`, 如 `a ^ a = 0`

按照上面的定义来理解:
* `(ones ^ c) & ~twos`: 当且仅当`c`没有收录在`twos`中, 把`ones`收录`c`，否则移除`c`。这样的话，任何第一次出现的数都会被收入`ones`中, 而任何第二次出现的数会从`ones`中移出.
So, effectively anything that appears for the first time will be in the set. Anything that appears a second time will be removed. We'll see what happens when an element appears a third time (thats handled by the set "twos").
* 紧接着, `(twos ^ c) & ~ones`用同样的逻辑更新`twos`. 这样意味着
    * `twos`不会收录第一次出现的数;
    * 但对于第二次出现的数, 因为上一步已经把这个数从`ones`中移除, 那么这个数就会被收录进`twos`中,
    * 对于第三次出现的数, 因为`twos`中已经收录了, 所以`ones`不会再收录, 而异或操作会把`twos`中的这个数移除.

最后的结果就是, `ones`仅保留出现了1次的数, `twos`仅保留出现了2次的数, 而那些出现了3次的数都被移除了.

这种方法可以扩展为通用方法, 适用于任何仅存在一个只出现了`M`次的数, 其他数都出现了`K`次的数组, 如`K = 4, M = 3`
```java
public int singleNumber(int[] nums) {
    int ones = 0, twos = 0, threes = 0;
    for (int c : nums)
    {
        ones = (ones ^ c) & ~twos & ~threes;
        twos = (twos ^ c) & ~ones & ~threes;
        threes = (threes ^ c) & ~twos & ~ones;
    }
    return threes;
}

public static void main(String[] args) {
    int[] nums = {1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    System.out.println(singleNumber(nums)); // 2
}
```

## 找出唯二的仅出现M次的数
[LeetCode原题](https://leetcode.com/problems/single-number-iii/):给定一个整数数组`nums`，其中恰好有两个元素只出现一次，其余所有元素均出现两次。 找出只出现一次的那两个元素。跟前面的问题类似, 我们需要再次使用XOR来解决这个问题。通过分割数组, 把出现一次的两个数, 划分到不同的数组中, 问题就转化为寻找唯一的出现一次的数问题. 所以关键就是如何拆分数组.

具体需要两次遍历：
* 第一次遍历，对数组所有元素进行异或，获得要找的两个数字的XOR。由于两个数字是不同的，因此在XOR结果中必定有一个`set bit`, 即位值为'1'的位。
* 找出任意`set bit`（如最右边的）。
* 第二次遍历，将所有数字分成两组: 一组为具有上述`set bit`的数, 另一组没有。按照这种方法分组, 相同的数字一定会被分配到同一组中, 而两个只出现一次的数会分配到不同数组中。

```java
/**代码来自: https://leetcode.com/problems/single-number-iii/discuss/68900/Accepted-C%2B%2BJava-O(n)-time-O(1)-space-Easy-Solution-with-Detail-Explanations
*/
public class Solution {
    public int[] singleNumber(int[] nums) {
        // Pass 1 :
        // Get the XOR of the two numbers we need to find
        int diff = 0;
        for (int num : nums) {
            diff ^= num;
        }
        // Get its last set bit
        diff &= -diff;

        // Pass 2 :
        int[] rets = {0, 0}; // this array stores the two numbers we will return
        for (int num : nums)
        {
            if ((num & diff) == 0) // the bit is not set
            {
                rets[0] ^= num;
            }
            else // the bit is set
            {
                rets[1] ^= num;
            }
        }
        return rets;
    }
}
```

