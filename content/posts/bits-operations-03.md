title: 位操作 - 风骚的走位操作
date: 2017-09-22
mathjax: true
categories:
- CS
tags:
- Bitwise Operation
- Software Engineer
- Computer Science
- Algorithms
---
通过位移实现很多风骚的操作， 参考[这个视频](https://www.youtube.com/watch?v=7jkIUgLC29I)。
<!-- more -->
检查一个数是否是偶数, 本质上就是取最后一位来判断, 如果是1那么就一定是奇数, 反之则为偶数:
```python
(x & 1) == 0
```

Check if power of two:
```python
(x & x - 1) == 0
```
因为如果数`x`是以2底的真数, 那么其二进制一定只有一个位置是`1`, 如`0b1000`, 那么`x-1`就会变成只有该位置是`0`其右边所有位变为`1`, 即`0b0111`, 也就是说这种情况下`x`和`x-1`所有位置都互异. 那么它们的位与运算就是`x & x - 1 = 0b0000`.

`x & x - 1`的广义用途是求`x`二进制中`1`的个数, [Counting bits set](https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan):
```
unsigned int v; // count the number of bits set in v
unsigned int c; // c accumulates the total bits set in v
for (c = 0; v; c++)
{
  v &= v - 1; // clear the least significant bit set
}
```
> Brian Kernighan's algorithm takes `O(log N)` to count set bits (1s) in an integer: each iteration sets the least significance bit that isn't zero to zero - and only it. Since each iteration converts exactly bit from 1 to 0, it'll take as many iterations as there are non-0 bits to convert all the bits to 0(and thus `v == 0` and the loop finishes). An integer n has `log(n)` bits, hence the worst case is `O(log(n))`

如果一个整数不为`0`, 那么其二进制就至少有一个`1`. 假设最右边一位是`1`, 那么减`1`就会把最后一位变为`0`, 前面所有位保持不变. 假如最后一位是`0`, 那么最靠右的`1`假设在`m`位置, 那么减去`1`, 该位置会变为`0`, 而其右边的所有`0`都会变为`1`, 其左边的所有位不变. `v &= v - 1`把最右的`1`变为`0`.

获取二进制的最后一个`1`:
```python
def last_set_bit(x):
    y = ~(x - 1) # = - (x - 1) - 1 = -x
    return x & y
```
假设最右边的`1`位于n, `-1`操作会把n右边所有`0`变为`1`, 而n位变为`0`. 接着`~`操作会把n左边所有位翻转, 而n及其右边的数会变为原来的样子, 也就是n为`1`, 右边全为`0`(或者没有右边). 最后`&`操作就只剩下n位的`1`和右边的`0`(如果有的话).
