---
title: 信息处理 - 数据压缩
date: 2017-10-10
mathjax: true
author: "Cong Chan"
tags: ['Algorithms', 'Information Retrieval', 'Data Compression']
---
## 数据压缩
压缩数据以节省储存空间，节省传输时间。同时很多文件都有很多冗余信息，这为压缩提供了很多可能性。
<!-- more -->
通用文件压缩
·文件：GZIP，BZIP，7z
·Archivers：PKZIP
·文件系统：NTFS，HFS +，ZFS

多媒体
·图像：GIF，JPEG
·声音：MP3
·视频：MPEG，DivX™，HDTV

通讯
·ITU-T T4 Group 3 Fax
·V.42bis调制解调器
·Skype

数据库

### 压缩率
`Compression ratio = Bits in Compressed B / bits in B`.
> 自然语言的压缩率为50-75％或更高.

### 读写二进制
```
public class BinaryStdIn {
    boolean readBoolean() // read 1 bit of data and return as a boolean value
    char readChar() // read 8 bits of data and return as a char value
    char readChar(int r) // read r bits of data and return as a char value
    // similar methods for byte (8 bits); short (16 bits); int (32 bits); long and double (64 bits)
    boolean isEmpty() // is the bitstream empty?
    void close() // close the bitstream
}

public class BinaryStdOut {
    void write(boolean b) // write the specified bit
    void write(char c) // write the specified 8-bit char
    void write(char c, int r) // write the r least significant bits of the specified char
    // similar methods for byte (8 bits); short (16 bits); int (32 bits); long and double (64 bits)
    void close() // close the bitstream
}
```
比如使用三种方法表达`12/31/1999`
1, A character stream (StdOut),
```java
StdOut.print(month + "/" + day + "/" + year);
```
`00110001` 1
`00110010` 2
`00101111` /
`00110111` 3
`00110001` 1
`00101111` /
`00110001` 1
`00111001` 9
`00111001` 9
`00111001` 1
共 80bits
2, Three ints (BinaryStdOut)
```java
BinaryStdOut.write(month);
BinaryStdOut.write(day);
BinaryStdOut.write(year);
```
`00000000 00000000 00000000 00001100` 12
`00000000 00000000 00000000 00011111` 31
`00000000 00000000 00000111 11001111` 1999
共96bits
3，A 4-bit field, a 5-bit field, and a 12-bit field (BinaryStdOut)
```java
BinaryStdOut.write(month, 4);
BinaryStdOut.write(day, 5);
BinaryStdOut.write(year, 12);
```
`1100` 12
`11111` 13
`0111110 01111` 1999
共21bits

### 通用数据压缩算法？
不存在的，因为假如真的存在一种可以压缩所有比特串的算法，那么该算法就可以继续压缩已经被它压缩过的数据，那意味着所有比特串可以被压缩为0比特.

## Run-length encoding
Simple type of redundancy in a bitstream. Long runs of repeated bits：
`0000000000000001111111000000011111111111`
Compression, 4-bit counts to represent alternating runs of 0s and 1s: 15 0s, then 7 1s, then 7 0s, then 11 1s.
`1111 0111 0111 1011`
<!-- more -->
```java
public class RunLength
{
    // maximum run-length count
   private final static int R    = 256;
   // number of bits per count
   private final static int LG_R = 8;

   /**
     * Reads a sequence of bits from standard input; compresses
     * them using run-length coding with 8-bit run lengths; and writes the
     * results to standard output.
     */
   public static void compress()
   {
        char run = 0;
        boolean old = false;
        while (!BinaryStdIn.isEmpty()) {
            boolean b = BinaryStdIn.readBoolean();
            if (b != old) {
                BinaryStdOut.write(run, LG_R);
                run = 1;
                old = !old;
            }
            else { // 如果长度超过最大值, 写入0
                if (run == R-1) {
                    BinaryStdOut.write(run, LG_R);
                    run = 0;
                    BinaryStdOut.write(run, LG_R);
                }
                run++;
            }
        }
        BinaryStdOut.write(run, LG_R);
        BinaryStdOut.close();
   }

   /**
     * Reads a sequence of bits from standard input (that are encoded
     * using run-length encoding with 8-bit run lengths); decodes them;
     * and writes the results to standard output.
     */
   public static void expand()
   {
      boolean bit = false;
      while (!BinaryStdIn.isEmpty())
      {
         int run = BinaryStdIn.readInt(lgR);
         for (int i = 0; i < run; i++)
            BinaryStdOut.write(bit);
         bit = !bit;
      }
      BinaryStdOut.close();
   }
}
```
