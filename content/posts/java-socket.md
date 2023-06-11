---
title: Java 套接字Socket
date: 2017-02-26
author: "Cong Chan"
tags: ['Java']
---
## Socket
网络上的两个程序通过一个双向的通讯连接实现数据的交换，这个双向链路的一端称为一个Socket，确切的说Socket是个代表两台机器之间网络连接的对象。Socket是TCP/IP协议的一个十分流行的编程界面，一个Socket由一个IP地址和一个端口号唯一确定。
`Socket socket = new Socket("192.168.2.1", 5000)`, 第一个参数是IP地址, 第二个参数是端口. Socket连接的建立代表两台机器之间存有对方的信息, 包括网络地址和TCP端口号.
<!-- more -->
但是，Socket所支持的协议种类也不光TCP/IP一种，因此两者之间是没有必然联系的。在Java环境下，Socket编程主要是指基于TCP/IP协议的网络编程。

使用BufferedReader从Socket上读取数据：
1. 建立对服务器的Socket连接`Socket socket = new Socket("192.168.2.1", 5000)`
2. 建立连接到Socket上低层输入串流的InputStreamReader `InputStreamReader stream = new InputStreamReader(socket.getInputStream());`, 作为从低层和高层串流间的桥梁, 把来自服务器的字节转换为字符
3. 建立缓冲区BufferedReader来读取
```java
BufferedReader reader = new BufferedReader(stream);
String msg = reader.readLine();
```
用PrintWriter写数据到Socket上:
1. 建立对服务器的Socket连接
2. 建立链接到Socket的PrintWriter `PrintWriter writer = new PrintWriter(socket.getOutputStream())`, 作为字符数据和字节间的转换桥梁, 可以衔接Strings和Socket两端
3. 写入数据
```java
writer.println("You have a message");
```

### TCP/IP
TCP/IP协议集包括**应用层, 传输层，网络层，网络访问层**。

应用层协议主要包括如下几个：FTP、TELNET、DNS、SMTP、NFS、HTTP。
* FTP(File Transfer Protocol）是文件传输协议，一般上传下载用FTP服务，数据端口是20H，控制端口是21H。
* Telnet服务是用户远程登录服务，使用23H端口，使用明码传送，保密性差、简单方便。
* DNS(Domain Name Service）是域名解析服务，提供域名到IP地址之间的转换，使用端口53。
* SMTP(Simple Mail Transfer Protocol）是简单邮件传输协议，用来控制信件的发送、中转，使用端口25。
* NFS（Network File System）是网络文件系统，用于网络中不同主机间的文件共享。
* HTTP(Hypertext Transfer Protocol）是超文本传输协议，用于实现互联网中的WWW服务，使用端口80。

网络接口层:
* Internet协议(IP): 数据链路层是负责接收IP数据包并通过网络发送，或者从网络上接收物理帧，抽出IP数据包，交给IP层。
* 正向地址解析协议(ARP): 通过已知的IP，寻找对应主机的MAC地址。
* 反向地址解析协议(RARP): 通过MAC地址确定IP地址。比如无盘工作站还有DHCP服务。

### TCP和UDP
TCP 协议和 UDP 协议都属于TCP/IP协议簇。也叫端到端传输协议，因为他们将数据从一个应用程序传输到另一个应用程序，而 IP 协议只是将数据从一个主机传输到另一个主机。

TCP是面向连接的通信协议，通过三次握手建立连接，通讯完成时要拆除连接，由于TCP是面向连接的所以只能用于端到端的通讯。TCP提供的是一种可靠的数据流服务，采用“带重传的肯定确认”技术来实现传输的可靠性。TCP还采用一种称为“滑动窗口”的方式进行流量控制，所谓窗口实际表示接收能力，用以限制发送方的发送速度。

UDP是面向无连接的通讯协议，UDP数据包括目的端口号和源端口号信息，
* UDP通讯时不需要接收方确认，属于不可靠的传输，可能会出现丢包现象，实际应用中要求程序员编程验证。
* 不保证可靠交付，不保证顺序，因此主机不需要维持复杂的链接状态表
* 由于传输数据不建立连接，因此也就不需要维护连接状态，包括收发状态等，因此一台服务机可同时向多个客户机传输相同的消息。

应用：
* TCP在网络通信上有极强的生命力，例如远程连接（Telnet）和文件传输（FTP）都需要不定长度的数据被可靠地传输。但是可靠的传输是要付出代价的，对数据内容正确性的检验必然占用计算机的处理时间和网络的带宽，因此TCP传输的效率不如UDP高。
* UDP操作简单，而且仅需要较少的监护，因此通常用于局域网高可靠性的分散系统中client/server应用程序。例如视频会议系统，并不要求音频视频数据绝对的正确，只要保证连贯性就可以了，这种情况下显然使用UDP会更合理一些。

### TCP连接与断开
最初两端的TCP进程都处于CLOSED关闭状态，客户A主动打开连接，而服务器B被动打开连接, 过程类似于`想给你发数据可以吗？ - 可以，现在发？ - 对，请准备接收”`。过程是**三次握手**：
1. 起初两端都处于CLOSED关闭状态，Client将标志位SYN置为1，随机产生一个值`seq=x`，并将该数据包发送给Server，Client进入SYN-SENT状态，等待Server确认；
2. Server收到数据包后由标志位`SYN=1`得知Client请求建立连接，Server将标志位SYN和ACK都置为1，`ack=x+1`，随机产生一个值`seq=y`，并将该数据包发送给Client以确认连接请求，Server进入SYN-RCVD状态，此时操作系统为该TCP连接分配TCP缓存和变量；
3. Client收到确认后，检查ack是否为`x+1`，ACK是否为1，如果正确则将标志位ACK置为1，`ack=y+1`，并且此时操作系统为该TCP连接分配TCP缓存和变量，并将该数据包发送给Server，Server检查ack是否为`y+1`，ACK是否为1，如果正确则连接建立成功，Client和Server进入ESTABLISHED状态，完成三次握手，随后Client和Server就可以开始传输数据。

之所以不可以仅靠二次握手, 主要为了防止已失效的连接请求报文段突然又传送到了B(因为延误等原因)，因而产生错误, 故A还要发送一次确认.

断开连接需要**四次挥手**:
1. A的应用进程先向其TCP发出连接释放报文段（`FIN=1`，序号`seq=u`），并停止再发送数据，主动关闭TCP连接，进入FIN-WAIT-1（终止等待1）状态，提出停止TCP连接的请求, 等待B的确认。
2. B收到连接释放报文段后即发出确认报文段，（`ACK=1`，确认号`ack=u+1`，序号`seq=v`），确认来路方向上的TCP连接将关闭, B进入CLOSE-WAIT（关闭等待）状态，此时的TCP处于半关闭状态，A到B的连接释放。A收到B的确认后，进入FIN-WAIT-2（终止等待2）状态，等待B发出的连接释放报文段。
3. B已经没有要向A发出的数据，B再提出反方向的关闭请求, B发出连接释放报文段（`FIN=1`，`ACK=1`，序号`seq=w`，确认号`ack=u+1`），B进入LAST-ACK（最后确认）状态，等待A的确认。
4. A收到B的连接释放报文段后，对此发出确认报文段（`ACK=1`，`seq=u+1`，`ack=w+1`），A进入TIME-WAIT（时间等待）状态。此时TCP未释放掉，需要经过时间等待计时器设置的时间2 Maximum Segment Lifetime (MSL)后，A才进入CLOSED状态。

总的流程图![](/images/tcp.png)
其中三个比较重要的状态
* `SYN_RECV` ：服务端收到建立连接的SYN没有收到ACK包的时候处在SYN_RECV状态。有两个相关系统配置. 这些处在SYNC_RECV的TCP连接称为**半连接**，并存储在内核的半连接队列中，在内核收到对端发送的ack包时会查找半连接队列，并将符合的requst_sock信息存储到完成三次握手的连接的队列中，然后删除此半连接。大量SYNC_RECV的TCP连接会导致半连接队列溢出，这样后续的连接建立请求会被内核直接丢弃，这就是**SYN Flood**攻击。
    1. `net.ipv4.tcp_synack_retries ：INTEGER` 默认值是5. 对于远端的连接请求SYN，内核会发送SYN + ACK数据报，以确认收到上一个 SYN连接请求包。这是所谓的三次握手(threeway handshake)机制的第二个步骤。这里决定内核在放弃连接之前所送出的 SYN+ACK 数目。不应该大于255，默认值是5，对应于`180秒`左右时间。通常我们不对这个值进行修改，因为我们希望TCP连接不要因为偶尔的丢包而无法建立。
    2. `net.ipv4.tcp_syncookies` 一般服务器都会设置 `net.ipv4.tcp_syncookies=1`来防止**SYN Flood**攻击。假设一个用户向服务器发送了SYN报文后突然死机或掉线，那么服务器在发出SYN+ACK应答报文后是无法收到客户端的ACK报文的（第三次握手无法完成），这种情况下服务器端一般会重试（再次发送SYN+ACK给客户端）并等待一段时间后丢弃这个未完成的连接，这段时间的长度我们称为`SYN Timeout`，一般来说这个时间是分钟的数量级（大约为30秒-2分钟）。
* `CLOSE_WAIT`: 被动关闭的server收到FIN后，但未发出ACK的TCP状态是CLOSE_WAIT。出现这种状况一般都是由于server端代码的问题，如果你的服务器上出现大量CLOSE_WAIT，应该要考虑检查代码。CLOSE_WAIT状态什么时候终结， 取决于应用程序什么时候来close socket, 从理论上来讲，只要被动关闭端不断电，进程不退出， 那么CLOSE_WAIT状态就会一直持续下去。因此理论上CLOSE_WAIT的最大时间可以达到无限长。
* `TIME_WAIT`: 发起socket主动关闭的一方 socket将进入TIME_WAIT状态。TIME_WAIT状态将持续2个MSL(Max Segment Lifetime),在Windows下默认为4分钟，即240秒。TIME_WAIT状态下的socket不能被回收使用. 具体现象是对于一个处理大量短连接的服务器,如果是由服务器主动关闭客户端的连接，将导致服务器端存在大量的处于TIME_WAIT状态的socket， 甚至比处于Established状态下的socket多的多,严重影响服务器的处理能力，甚至耗尽可用的socket，停止服务。TIME_WAIT是TCP协议用以保证被重新分配的socket不会受到之前残留的延迟重发报文影响的机制,是必要的逻辑保证。

### 端口
面向连接服务TCP协议和无连接服务UDP协议使用16bits端口号来表示和区别网络中的不同应用程序，网络层协议IP使用特定的协议号（TCP 6，UDP 17）来表示和区别传输层协议。任何TCP/IP实现所提供的服务都是1-1023之间的端口号，这些端口号由IANA分配管理, 作为保留端口供特定服务使用。其中，低于255的端口号保留用于公共应用；255到1023的端口号分配给各个公司，用于特殊应用；对于高于1023的端口号，称为临时端口号，IANA未做规定。不同程序无法共享一个端口, 因此新程序只能使用空闲端口.

常用的保留TCP端口号有：
HTTP 80，FTP 20/21，Telnet 23，SMTP 25，DNS 53等。
常用的保留UDP端口号有：
DNS 53，BootP 67（server）/ 68（client），TFTP 69，SNMP 161等。

每个TCP报文头部都包含**源端口号source port**和**目的端口号destination port**，用于标识和区分源端设备和目的端设备的应用进程。在TCP/IP协议栈中，源端口号和目的端口号分别与源IP地址和目的IP地址组成套接字，唯一的确定一条TCP连接。
相对于TCP报文，UDP报文只有少量的字段：源端口号、目的端口号、长度、校验和等，各个字段功能和TCP报文相应字段一样。

在linux一般使用netstat来查看系统端口使用情况。netstat命令的功能是显示网络连接、路由表和网络接口信息，可以让用户得知目前都有哪些网络连接正在运作。
比如查看 TCP 22 端口有两种方法：
第一种查看方法
```
[root@Demon proc]# netstat -ntlp | grep 22

tcp 0 0 0.0.0.0:22 0.0.0.0:* LISTEN 1960/sshd

tcp 0 0 0.0.0.0:3306 0.0.0.0:* LISTEN 2263/mysqld

tcp 0 0 :::22 :::* LISTEN 1960/sshd
```
第二种查看方法
```
[root@Demon proc]# lsof -i tcp:22

COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME

sshd 1960 root 3u IPv4 14435 0t0 TCP *:ssh (LISTEN)

sshd 1960 root 4u IPv6 14441 0t0 TCP *:ssh (LISTEN)
```
`-i` 显示所有网络接口的信息。

### Socket通讯的过程

Server端Listen(监听)某个端口是否有连接请求，Client端向Server 端发出Connect(连接)请求，Server端向Client端发回Accept（接受）消息。一个连接就建立起来了。Server端和Client 端都可以通过Send，Write等方法与对方通信。

对于一个功能齐全的Socket，都要包含以下基本结构，其工作过程包含以下四个基本的步骤：

　　（1） 创建Socket；

　　（2） 打开连接到Socket的输入/出流；

　　（3） 按照一定的协议对Socket进行读/写操作；

　　（4） 关闭Socket.（在实际应用中，并未使用到显示的close，虽然很多文章都推荐如此，不过在我的程序中，可能因为程序本身比较简单，要求不高，所以并未造成什么影响。）
