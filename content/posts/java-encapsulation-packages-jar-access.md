---
title: Java 封装, 包, JAR, 权限控制
date: 2017-05-29
author: "Cong Chan"
tags: ['Java']
---
## Encapsulation
封装是面向对象编程的基本原则之一，也是程序员处理复杂性一个方法。管理复杂性是编写大型程序时必须面对的主要挑战之一。
<!-- more -->
对抗复杂性的一些工具包括:
* Hierarchical abstraction: 创建一个个具有明确的 abstraction barriers 的抽象层
    * **Abstraction Barriers**：使用`private`, 保证对象内部不能被查看, 确保底层的复杂性不会暴露给外部世界。
* “Design for change” (D. Parnas)
    * Organize program around objects.
    * Let objects decide how things are done.
    * **Hide information** others don’t need.

大概的想法都是 - 程序应该被构建成模块化，可互换的片段，可以在不破坏系统的情况下进行交换。

封装就是构建在这种对外部隐藏信息的概念上。以细胞为类比：细胞内部可能非常复杂，由染色体，线粒体，核糖体等组成，但它完全被封装在一个单一模块中 - 抽象了内部的复杂性。

> Module: A set of methods that work together as a whole to perform some task or set of related tasks.
> Encapsulated: A module is said to be encapsulated if its implementation is completely hidden, and it can be accessed only through a documented interface.

## Packages
同样功能的类可能有多种版本, 或者不同类刚好命名相同。通过 packages 来为每个 classes 提供唯一的标识名称，如`java.util.`
> A package is a namespace that organizes classes and interfaces.

在IntelliJ的操作：
* 创建 package：
1, File → New Package
2, 选择 package name (i.e. “ug.joshh.animal”)

* 给 Package 添加(新) Java 文件：
1, 右键 package name
2, New → Java Class
3, 命名 class, 然后 IntelliJ 会自动把文件放进正确的路径, 并添加 package declaration.

* 移动其他`.java`文件到 Package
1, 在文件顶部声明 `package [packagename]`
2, 将文件存储在（移动到）与 package name 对应的文件夹中：如`ug.joshh.animal` 对应`ug/joshh/animal`文件路径.

注意, 不存在`sub-package`这种概念, 即`ug.joshh.Animal`和`ug.joshh.Plant`是完全不同的.

Package好处: Organizing, making things package private
坏处: Specific

### Default packages
没有在文件顶部明确指明 package name 的Java类默认属于 default package 的一部分。

一般而言, Java文件应该以明确的 package 声明开头以避免将文件留在 default package 中（除非它是一个非常小的示例程序）。因为来自 default package 的代码无法 import，并且可能会意外地在 default package 下创建相同名称的类。

## JAR
一般情况下，程序会包含多个`.class`文件。如果想共享此程序，可以把压缩成一个`.jar`文件，此`.jar`文件将包含程序所有`.class`文件以及其他附加信息。JAR文件就像zip文件一样, 可以将文件解压缩回`.java`文件。JAR文件并不会加密保护代码.
> Creating a JAR File (IntelliJ)
>1. Go to File → Project Structure → Artifacts → JAR → “From modules with dependencies”
>2. Click OK a couple of times
>3. Click Build → Build Artifacts (this will create a JAR file in a folder called “Artifacts”)
>4. Distribute this JAR file to other Java programmers, who can now import it into IntelliJ (or otherwise)

## 权限控制
cs61b Josh Hug:
`Private`
> Only code from the given class can access private members.

`Package Private`
> The default access given to Java members if there is no explicit modifier written. Classes that belong in the same package can access, but not subclasses!

`Protected`
> Classes within the same package and subclasses can access these members, but the rest of the world (e.g. classes external to the package or non-subclasses) cannot! Subtypes might need it, but subtype clients will not.

`Public`
> Open and promised to the world, once deployed, the public members’ signatures should not change.
就像承诺和合同，尽量不要更改，以便用户始终可以（用已有的代码）访问。如果开发者要舍弃某一个`Public`，一般标识为`deprecated`.

细节:
1. **Access is Based Only on Static Types**
2. 接口的方法默认是`public`的
