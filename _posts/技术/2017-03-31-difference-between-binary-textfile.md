---
layout: post
comments: true
title: 二进制文件和文本文件的区别
category: 技术
tags: Tips
keywords: 二进制文件,文本文件
description: 
---
### 1. 起因
   今天在实验室谈论到bjson数据格式，从而引出是binary和textfile的区别，我第一感觉是binary文件省空间，但是又说不清为什么...


### 2.实验

首先定义结构体

```
struct Student
{
    int num;
    char name[20];
    float score;
};
```

将stud写入到二进制文件

```
void write_to_binary_file(){
    char test_file_name[100] = "test1.dat";
    FILE *fo = fopen(test_file_name, "wb");
    struct Student stdu;
    stdu.num = 111;
    strcpy(stdu.name,"lei");
    stdu.score = 80.0f;
    fwrite(&stdu, sizeof(struct Student), 1, fo);
    fclose(fo);
}
```

将stud写入到文本文件

```
void write_to_text_file(){
    char test_file_name[100] = "test2.dat";
    FILE *fo = fopen(test_file_name, "wb");
    struct Student stdu;
    stdu.num = 111;
    strcpy(stdu.name,"lei");
    stdu.score = 80.0f;
    fprintf(fo, "%d%s%f",stdu.num,stdu.name,stdu.score);
    fclose(fo);
}
```

### 3. 使用HEXDUMP分析实验结果
hexdump命令一般用来查看“二进制”文件的十六进制编码，但实际上它的用途不止如此，手册页上的说法是“ascii, decimal, hexadecimal, octal dump”，它能查看任何文件，而不只限于二进制文件。另外还有 xxd 和 od 也可以做类似的事情。在程序输出二进制格式的文件时，常用hexdump来检查输出是否正确。当然也可以使用Windows上的UltraEdit32之类的工具查看文件的十六进制编码，但Linux上有现成的工具，何不拿来用呢

#### 3.1 常用参数
>hexdump -C -n length -s skip file_name

>-C 定义了导出的格式；-s skip 指定了从文件头跳过多少字节，或者说是偏移量，默认是十进制，如果是0x开头，则是十六进制；-n 指定了导出多少长度

>如果要看到较理想的结果，推荐使用-C参数，显示结果分为三列（文件偏移量、字节的十六进制、ASCII字符）。


#### 3.2 实验结果和结论

![二进制结果](/public/img/binary-text.png "二进制结果")

二进制文件里面将111编码成6f，1个字节，这刚好是111的16进制表示，而文本文件中则写成31，31，31用了3个字节，表示111。6c   65   69  表示lei，之后2进制文件里是几个连续的00，而文本文件中是38   30......文本文件将浮点数80.000000用了38(表示8)   30(表示0)  2E(表示.)   30(表示0)   30(表示0)   30(表示0)   30(表示0)   30(表示0)   30(表示0)，二进制文件用了4个字节表示浮点数00   00   A0   42
通过这里我们可以初见端倪了，二进制将数据在内存中的样子原封不动的搬到文件中，文本格式则是将每一个数据转换成字符写入到文件中，他们在大小上，布局上都有着区别。由此可以看出，2进制文件可以从读出来直接用，但是文本文件还多一个“翻译”的过程。

### 4. 资料
[文本文件和二进制文件的区别？请举例说明。](https://www.zhihu.com/question/19971994)

[文本文件与二进制文件区别](http://www.cnblogs.com/zhangjiankun/archive/2011/11/27/2265184.html)

[浅谈二进制文件读写和文本文件读写的区别](http://www.cppblog.com/yg2362/archive/2012/07/12/182956.html)