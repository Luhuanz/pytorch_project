# 项目目标：了解上下文的概念和使用方法


<!-- vscode-markdown-toc -->
* 1. [运行](#)
* 2. [context 上下文](#context)
* 3. [CUDA文档：](#CUDA)
* 4. [如果报错，提示nvcc错误](#nvcc)
* 5. [C++基础](#C)
	* 5.1. [#include<> 和 #include ''的区别](#includeinclude)
	* 5.2. [定义 与 使用](#-1)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->



##  1. <a name=''></a>运行
`make run`

##  2. <a name='context'></a>context 上下文
<details> <!-- 上下文 -->
<summary> 详情 </summary>

1. 设备与特定进程相关连的所有状态。比如，你写的一段kernel code对GPU的使用会造成不同状态（内存映射、分配、加载的code），Context则保存着所有的管理数据来控制和使用设备。
   - <details> <!--一个类比的例子 -->
      <summary> 一个形象的例子 </summary>
      一个类比的例子就像你与小明和小红分别对话，你与小明聊China（中国）,你与小红也聊china（瓷器），但是你们聊的可能都不是一个东西，我们管理这样的数据，所以我们创造了context。
      </details> <!--一个类比的例子 -->
   - gpu 的 context 相当于 cpu 的 program，一块gpu上可以有多个contexts，但是它们之间是相互隔离的。我们建议一块设备就一个context
     - 参考：https://dragan.rocks/articles/18/Interactive-GPU-Programming-3-CUDA-Context

2. 上下文管理可以干的事儿：
   1. 持有分配的内存列表
   2. 持有加载进该设备的kernel code
   3. cpu与gpu之间的unified memory
   5. ...

3. 如何管理上下文：
   1. 在cuda driver同样需要显示管理上下文
        - 开始时`cuCtxCreate()`创建上下文，结束时`cuCtxDestroy`销毁上下文。像文件管理一样须手动开关。
        - 用`cuDevicePrimaryCtxRetain()`创建上下文更好！
        - `cuCtxGetCurrent()`获取当前上下文
        - 可以使用堆栈管理多个上下文`cuCtxPushCurrent()`压入，`cuCtxPopCurrent()`推出
        - 对ctxA使用`cuCtxPushCurrent()`和`cuCtxCreate()`都相当于将ctxA放到栈顶（让它成为current context）
   2. cuda runtime可以自动创建，是基于`cuDevicePrimaryCtxRetain()`创建的。

</details> <!-- 上下文 -->

<br>







##  3. <a name='CUDA'></a>CUDA文档：
1. https://developer.nvidia.com/cuda-toolkit-archive
2. https://docs.nvidia.com/cuda/archive/11.2.0/

##  4. <a name='nvcc'></a>如果报错，提示nvcc错误
- 对于gcc版本大于等于8的不支持，需要修改Makefile中的g++为g++7或者更低

##  5. <a name='C'></a>C++基础（供C++小白参考）

<details> <!-- C++基础 -->
<summary> 详情 </summary>

###  5.1. <a name='includeinclude'></a>#include<> 和 #include ''的区别
- 对于#include <filename.h>，编译器先从标准库路径开始搜索filename.h，使得系统文件调用比较快
- 对于#include "filename.h"，编译器先从用户的工作路径开始搜索filename.h，后去寻找系统路径，使得自定义文件较快。
所以在写代码的过程中要根据实际情况选择是<>还是""

###  5.2. <a name='-1'></a>定义 与 使用
- 定义的时候* 和 & 指的是指针变量和引用
  - ```c++
    如下：
    int* a = nullptr;  // 指的是指向 int的指针类型
    void func(int* a); // 形参时，你传入的得是 地址/指针
    void func(int& a); // 形参时，你传入的得是 引用
    ```
- 使用的时候* 和 & 指的是取地址上的值 和 取变量的地址
  - ```c++
    如下：
    int value = *a;   // 此时a是个指针变量，取该地址的变量值
    int* ptr = &b;    // 此时b是个非指针变量，取该变量的地址
    func(*a);         
    func(&b); 
    ```
</details> <!-- C++基础 -->






