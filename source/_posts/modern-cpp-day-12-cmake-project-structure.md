---
title: C++ Day 12 - CMake & C++ Project Structure
categories:
  - 学习笔记
  - C++
tags:
  - C++
  - CMake
  - Build System
  - 工程化
description: 从编译链接流程到 Target 设计，搭建可维护的多文件 CMake C++ 项目。
readmore: true
abbrlink: 2e8b54aa
date: 2026-09-05 09:00:00
updated: 2026-09-09 23:41:00
---
> **学习信息**
> **学习日期：** 2026-09-05（周六）
> **重点：** 编译流程、Header/Source、Target、CMake、Out-of-source Build
> **所属计划：** 14 天 C++ 学习计划 · Day 12
> **前置笔记：** C++ Day 6 - Class Templates、C++ Day 10 - RAII & unique_ptr

![CMake 是生成原生构建系统的跨平台工具](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@main/img/cs106l-2026/day12-cmake.png)

> 图源：Stanford CS106L Spring 2026，[RAII & Smart Pointers Slides](https://web.stanford.edu/class/cs106l/lectures/2026Spring-16-RAII-SmartPointers.pdf) 第 91 页。CMake 负责描述 Target 与依赖，并生成 Makefile 或 Ninja 等原生构建文件；它不是编译器。


<!-- more -->
## 今日目标

- [x] 解释预处理、编译、汇编、链接。
- [x] 区分 header 与 source 的职责。
- [x] 理解 CMake、Make/Ninja 与编译器的关系。
- [x] 写出最小 CMake Target 结构。
- [x] 理解模板定义通常保留在 Header 的原因。

## 1. 从源文件到可执行文件

```text
main.cpp
  ↓ preprocessing
translation unit
  ↓ compilation / assembly
main.o

tensor.cpp
  ↓ preprocessing / compilation / assembly
tensor.o

main.o + tensor.o
  ↓ linking
mini_app
```

编译器驱动程序通常依次完成预处理、编译与汇编，把每个 Translation Unit 变为 Object File；Linker 再把多个 Object File 与库连接为 Executable。

## 2. Header 与 Source

| 文件 | 主要内容 |
| --- | --- |
| .hpp / .h | 声明、接口、类型定义、模板定义 |
| .cpp | 普通函数与非模板成员函数实现 |

```text
include/
  tensor.hpp
src/
  tensor.cpp
main.cpp
```

模板调用点在编译时需要看到完整定义，所以 Template Definition 通常留在 hpp 或 tpp 中；普通函数可以只在 hpp 声明、在 cpp 实现，再由 linker 连接。

## 3. CMake、Make 与编译器

```text
CMakeLists.txt
  ↓ cmake configure
Makefile / Ninja build files
  ↓ make / ninja
g++ / clang++
  ↓
executable
```

```text
compiler = 真正编译代码
make/ninja = 根据依赖决定构建顺序与增量重编译
cmake = 生成具体构建系统的跨平台配置工具
```

## 4. 最小多文件工程

```text
MiniInfer/
├── CMakeLists.txt
├── include/
│   └── tensor.hpp
├── src/
│   └── tensor.cpp
└── main.cpp
```

```cmake
cmake_minimum_required(VERSION 3.20)
project(MiniInfer LANGUAGES CXX)

add_library(miniinfer src/tensor.cpp)
target_include_directories(miniinfer PUBLIC include)
target_compile_features(miniinfer PUBLIC cxx_std_20)

add_executable(mini_app main.cpp)
target_link_libraries(mini_app PRIVATE miniinfer)
```

Target 是现代 CMake 的中心：include path、language standard 和 link dependency 都尽量附着在真实 target 上，而不是散落在全局变量里。

## 5. Out-of-source Build

```bash
cmake -S . -B build
cmake --build build
./build/mini_app
```

源目录保持干净；生成的 cache、object file 与 executable 都进入 build 目录。需要重新配置或切换构建类型时，也更容易清理和复现。

## 6. Makefile 的位置

Makefile 也能描述编译依赖：

```makefile
main: main.o tensor.o
	g++ main.o tensor.o -o main

main.o: main.cpp
	g++ -c main.cpp -o main.o
```

它的关键价值是增量构建：只改 tensor.cpp 时，重编 tensor.o 后再链接，无须全量重编。CMake 的价值是用更高层、跨平台的方式生成此类规则。

## 7. 易错点

| 易错认识 | 正确理解 |
| --- | --- |
| CMake 是编译器 | 它生成构建系统；编译器仍是 g++ / clang++ |
| Header 都只放声明 | 模板定义通常必须可见 |
| include directory 是全局环境设置 | 应尽量通过 target_include_directories 传播 |
| build 目录可与源目录混在一起 | 可行但不利于清理、复现和版本控制 |

## 8. 过关自测

- [x] 能画出 cpp 到 executable 的编译和链接路径。
- [x] 能说明 hpp 与 cpp 的职责。
- [x] 能解释 CMake 与 Make 的关系。
- [x] 能写出 add_library、add_executable、target_link_libraries。
- [x] 能说明模板为什么通常定义在 Header。

> **Day 12 完成**
> 已建立多文件工程与构建系统的心智模型。后续以 MiniInfer 和 KuiperInfer 的真实 CMake Target 继续验证。

## 下一步

> C++ Day 13 - KuiperInfer Source Reading：把前 12 天的 C++ 机制映射到真实推理框架。
