---
title: 20260827-20260909 C++ 学习计划
categories:
  - 学习笔记
  - C++
tags:
  - C++
  - Modern C++
  - 学习计划
  - AI工程
description: 14 天完成面向 AI/CV 工程的 Modern C++ 第一轮学习：从对象模型、STL 到 RAII、CMake 与 MiniInfer。
readmore: true
abbrlink: 2e7b53dc
date: 2026-08-26 08:00:00
updated: 2026-09-09 23:41:00
---
> **最终进度**
> Day 1～14 已完成。前半程因 Iterator、Class、Inheritance 掌握较快而压缩；节省的时间投入了 Copy / Move、RAII、智能指针、CMake、KuiperInfer 与 MiniInfer。基础阶段测试为 29 / 30，最终综合测试为 35 / 35。


<!-- more -->
## 最终目标

在 2026-09-09 前完成 **AI/CV 工程所需的核心 Modern C++ 第一轮学习**。

目标不是穷尽 C++ 标准，而是能够：

- 阅读 KuiperInfer、OpenCV、TensorRT 风格的 C++ 代码；
- 判断对象的访问方式、生命周期和 Ownership；
- 理解 Template、Polymorphism、Copy / Move 与 RAII；
- 使用 CMake 组织一个小型 C++ 工程；
- 完成 100～300 行的 MiniInfer；
- 达到退出条件后停止系统刷课，转入项目驱动学习。

## 能力进度

- [x] Pointer / Reference / Iterator 基础
- [x] Const Correctness
- [x] Scope / Lifetime / Stack / Heap
- [x] `new/delete` 与 Dangling Pointer
- [x] Sequence Container 与 Vector Reallocation
- [x] Class / Constructor / Destructor / `this`
- [x] Inheritance / Virtual / Polymorphism
- [ ] Class Template / Function Template
- [ ] Lambda / Algorithm / Associative Container 补丁
- [ ] Special Member Functions / Copy
- [ ] Move Semantics
- [ ] RAII / Smart Pointer / Ownership
- [ ] CMake / C++ Project Structure
- [ ] KuiperInfer 源码阅读
- [ ] MiniInfer / Final Test

## 为什么调整计划

原计划前半程是：

```text
Day 3  Sequence Containers
Day 4  Associative Containers
Day 5  Iterator + Algorithm
Day 6  Classes
Day 7  Inheritance
```

实际进度是：

```text
Day 1  Pointer / Reference / Iterator
Day 2  Const / Lifetime
Day 3  Containers + Iterator
Day 4  Classes
Day 5  Inheritance / Polymorphism
```

调整原则：

1. Day 1 已完整学过新版 L6，不再单独重复 Iterator；
2. Associative Container 与 Algorithm 保留，但压缩到 Day 7 作为补丁；
3. Classes 与 Inheritance 以测试结果为准，视为已经通过；
4. 节省的时间用于 CMake、KuiperInfer 和毕业项目；
5. 计划服务学习，不为了维持旧编号而降低速度。

## 固定学习资源

### Stanford CS106L

**课程主页：** [Stanford CS106L Spring 2026](https://web.stanford.edu/class/cs106l/)

后续只看与新计划相关的内容：

| Day | 资源 | 观看策略 |
|---:|---|---|
| 6 | 2026 L9、旧 L6 | Const 快速过，重点看 Class Template |
| 7 | 2026 L10 / L11、旧 L7 / L8 | Function、Lambda、Algorithm；关联容器只补基本用法 |
| 8 | 旧 L12 | 完整看，不倍速赶 |
| 9 | 旧 L13 | 完整看，不压缩 |
| 10～11 | 旧 L16、2026 L16 | RAII 与 Smart Pointer |

### 工程实践

```text
Mini examples
      ↓
CMake project
      ↓
KuiperInfer
      ↓
MiniInfer
```

KuiperInfer 按以下顺序阅读：

```text
Tensor → Layer → Operator → Runtime → Conv
```

关联已有笔记：

- 第1课 深度学习推理框架基础
- 第2课 张量的设计
- 第3课 计算图的设计
- 第4课 计算图的构建
- 第5课 算子和算子注册器的设计与实现

## 执行规则

### 时间伸缩

| 当天时间 | 视频 / Slides | Coding | 执行要求 |
|---:|---:|---:|---|
| 30 分钟 | 20 分钟 | 10 分钟 | 只做核心概念与最小实验 |
| 60 分钟 | 35～40 分钟 | 20～25 分钟 | 完成主线例程 |
| 90 分钟 | 50～55 分钟 | 35～40 分钟 | 完成例程和过关题 |
| 120 分钟 | 70～80 分钟 | 40～50 分钟 | 学习、实验、复盘全部完成 |

> **硬规则**
> 可以没看完视频，但不能连续两天完全不写代码。总体保持约 **视频 65% / Coding 35%**。

### 每日三问

1. **我能解释吗？** 为什么需要这个概念？
2. **我能写吗？** 不看资料写出最小示例。
3. **我能在工程里认出来吗？** 在真实代码中指出它的作用。

三个问题至少通过两个才能进入下一 Day；否则只补测薄弱点，不重刷整节课。

## 新版总览

| 状态  | 实际/建议日期     |    Day | 主题                                     | 核心产出                  |
| --- | ----------- | -----: | -------------------------------------- | --------------------- |
| [x] | 08-27 周四       |      1 | Pointer / Reference / Iterator         | 参数传递与 Iterator 心智模型   |
| [x] | 08-28 周五       |      2 | Const / Lifetime / Dynamic Memory      | 生命周期与悬空访问实验           |
| [x] | 08-29 周六（压缩完成） |      3 | Containers + Vector                    | Scores 统计与扩容模型        |
| [x] | 08-29 周六（提前完成） |      4 | Classes + Const Correctness            | `Tensor` Class        |
| [x] | 08-29 周六（提前完成） |      5 | Inheritance + Polymorphism             | `Layer` 多态体系          |
| [ ] | 08-30 周日       |      6 | Class Template                         | `Tensor<T>`           |
| [ ] | 08-31 周一       |      7 | Function Template / Lambda / Algorithm | Algorithm + STL 补丁    |
| [ ] | 09-01 周二       |      8 | Special Member Functions               | Deep Copy 实验          |
| [ ] | 09-02 周三       |      9 | Move Semantics                         | Move Constructor 实验   |
| [ ] | 09-03 周四       |     10 | RAII + `unique_ptr`                    | 独占 Ownership Layer    |
| [ ] | 09-04 周五       |     11 | `shared_ptr` + `weak_ptr`              | 引用计数与 Cycle 实验        |
| [ ] | 09-05 周六       |     12 | CMake + Project Structure              | 多文件 C++ 工程            |
| [ ] | 09-06 周日       | Buffer | 补缺 / 复测                                | 不新增必学主题               |
| [ ] | 09-07 周一～09-08 周二 |     13 | KuiperInfer                            | Tensor → Runtime 源码阅读 |
| [ ] | 09-09 周三       |     14 | MiniInfer + Final Test                 | 项目、构建与毕业考试            |

## 已完成：Day 1～5

### Day 1｜Pointer / Reference / Iterator ✅

**笔记：** C++ Day 1 - Pointer, Reference & Iterator

- [x] `T`、`T*`、`T&`、`const T&`
- [x] `p`、`*p`、`&p` 与 `T**`
- [x] Pass by Value / Reference / Pointer
- [x] `begin()`、`end()` 与 Iterator 基础
- [x] 小测：100 / 100

### Day 2｜Const / Lifetime / Dynamic Memory ✅

**笔记：** C++ Day 2 - Const Correctness, Lifetime & Dynamic Memory

- [x] `const int*`、`int* const`、`const int* const`
- [x] Const Reference 与 Const Member Function
- [x] Scope / Lifetime / Stack / Heap
- [x] `new/delete`、Memory Leak、Dangling Pointer
- [x] Resource Constructor / Destructor 实验

### Day 3｜Containers + Iterator ✅

**笔记：** C++ Day 3 - Containers & Iterator

- [x] `vector`、`string` 与常用接口
- [x] `size`、`capacity`、`reserve`、Reallocation
- [x] Iterator / Pointer / Reference Invalidation
- [x] 20 个 Scores 的最大值、最小值与平均值

### Day 4｜Classes + Const Correctness ✅

**笔记：** C++ Day 4 - Classes & Const Correctness

- [x] `public/private`、Constructor、Destructor、`this`
- [x] Member Initializer List
- [x] Const Member Function
- [x] `T&` / `const T&` Const Overload
- [x] `Tensor` Class

### Day 5｜Inheritance + Polymorphism ✅

**笔记：** C++ Day 5 - Inheritance & Polymorphism

- [x] Base / Derived 与 Upcast
- [x] `virtual`、`override`、Dynamic Dispatch
- [x] Pure Virtual Function 与 Abstract Class
- [x] Virtual Destructor
- [x] Object Slicing
- [x] `Layer / Conv / ReLU` 多态实验

## 已完成：Day 6～14

### Day 6｜08-30｜Class Template ✅

**笔记：** C++ Day 6 - Class Templates

#### 学习

- [ ] 看 2026 L9 — Class Templates & Const Correctness
- [ ] Const 部分快速复习，不重复精学
- [ ] 看旧 L6 — Templates
- [ ] 理解 Template 是编译期生成具体类型的蓝图
- [ ] 暂不学习 SFINAE、复杂模板元编程和高级 `type_traits`

#### Coding：`Tensor<T>`

- [ ] 把固定 `float` 的 `Tensor` 改为 `template<typename T>`
- [ ] 创建 `Tensor<float>`、`Tensor<int>`、`Tensor<double>`
- [ ] 保留 `size() const` 与 Const Overload
- [ ] 观察 Template Definition 为什么通常放在 Header

```cpp
template<typename T>
class Tensor {
public:
    explicit Tensor(std::size_t size)
        : data_(size) {}

    T& at(std::size_t index) {
        return data_.at(index);
    }

    const T& at(std::size_t index) const {
        return data_.at(index);
    }

private:
    std::vector<T> data_;
};
```

#### 过关

- [ ] 从内向外解释 `Tensor<float>`
- [ ] 解释 Template 与普通 Class 的关系
- [ ] 不看资料写出最小 Class Template

**复盘：**

- 实际用时：
- 卡点：
- 一句话总结：

---

### Day 7｜08-31｜Function Template + Lambda + Algorithm ✅

**笔记：** C++ Day 7 - Lambda, Algorithms & Associative Containers

这一天同时补回原计划中被压缩的 Associative Container 与 Algorithm，不再单独占两天。

#### 学习

- [ ] 看旧 L7 + L8，按已掌握部分倍速
- [ ] 按课程实际标题看 2026 L10 / L11 的 Function、Lambda、Algorithm 内容
- [ ] 掌握 Function Template
- [ ] 掌握 Lambda 的最小语法与 `[]`、`[&]`、`[=]`
- [ ] 掌握 `sort`、`find`、`transform`

#### Associative Container 补丁（20～30 分钟）

- [ ] 知道 `map`、`unordered_map`、`set` 的核心区别
- [ ] 用 `unordered_map<string, int>` 建立 `class_ids`
- [ ] 用 `find()` 查询，不存在时输出提示
- [ ] 知道 `operator[]` 查询失败时会插入默认 Value
- [ ] 用 `conv / relu / pool / linear` 实现最小 Operator Registry

#### Coding

- [ ] 实现 Function Template `add(T a, T b)`
- [ ] 用 `sort()` 对 Day 3 Scores 升序和降序排列
- [ ] 用 `find()` 查找指定 Score
- [ ] 用 `transform()` 生成归一化 Scores
- [ ] 用 Lambda 编写降序比较或转换逻辑

#### 过关

- [ ] 解释 Iterator Range 如何连接 Container 与 Algorithm
- [ ] 解释 `auto`、`auto&`、`const auto&`
- [ ] 根据需求选择 `vector` 或 `unordered_map`
- [ ] 独立拆解 Lambda 的 Capture、Parameter 与 Body

**复盘：**

- 实际用时：
- 卡点：
- 一句话总结：

---

### Day 8｜09-01｜Special Member Functions ✅

**笔记：** C++ Day 8 - Special Member Functions & Copy Semantics

#### 学习

- [ ] 完整看旧 L12，不倍速赶
- [ ] 理解 Default Constructor、Destructor
- [ ] 理解 Copy Constructor 与 Copy Assignment
- [ ] 理解 Shallow Copy / Deep Copy
- [ ] 理解 Rule of 3 / Rule of 5 / Rule of 0

#### Coding

- [ ] 编写一个拥有动态资源的最小 Class
- [ ] 实现 Copy Constructor
- [ ] 实现 Copy Assignment，并处理 Self-assignment
- [ ] 打印 Constructor / Copy / Assignment / Destructor 调用时机
- [ ] 证明 Deep Copy 后两个对象可独立修改

#### 过关

- [ ] 解释两个对象拥有同一个裸 Pointer 的风险
- [ ] 解释为什么标准 Container 有助于实现 Rule of 0
- [ ] 区分 Copy Constructor 与 Copy Assignment 的触发场景

**复盘：**

- 实际用时：
- 卡点：
- 一句话总结：

---

### Day 9｜09-02｜Move Semantics ✅

**笔记：** C++ Day 9 - Move Semantics

#### 学习

- [ ] 完整看旧 L13，不压缩
- [ ] 区分 `T&` 与 `T&&`
- [ ] 理解 Lvalue / Rvalue 的工程直觉
- [ ] 理解 `std::move()` 只进行类型转换，不直接搬内存
- [ ] 理解 Move Constructor 与 Move Assignment

#### Coding

- [ ] 为 Day 8 的 Class 增加 Move Constructor
- [ ] 增加 Move Assignment
- [ ] 打印并比较 Copy 与 Move 调用路径
- [ ] 验证 Moved-from Object 仍可析构和重新赋值

#### 过关

- [ ] 用自己的话解释“Copy 复制资源，Move 转移资源”
- [ ] 解释为什么 `std::move()` 后不能假设原值不变
- [ ] 看到 `std::move(layer)` 能联想到 Ownership Transfer

**复盘：**

- 实际用时：
- 卡点：
- 一句话总结：

---

### Day 10｜09-03｜RAII + `unique_ptr` ✅

**笔记：** C++ Day 10 - RAII & unique_ptr

#### 学习

- [ ] 看旧 L16 的 RAII / `unique_ptr` 部分
- [ ] 理解资源 Lifetime 绑定对象 Lifetime
- [ ] 理解 Ownership
- [ ] 掌握 `unique_ptr` 与 `make_unique`

#### Coding

- [ ] 把 `new Conv()` / `delete` 改为 `make_unique<Conv>()`
- [ ] 用 `unique_ptr<Layer>` 指向 Derived Object
- [ ] 用 `vector<unique_ptr<Layer>>` 管理多个 Layer
- [ ] 用 `std::move()` 把 Ownership 交给 Network
- [ ] 观察离开作用域时的自动析构

#### 过关

- [ ] 看到 `unique_ptr<Layer>` 能立刻说出“唯一 Owner”
- [ ] 解释它为什么不能 Copy、为什么可以 Move
- [ ] 解释 RAII 如何消除手动 `delete`

**复盘：**

- 实际用时：
- 卡点：
- 一句话总结：

---

### Day 11｜09-04｜`shared_ptr` + `weak_ptr` ✅

**笔记：** C++ Day 11 - shared_ptr, weak_ptr & Ownership

#### 学习

- [ ] 继续旧 L16，并补 2026 L16
- [ ] 掌握 `shared_ptr`、`make_shared`、Reference Count
- [ ] 掌握 `weak_ptr`、`lock()`、`expired()`
- [ ] 理解 Shared Ownership 与 Cycle

```text
unique_ptr → 一个 Owner
shared_ptr → 多个 Owner
weak_ptr   → 不拥有，只观察
```

#### Coding

- [ ] 用 `make_shared` 创建对象并观察 `use_count()`
- [ ] 在嵌套作用域中复制 `shared_ptr`
- [ ] 用 `weak_ptr::lock()` 安全访问对象
- [ ] 释放所有 Owner 后检查 `expired()`
- [ ] 画出一个 Shared Cycle，并说明如何用 `weak_ptr` 破环

#### 过关

- [ ] 从内向外拆解 `vector<shared_ptr<Tensor<float>>>`
- [ ] 根据 Ownership 选择 `unique_ptr` / `shared_ptr` / `weak_ptr`
- [ ] 解释 Reference Count 为什么不能自动解决 Cycle

**复盘：**

- 实际用时：
- 卡点：
- 一句话总结：

---

### Day 12｜09-05｜CMake + C++ Project Structure ✅

**笔记：** C++ Day 12 - CMake & C++ Project Structure

这是新版计划新增的完整工程日。

#### 学习

- [ ] 理解 `.hpp` / `.cpp` 的职责
- [ ] 理解 `include/` / `src/` / `build/` 目录
- [ ] 掌握最小 `CMakeLists.txt`
- [ ] 掌握 `add_executable()` 与 `add_library()`
- [ ] 掌握 `target_include_directories()`
- [ ] 了解 `target_link_libraries()`
- [ ] 使用 Out-of-source Build

#### Coding：多文件项目

```text
CppProject/
├── CMakeLists.txt
├── include/
│   └── tensor.hpp
├── src/
│   └── tensor.cpp
└── main.cpp
```

- [ ] 把一个非 Template Class 拆成 Header 与 Source
- [ ] 用 `add_library()` 建立 Library Target
- [ ] 用 `add_executable()` 建立 Executable Target
- [ ] 配置 Include Directory 与 Link Dependency
- [ ] 执行 Configure、Build、Run

```bash
cmake -S . -B build
cmake --build build
./build/mini_app
```

#### 过关

- [ ] 解释为什么 Template Definition 通常留在 Header
- [ ] 解释 Target 比全局编译选项更清晰的原因
- [ ] 能从零写出最小多文件 CMake Project

**复盘：**

- 实际用时：
- 卡点：
- 一句话总结：

---

### Buffer｜09-06｜补缺与复测

> **不新增必学主题**
> 只处理 Day 6～12 中未通过的过关条件；若全部通过，休息或提前搭建 MiniInfer 目录。

- [ ] 列出仍不能独立解释的三个概念
- [ ] 只重做失败的最小实验
- [ ] 检查 CMake Project 能否从干净 `build/` 重新构建
- [ ] 更新 #Modern C++ 检查点

---

### Day 13｜09-07～09-08｜KuiperInfer ✅

**笔记：** C++ Day 13 - KuiperInfer Source Reading

> **禁止从仓库入口逐文件乱读**
> 按 `Tensor → Layer → Operator → Runtime → Conv` 追踪类型、接口与 Ownership。

#### 阅读顺序

- [ ] Tensor：Template、Container、Const Interface、数据成员
- [ ] Layer：Abstract Interface、Virtual Destructor、Polymorphism
- [ ] Operator：参数结构与 Registry
- [ ] Runtime：对象创建、保存、传递与销毁
- [ ] Conv：Derived Implementation 与 Forward Entry

#### 每个模块只回答四个问题

1. 这个 Type 负责什么？
2. 谁拥有它？
3. 数据如何传入和返回？
4. 哪些 C++ 机制出现在这里？

#### 语法拆解

- [ ] `const shared_ptr<Tensor<float>>&`
- [ ] `virtual Status Forward(...) = 0`
- [ ] `std::move(layer)`
- [ ] `vector<unique_ptr<Layer>>`
- [ ] 找到一个 Registry 或 Factory 的实际实现

#### 输出

- [ ] 为 Tensor、Layer、Operator、Runtime 各写 3～5 句阅读记录
- [ ] 画出一条对象 Ownership / Data Flow
- [ ] 记录最多五个需要项目中反查的问题

---

### Day 14｜09-09｜MiniInfer + Final Test ✅

**笔记：** C++ Day 14 - MiniInfer & Final Review

> **今天禁止继续刷 C++ 视频**
> 只完成工程、构建、运行和毕业测试。

#### 项目结构

```text
MiniInfer/
├── CMakeLists.txt
├── include/
│   ├── tensor.hpp
│   ├── layer.hpp
│   └── network.hpp
├── src/
│   └── network.cpp
└── main.cpp
```

#### 最小功能

- [ ] `Tensor<T>`：保存一维数据，提供 `size()` 和 Const Overload
- [ ] `Layer`：Pure Virtual `forward(Tensor<float>&)` + Virtual Destructor
- [ ] `ReLU`：把负数置零
- [ ] `Network`：用 `vector<unique_ptr<Layer>>` 管理 Layer
- [ ] `Network::add()`：通过 Move 接收 Ownership
- [ ] `Network::forward()`：按顺序执行所有 Layer
- [ ] CMake：能够 Configure、Build、Run

验收输入：

```text
[-2.0, -0.5, 0.0, 1.5, 3.0]
```

验收输出：

```text
[0.0, 0.0, 0.0, 1.5, 3.0]
```

## Modern C++ 检查点

Day 11 后，不看资料解释：

- [ ] `template<typename T>`
- [ ] Function Template
- [ ] Lambda Capture
- [ ] Copy Constructor / Copy Assignment
- [ ] `T&&` / `std::move`
- [ ] RAII / Ownership
- [ ] `unique_ptr`
- [ ] `shared_ptr`
- [ ] `weak_ptr`
- [ ] Shared Cycle

**目标：** 至少 `8/10`，未通过项在 Buffer Day 补测。

## 最终毕业考试

每项要求“能解释 + 能写最小示例”。达到 **12/15** 即结束系统刷课。

| 内容 | 当前状态 |
|---|:---:|
| Pointer | ✅ |
| Reference | ✅ |
| const | ✅ |
| Stack / Heap | ✅ |
| Lifetime | ✅ |
| vector | ✅ |
| unordered_map | ⬜ |
| iterator | ✅ |
| Class | ✅ |
| Polymorphism | ✅ |
| Template | ⬜ |
| Copy | ⬜ |
| Move | ⬜ |
| RAII | ⬜ |
| Smart Pointer | ⬜ |

**当前进度：** `9/15` <br>
**毕业要求：** `12/15`，且 MiniInfer 能通过 CMake 构建运行。

> **退出条件**
> 达到毕业要求后，立刻停止系统学习 C++，进入：
>
> `KuiperInfer → OpenCV C++ → PyTorch / ONNX → TensorRT C++ → CUDA`
>
> 以后遇到不会的 C++，按问题反查 LearnCpp / CS106L，不重新从头刷课。

## 最终路线

```text
前五天：语言与对象模型                ✅
       ↓
Day 6～7：Template / Function / STL 补丁
       ↓
Day 8～11：Copy / Move / RAII / Ownership
       ↓
Day 12：CMake 与多文件工程
       ↓
Day 13：KuiperInfer 真实源码
       ↓
Day 14：MiniInfer 与毕业测试
       ↓
停止系统刷 C++，进入 AI/CV 项目驱动学习
```
