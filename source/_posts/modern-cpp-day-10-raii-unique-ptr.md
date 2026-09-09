---
title: C++ Day 10 - RAII & unique_ptr
categories:
  - 学习笔记
  - C++
tags:
  - C++
  - Modern C++
  - RAII
  - unique_ptr
  - Ownership
description: 以异常安全和多态 Layer 管理为线索，掌握 RAII、unique_ptr 与唯一所有权。
readmore: true
abbrlink: b85144b7
date: 2026-09-03 09:00:00
updated: 2026-09-09 23:41:00
---
> **学习信息**
> **学习日期：** 2026-09-03（周四）
> **重点：** RAII、异常安全、唯一所有权、make_unique、多态对象管理
> **所属计划：** 14 天 C++ 学习计划 · Day 10
> **前置笔记：** C++ Day 2 - Const Correctness, Lifetime & Dynamic Memory、C++ Day 9 - Move Semantics

![lock_guard 以对象作用域自动释放锁的 RAII 示例](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@main/img/cs106l-2026/day10-raii-lockguard.png)

> 图源：Stanford CS106L Spring 2026，[RAII & Smart Pointers Slides](https://web.stanford.edu/class/cs106l/lectures/2026Spring-16-RAII-SmartPointers.pdf) 第 44 页。`lock_guard` 把锁的获取与释放绑在对象生命周期上，是 RAII 的典型例子。


<!-- more -->
## 今日目标

- [x] 用对象生命周期解释 RAII。
- [x] 理解异常发生时 stack unwinding 仍会析构局部对象。
- [x] 掌握 unique_ptr 的唯一所有权、不可复制、可移动。
- [x] 用 vector<unique_ptr<Layer>> 管理多态 Layer。

## 1. RAII 的核心

```text
构造对象 → 获取资源
对象存活 → 使用资源
析构对象 → 释放资源
```

RAII = Resource Acquisition Is Initialization。资源不只包括 heap memory：

```text
memory / file / socket / mutex / GPU handle / database connection
```

它的本质不是“智能指针语法”，而是让资源释放由 C++ 的确定性析构来保证。

## 2. 为什么 RAII 能穿过异常路径

危险的手动协议：

```cpp
auto* p = new Pet{};
do_something();  // 可能 throw
delete p;
```

若中途抛异常，delete 不会运行。

```cpp
void work() {
    std::ifstream input{"data.txt"};
    std::lock_guard<std::mutex> guard{mutex};
    do_something();  // 即使 throw，局部对象仍会在 unwinding 中析构
}
```

```text
throw
  ↓
stack unwinding
  ↓
局部 RAII 对象按逆序析构
  ↓
file close / mutex unlock / memory release
```

## 3. unique_ptr 表达唯一 Owner

![Smart Pointer 用 RAII 表达资源所有权](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@img/img/cs106l-2026-day10-smart-pointers-v2.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 16](https://web.stanford.edu/class/cs106l/lectures/2026Spring-16-RAII-SmartPointers.pdf)，第 56 页。

```cpp
auto layer = std::make_unique<ConvLayer>();
```

它表示：当前这一个 unique_ptr 是该对象唯一的所有者；离开作用域时自动 delete。

```cpp
std::unique_ptr<Layer> a = std::make_unique<ConvLayer>();
// auto b = a;              // 错误：不能 copy
auto b = std::move(a);      // 正确：ownership 转移
```

Move 后 `a` 为空，新 Owner 是 `b`。

## 4. 在 Network 中管理多态 Layer

```cpp
class Network {
public:
    void add(std::unique_ptr<Layer> layer) {
        layers_.push_back(std::move(layer));
    }

private:
    std::vector<std::unique_ptr<Layer>> layers_;
};
```

```text
Network
  └─ vector
       ├─ unique_ptr → ConvLayer
       ├─ unique_ptr → ReLULayer
       └─ unique_ptr → PoolLayer
```

Network 销毁时，vector 销毁每个 unique_ptr；每个指针再正确销毁实际 Derived Object。因此 Base Class 要提供 virtual destructor。

## 5. 为什么优先 make_unique

优点：

- 不重复写类型。
- 所有权语义清晰。
- 避免裸 new 暴露在调用点。
- 更容易组合在异常安全代码中。

## 6. 易错点

| 易错认识 | 正确理解 |
| --- | --- |
| RAII 只管理内存 | 任何需要释放的资源都可封装 |
| 有 unique_ptr 就不需要 virtual destructor | 通过 unique_ptr<Base> 管理 Derived 时仍需要 |
| unique_ptr 不能传参 | 可以按值传入并 std::move 接管 |
| release() 是普通取值操作 | 它放弃所有权，应少用且明确谁负责释放 |

## 7. 过关自测

- [x] 能说明 RAII 为什么在异常时仍能清理资源。
- [x] 看到 unique_ptr<Layer> 能说出“唯一 Owner”。
- [x] 能解释不可 Copy、可 Move 的原因。
- [x] 能画出 Network 对 Layer 的 ownership。
- [x] 能解释 make_unique 与裸 new 的设计差异。

> **Day 10 完成**
> 已掌握 RAII 与 unique_ptr 的概念模型；工程中继续用它验证多态资源生命周期。

## 下一步

> C++ Day 11 - shared_ptr, weak_ptr & Ownership：当唯一所有权不适用时，如何表达共享与观察？
