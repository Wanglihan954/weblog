---
title: C++ Day 8 - Special Member Functions & Copy Semantics
categories:
  - 学习笔记
  - C++
tags:
  - C++
  - Modern C++
  - Copy
  - Rule of Five
description: 区分 copy constructor 与 copy assignment，理解 shallow/deep copy 和 Rule of 3/5/0。
readmore: true
abbrlink: 2ab8fe9f
date: 2026-09-01 09:00:00
updated: 2026-09-09 23:41:00
---
> **学习信息**
> **学习日期：** 2026-09-01（周二）
> **重点：** Copy Constructor、Copy Assignment、Shallow/Deep Copy、Rule of 3/5/0
> **所属计划：** 14 天 C++ 学习计划 · Day 8
> **前置笔记：** C++ Day 4 - Classes & Const Correctness、C++ Day 2 - Const Correctness, Lifetime & Dynamic Memory

![指针成员复制需要明确深拷贝语义](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@main/img/cs106l-2026/day8-deep-copy.png)

> 图源：Stanford CS106L Spring 2026，[Special Member Functions Slides](https://web.stanford.edu/class/cs106l/lectures/2026Spring-13-SpecialMemberFunctions.pdf) 第 40 页。资源型成员若只复制地址，会让多个对象错误地共享同一份资源。


<!-- more -->
## 今日目标

- [x] 区分 Copy Constructor 与 Copy Assignment。
- [x] 解释裸指针成员的 Shallow Copy 风险。
- [x] 理解 Deep Copy、自赋值和资源独立性。
- [x] 记住 Rule of 3、Rule of 5 与 Rule of 0 的适用场景。

## 1. 特殊成员函数地图

```cpp
Widget();                          // Default Constructor
Widget(const Widget&);             // Copy Constructor
Widget& operator=(const Widget&);  // Copy Assignment
~Widget();                         // Destructor
Widget(Widget&&);                  // Move Constructor
Widget& operator=(Widget&&);       // Move Assignment
```

本节先聚焦 Copy 相关成员；Move 在 C++ Day 9 - Move Semantics 继续。

## 2. “出生时复制”与“出生后赋值”

```cpp
Tensor b = a;  // b 正在初始化：Copy Constructor

Tensor b;
b = a;         // b 已存在：Copy Assignment
```

```text
源对象 a
  ├─ 创建新对象 b 时复制 ─→ Copy Constructor
  └─ 给已有对象 b 赋值 ───→ Copy Assignment
```

Copy Assignment 通常返回 T&，从而支持链式赋值。

## 3. Shallow Copy 为什么危险

```cpp
class Buffer {
    int* data_ = nullptr;
    std::size_t size_ = 0;
};
```

若默认复制只是复制地址：

```text
a.data_ ─┐
         ├──→ 同一块 heap resource
b.data_ ─┘
```

问题是两者都可能认为自己拥有资源：修改会互相影响、析构后会留下 dangling pointer、两次 delete[] 会触发 undefined behavior。

## 4. Deep Copy 与 Self-assignment

Deep Copy 的目标：

```text
a.data_ ─→ resource A
b.data_ ─→ resource B（内容相同、资源独立）
```

```cpp
Buffer& operator=(Buffer other) { // 参数副本已完成 Deep Copy
    using std::swap;
    swap(data_, other.data_);
    swap(size_, other.size_);
    return *this;
}
```

这是 Copy-and-swap：先取得资源独立的副本，再交换资源。临时对象析构时会释放旧资源，自赋值也自然安全。

> **> 若采用传统的 `const Buffer&` 赋值实现，必须处理 `a = a`，并避免“先释放自己、再从同一个对象读取”的错误。**

## 5. Rule of 3 / 5 / 0

| 规则 | 含义 |
| --- | --- |
| Rule of 3 | 自己管理资源并声明析构、复制构造或复制赋值之一时，通常要考虑三者 |
| Rule of 5 | 再加入 move constructor 与 move assignment |
| Rule of 0 | 优先让 vector、string、智能指针等 RAII 成员管理资源 |

```cpp
class Tensor {
private:
    std::vector<float> data_;
};
```

这种设计通常天然接近 Rule of 0。

## 6. 禁止复制

```cpp
class UniqueHandle {
public:
    UniqueHandle(const UniqueHandle&) = delete;
    UniqueHandle& operator=(const UniqueHandle&) = delete;
};
```

这比“让编译器在深处报错”更清晰。

## 7. 易错点

| 易错认识 | 正确理解 |
| --- | --- |
| Tensor b = a 是 assignment | 对新对象初始化，调用 Copy Constructor |
| 默认 copy 总是安全 | 裸 pointer 的默认 copy 常是浅拷贝 |
| Deep Copy 就是多写一份 new | 还要保证异常安全、析构和赋值语义正确 |
| Rule of 5 要死背 | 它提醒资源所有权操作必须成套设计 |
| 所有类都要手写 Rule of 5 | 优先 Rule of 0 |

## 8. 过关自测

- [x] 能判断 Copy Constructor 与 Copy Assignment 的触发时机。
- [x] 能画出 Shallow Copy 导致 double free 的对象关系。
- [x] 能解释 Self-assignment guard。
- [x] 能说明为什么 vector 成员倾向于 Rule of 0。
- [x] 能写出禁止复制的 delete 声明。

> **Day 8 完成**
> 本日按摸底结果压缩学习；核心目标是建立“复制的是值还是资源所有权”的判断，而非机械背诵六个函数。

## 下一步

> C++ Day 9 - Move Semantics：当资源不必复制时，如何安全地转移所有权？
