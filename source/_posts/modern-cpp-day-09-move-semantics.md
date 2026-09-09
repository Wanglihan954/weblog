---
title: C++ Day 9 - Move Semantics
categories:
  - 学习笔记
  - C++
tags:
  - C++
  - Modern C++
  - Move
  - Rvalue Reference
description: '从值类别到 move constructor，厘清 std::move、资源转移和 moved-from object。'
readmore: true
abbrlink: 8c6bcbeb
date: 2026-09-02 09:00:00
updated: 2026-09-09 23:41:00
---
> **学习信息**
> **学习日期：** 2026-09-02（周三）
> **重点：** T&&、lvalue/rvalue、Move Constructor、Move Assignment、std::move、moved-from state
> **所属计划：** 14 天 C++ 学习计划 · Day 9
> **前置笔记：** C++ Day 8 - Special Member Functions & Copy Semantics

![Copy 与 Move Constructor 的资源处理对比](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@main/img/cs106l-2026/day9-copy-move.png)

> 图源：Stanford CS106L Spring 2026，[Move Semantics Slides](https://web.stanford.edu/class/cs106l/lectures/2026Spring-14-MoveSemantics.pdf) 第 68 页。Copy 创建独立资源；Move 转交资源并使来源对象保持可析构的有效状态。


<!-- more -->
## 今日目标

- [x] 区分 Copy 与 Move 的资源语义。
- [x] 理解 T&、const T&、T&& 的工程直觉。
- [x] 解释 std::move 不会自行移动资源。
- [x] 写出 Move Constructor / Assignment 的最小所有权转移。
- [x] 正确对待 moved-from object。

## 1. Move 的目标

```text
Copy：创建一份独立资源；源对象和目标对象都完整拥有资源
Move：目标对象接管资源；源对象保留合法但未指定状态
```

```cpp
std::vector<int> a(1'000'000);
std::vector<int> b = a;            // 复制元素与存储
std::vector<int> c = std::move(a); // 通常转移内部指针、size、capacity
```

Move 的价值是避免不必要的大量数据复制，不是某条语句神奇地“搬内存”。

## 2. 值类别的实用地图

| 表达式 | 实用分类 |
| --- | --- |
| x | lvalue：有名字、通常还要继续使用 |
| T{} | prvalue：临时值 |
| std::move(x) | xvalue：可被当作将亡值处理 |

```text
lvalue  ─→ 倾向匹配 const T& ─→ copy
rvalue/xvalue ─→ 倾向匹配 T&& ─→ move
```

这只是重载解析的常见结果；是否真的移动取决于类型是否提供可用 move 操作。

## 3. std::move 是转换，不是动作

它表达的是：

> 我允许后续代码把 x 当作可移动资源。

真正改变资源的是被选中的 Move Constructor 或 Move Assignment。因此调用 std::move(x) 后，不能再假设 x 的原值保持不变；但也不能假设它已经必然被清空。

## 4. 裸资源的最小 Move Constructor

```cpp
Buffer(Buffer&& other) noexcept
    : size_(other.size_),
      data_(other.data_) {
    other.size_ = 0;
    other.data_ = nullptr;
}
```

```text
other.data_ ──→ heap data
       ↓ transfer
this->data_ ──→ heap data
other.data_ ──→ nullptr
```

把源对象置空能避免两个对象析构时重复释放同一地址。

## 5. 有名字的 T&& 仍是 lvalue

```cpp
void consume(Tensor&& value) {
    // value 有名字，因此表达式 value 是 lvalue
    Tensor next = std::move(value);
}
```

尽管 `value` 的声明类型是 `Tensor&&`，表达式 `value` 仍是 Lvalue；转交其资源时需要再次使用 `std::move(value)`。

## 6. const 与 noexcept

`const Tensor` 经 `std::move` 后得到 `const Tensor&&`；常规 `Tensor&&` Move Constructor 不能绑定它，因此通常只能走 Copy 路径。

noexcept 表示移动不抛异常。标准容器扩容时通常更愿意使用 noexcept move，因为它更容易维持异常安全保证。

## 7. 易错点

| 易错认识 | 正确理解 |
| --- | --- |
| std::move 已经完成移动 | 它只转换值类别，移动函数才转移资源 |
| moved-from object 已销毁 | 仍有效，可析构、可重新赋值 |
| T&& r 中的 r 是 rvalue | 有名字的表达式 r 是 lvalue |
| const T&& 更安全 | 常阻止需要修改源对象的 move |
| move 一定更快 | 对小对象或无资源对象未必有收益 |

## 8. 过关自测

- [x] 能用“复制资源 / 转移所有权”解释 Copy 与 Move。
- [x] 能写出四种典型构造、赋值触发路径。
- [x] 能解释 std::move 与 Move Constructor 的分工。
- [x] 能说明 moved-from state 的使用边界。
- [x] 能解释 named rvalue reference 为什么仍要 std::move。

> **Day 9 完成**
> 对话整理记录中的阶段测试为 **27 / 28（96%）**。学习重点是 Ownership Transfer，而不是把 Move 简化成“搬内存”。

## 下一步

> C++ Day 10 - RAII & unique_ptr：把资源释放从手工协议变成对象生命周期的自然结果。
