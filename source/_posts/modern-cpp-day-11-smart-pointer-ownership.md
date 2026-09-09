---
title: 'C++ Day 11 - shared_ptr, weak_ptr & Ownership'
categories:
  - 学习笔记
  - C++
tags:
  - C++
  - Modern C++
  - shared_ptr
  - weak_ptr
  - Ownership
description: 比较 unique_ptr、shared_ptr 和 weak_ptr，理解引用计数、观察关系与循环引用。
readmore: true
abbrlink: 1390f9f3
date: 2026-09-04 09:00:00
updated: 2026-09-09 23:41:00
---
> **学习信息**
> **学习日期：** 2026-09-04（周五）
> **重点：** Shared Ownership、Reference Count、weak_ptr、Cycle、复杂类型从内向外拆解
> **所属计划：** 14 天 C++ 学习计划 · Day 11
> **前置笔记：** C++ Day 10 - RAII & unique_ptr

![互相持有 shared_ptr 会形成无法释放的环](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@main/img/cs106l-2026/day11-weakptr-cycle.png)

> 图源：Stanford CS106L Spring 2026，[RAII & Smart Pointers Slides](https://web.stanford.edu/class/cs106l/lectures/2026Spring-16-RAII-SmartPointers.pdf) 第 74 页。两个对象互持 `shared_ptr` 时计数无法归零；观察关系应用 `weak_ptr` 表达。


<!-- more -->
## 今日目标

- [x] 选择 unique_ptr、shared_ptr 与 weak_ptr。
- [x] 理解 shared_ptr 的强引用计数。
- [x] 用 weak_ptr 打破 shared ownership cycle。
- [x] 安全使用 weak_ptr::lock()。
- [x] 从内向外解释复杂智能指针类型。

## 1. 三种所有权语义

| 类型 | 是否拥有对象 | 可复制 | 是否增加 strong count |
| --- | ---: | ---:| ---: |
| `unique_ptr<T>` | 是 | 否 | 无计数 |
| `shared_ptr<T>` | 是 | 是 | 是 |
| `weak_ptr<T>` | 否 | 是 | 否 |

一句话记忆：

```text
unique_ptr = 我独占
shared_ptr = 我们共同拥有
weak_ptr = 我知道它，但我不拥有
```

选择顺序应是：先问能否唯一拥有；只有确实存在共同生命周期时才选择 shared_ptr。

## 2. shared_ptr 与 Reference Count

```cpp
auto tensor = std::make_shared<Tensor<float>>();
auto alias = tensor;
```

复制 shared_ptr 不复制 Tensor，而是让两个 owner 指向同一个对象，并增加 strong reference count。

```text
tensor ─┐
        ├──→ Tensor<float>
alias  ─┘
strong count = 2
```

最后一个 shared_ptr 销毁或 reset 后，强计数归零，对象才析构。实现通常还涉及 control block，其中可保存计数、deleter 和 allocator 等元数据。

## 3. weak_ptr：观察而非拥有

```cpp
std::weak_ptr<Tensor<float>> observer = tensor;

if (auto locked = observer.lock()) {
    // locked 是临时 shared_ptr；对象仍活着
}
```

`weak_ptr` 不延长对象生命周期。不要先检查 `expired()`、再假定对象仍存在；需要访问时应直接调用 `lock()`，取得临时 `shared_ptr`。

## 4. 为什么会有 Cycle

错误设计：

```text
A --shared_ptr--> B
↑                |
|                |
└---shared_ptr---┘
```

外部 owner 消失后，A 与 B 仍互相增加对方 strong count，因此两个对象都不会释放。

修正：把不表达所有权的一条边改为 weak_ptr。

```text
A --shared_ptr--> B
↑
|
weak_ptr
```

![weak_ptr 让回指不再增加 strong reference count](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@img/img/cs106l-2026-day11-weak-ptr-v2.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 16](https://web.stanford.edu/class/cs106l/lectures/2026Spring-16-RAII-SmartPointers.pdf)，第 75 页。

判断口诀：

> 一条关系如果只表示“能访问 / 观察 / 回指”，而不表示“负责让对方活着”，就不应使用 shared_ptr。

## 5. 复杂类型拆解

```cpp
const std::shared_ptr<Tensor<float>>& input
```

从内向外：

```text
float
  ↓
Tensor<float>
  ↓
shared_ptr<Tensor<float>>
  ↓
const shared_ptr<Tensor<float>>&
```

它表示“以 const reference 传入一个 shared_ptr”，因此函数不会复制或重置该 shared_ptr。

它不等于：

```cpp
std::shared_ptr<const Tensor<float>>
```

后者才是“指向只读 Tensor 的 shared_ptr”。const 修饰的位置决定限制谁。

## 6. 与 KuiperInfer 的对象关系

推理框架中常见的合理方向：

```text
RuntimeGraph ─shared_ptr→ RuntimeOperator
RuntimeOperator ─shared_ptr→ Layer
Layer ─weak_ptr→ RuntimeOperator
```

Layer 回指 RuntimeOperator 时若只为取得上下文、不负责其生命周期，weak_ptr 可以避免双方互相拥有。

## 7. 易错点

| 易错认识 | 正确理解 |
| --- | --- |
| shared_ptr 是默认安全选择 | 默认应优先 unique_ptr；shared ownership 应有明确理由 |
| weak_ptr 可以直接访问对象 | 必须先 lock() |
| 引用计数能解决全部内存泄漏 | 无法解决 shared_ptr cycle |
| const shared_ptr<T>& 让 Tensor 只读 | 它限制 shared_ptr 句柄，不限制被指向 Tensor |

## 8. 过关自测

- [x] 能按 ownership 选择三种智能指针。
- [x] 能解释最后一个 shared_ptr 消失时发生什么。
- [x] 能画出 cycle 并用 weak_ptr 破环。
- [x] 能写出 lock() 的安全访问方式。
- [x] 能拆解 const shared_ptr<Tensor<float>>&。

> **Day 11 完成**
> 智能指针的核心不在语法，而在能否说清“谁负责让对象继续活着”。

## 下一步

> C++ Day 12 - CMake & C++ Project Structure：把对象与接口组织成可重复构建的多文件项目。
