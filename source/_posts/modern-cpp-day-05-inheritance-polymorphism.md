---
title: C++ Day 5 - Inheritance & Polymorphism
categories:
  - 学习笔记
  - C++
tags:
  - C++
  - Modern C++
  - 继承
  - 多态
  - virtual
description: 用 Layer、Conv 与 ReLU 的例子理解继承、虚函数、动态多态和 object slicing。
readmore: true
abbrlink: 96a2e48
date: 2026-08-29 11:00:00
updated: 2026-09-09 23:41:00
---
> **学习信息**
> **学习日期：** 2026-08-29（周六，提前完成）
> **重点：** Inheritance、`virtual/override`、Abstract Class、Runtime Polymorphism、Virtual Destructor、Object Slicing
> **所属计划：** 14 天 C++ 学习计划 · Day 5
> **前置笔记：** C++ Day 4 - Classes & Const Correctness


<!-- more -->
## 今日目标

- [x] 理解 Base Class 与 Derived Class 的 Is-a 关系
- [x] 理解 Base Pointer/Reference 与 Upcast
- [x] 掌握 `virtual`、`override` 与 Dynamic Dispatch
- [x] 掌握 Pure Virtual Function 与 Abstract Class
- [x] 理解 Virtual Destructor 和 Object Slicing

## 1. Inheritance 与 Upcast

![Inheritance Tree 表达 is-a 关系](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@main/img/20260829-cs106l-l08-inheritance-tree.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 8 — Inheritance](https://web.stanford.edu/class/cs106l/lectures/2026Spring-08-Inheritance.pdf)

图中的 `Player is an Actor`、`Actor is an Entity`，对应推理框架里的 `Conv is a Layer`。

Public Inheritance 用类型系统表达 Is-a 关系，让 Derived 对象可以通过统一的 Base Interface 使用。

```cpp
Conv conv;
Layer* pointer = &conv;
Layer& reference = conv;
```

Derived 对象可以粗略理解为包含一个 Base 子对象：

```text
Conv Object
┌──────────────────┐
│ Layer Base Part  │
├──────────────────┤
│ Conv Own Part    │
└──────────────────┘
```

`Layer* pointer = &conv` 让 Base Pointer 指向其中的 Base 子对象。这种从 Derived 到 Base 的转换称为 Upcast。

反方向不能自动成立：

```cpp
Layer layer;
// Conv* pointer = &layer; // 编译错误
```

原因是：

```text
Conv is a Layer  → 成立
Layer is a Conv  → 不一定成立
```

## 2. Static Binding 与 Dynamic Dispatch

```cpp
#include <iostream>

class Base {
public:
    void print() const {
        std::cout << "Base\n";
    }
};

class Derived : public Base {
public:
    void print() const {
        std::cout << "Derived\n";
    }
};

Derived derived;
Base* pointer = &derived;
pointer->print(); // Base
```

非 Virtual Call 根据表达式的静态类型解析，因此这里调用 `Base::print()`；Derived 的同名函数只是隐藏 Base 函数，并未形成 Runtime Polymorphism。

![virtual 开启动态分派，override 明确重写关系](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@main/img/20260829-cs106l-l08-virtual-functions.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 8 — Inheritance](https://web.stanford.edu/class/cs106l/lectures/2026Spring-08-Inheritance.pdf)

`virtual` 让调用根据对象的 Dynamic Type 进行分派：

```text
表达式静态类型 → Base*
对象动态类型   → Derived
Virtual Call   → Derived::print()
```

Base Pointer 提供统一接口，Virtual Function 则让调用在运行时选择实际 Derived 行为。

## 3. Runtime Polymorphism 与 Abstract Class

```cpp
class Layer {
public:
    virtual void forward() = 0;
    virtual ~Layer() = default;
};

class Conv : public Layer {
public:
    void forward() override {
        std::cout << "Conv forward\n";
    }
};

class ReLU : public Layer {
public:
    void forward() override {
        std::cout << "ReLU forward\n";
    }
};
```

```cpp
Conv conv;
ReLU relu;

Layer* layers[] = {&conv, &relu};

for (Layer* layer : layers) {
    layer->forward();
}
```

输出：

```text
Conv forward
ReLU forward
```

同一个 Base Interface 根据对象的动态类型产生不同运行行为，这就是 Runtime Polymorphism。`Layer` 含有 Pure Virtual Function，因此不能直接实例化。

> **> `vptr/vtable` 是 Dynamic Dispatch 的常见实现模型，不是 C++ 标准规定的唯一实现方式。**

## 4. `override` 与 Pure Virtual Function

```cpp
class Base {
public:
    virtual void foo(int x);
};

class Derived : public Base {
public:
    void foo(int x) override;
};
```

`override` 表示：

> 这个函数必须重写某个 Base Virtual Function，请编译器验证签名。

```cpp
class Derived : public Base {
public:
    // void foo() override; // 编译错误，签名不匹配
};
```

`virtual` 声明动态接口；`override` 明确 Derived 接入该接口，并让编译器验证签名。

![Pure Virtual Function、Abstract Class 与 Concrete Class](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@main/img/20260829-cs106l-l08-pure-virtual.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 8 — Inheritance](https://web.stanford.edu/class/cs106l/lectures/2026Spring-08-Inheritance.pdf)

`virtual void forward() = 0` 表示 Base 只规定 Interface，由 Derived 提供实现；`= 0` 不是“返回 0”。

Derived 若未实现所有 Pure Virtual Function，也仍是 Abstract Class。

## 5. 为什么推理框架使用 Abstract Base Class

```text
Layer
 ├── Conv
 ├── ReLU
 ├── Pooling
 ├── Linear
 └── Softmax
```

Network 只依赖统一接口：

```cpp
for (Layer* layer : layers) {
    layer->forward();
}
```

它不需要知道每个对象具体属于哪个 Derived Class。

新增类型：

```cpp
class Attention : public Layer {
public:
    void forward() override {
        // Attention implementation
    }
};
```

只要遵守 `Layer` Interface，原有调用代码通常无需修改。

工程价值是统一接口与管理方式、降低调用方对具体 Layer 的依赖，并允许增加新类型时少改既有执行代码。

## 6. 构造、析构与 Virtual Destructor

```cpp
class Base {
public:
    Base() {
        std::cout << "Base construct\n";
    }

    ~Base() {
        std::cout << "Base destroy\n";
    }
};

class Derived : public Base {
public:
    Derived() {
        std::cout << "Derived construct\n";
    }

    ~Derived() {
        std::cout << "Derived destroy\n";
    }
};
```

创建 `Derived`：

```text
Base Constructor
      ↓
Derived Constructor
```

销毁 `Derived`：

```text
Derived Destructor
      ↓
Base Destructor
```

可以理解为：先建地基，再建上层；销毁时先拆上层，再拆地基。

多态 Base Class 的经典写法：

```cpp
class Layer {
public:
    virtual void forward() = 0;
    virtual ~Layer() = default;
};
```

目的：

> 允许通过 Base Pointer 正确销毁实际的 Derived 对象。

```cpp
Layer* layer = new Conv();
delete layer;
```

当 Destructor 为 Virtual 时：

```text
动态类型是 Conv
      ↓
~Conv()
      ↓
~Layer()
```

如果 Base Destructor 不是 Virtual，通过 Base Pointer 删除 Derived 对象会产生 Undefined Behavior。

> 如果 Class 被设计为通过 Base Pointer 多态使用并销毁，Base Destructor 必须是 Virtual。

## 7. Object Slicing

```cpp
Derived derived;
Base base = derived;
```

Value Copy 只保留 Base Part，Derived 特有部分被切掉，这称为 Object Slicing。

![使用 Base Pointer 避免复制导致的 Object Slicing](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@main/img/20260829-cs106l-l08-object-slicing.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 8 — Inheritance](https://web.stanford.edu/class/cs106l/lectures/2026Spring-08-Inheritance.pdf)

Reference 或 Pointer 不复制对象，因此不会发生 Slicing：

```cpp
Base& reference = derived;
Base* pointer = &derived;
```

若函数是 Virtual，通过它们调用仍会分派到 Derived Implementation。

## 8. 代码实验

### Experiment 1｜Virtual 与非 Virtual 调用

```cpp
#include <iostream>

class Base {
public:
    void normal() const {
        std::cout << "Base normal\n";
    }

    virtual void dynamic() const {
        std::cout << "Base dynamic\n";
    }

    virtual ~Base() = default;
};

class Derived : public Base {
public:
    void normal() const {
        std::cout << "Derived normal\n";
    }

    void dynamic() const override {
        std::cout << "Derived dynamic\n";
    }
};

int main() {
    Derived derived;
    Base* pointer = &derived;

    pointer->normal();  // Base normal
    pointer->dynamic(); // Derived dynamic
}
```

输出应为 `Base normal` 与 `Derived dynamic`：非 Virtual Call 依据静态类型解析，Virtual Call 根据对象动态类型分派。

## 9. 易错点

1. `Derived is a Base` 成立，不代表任意 `Base is a Derived`。
2. Derived 中出现同名函数不等于形成 Runtime Polymorphism，Base Interface 必须是 Virtual。
3. `override` 不是多余装饰，它能发现函数签名不匹配。
4. Pure Virtual 的 `= 0` 不代表返回数值 0。
5. `vptr/vtable` 是常见实现直觉，不是语言标准对 Virtual 的定义。
6. 多态 Base Class 若通过 Base Pointer 删除对象，Destructor 必须是 Virtual。
7. `Base value = derived` 会发生 Object Slicing；Base Pointer/Reference 不复制完整对象。
8. 当前示例中的 `vector<Layer*>` 不拥有对象；后续应使用 Smart Pointer 明确 Ownership。

## 10. 自测

- [x] 为什么 Base Pointer 可以指向 Derived 对象？
- [x] 非 Virtual 与 Virtual Call 分别依据什么选择函数？
- [x] `override` 能帮助编译器检查什么？
- [x] `virtual void forward() = 0` 表达什么？
- [x] 为什么 Abstract Class 不能直接实例化？
- [x] 为什么多态 Base Destructor 通常必须是 Virtual？
- [x] Object Slicing 在什么情况下发生？

> **一句话总结**
> Inheritance 建立类型层级，`virtual + override` 提供动态行为，Polymorphism 让框架通过统一 Base Interface 管理不同 Derived 对象。

## 下一步

> 学习 Template 后，将 `Tensor` 从固定的 `float` 数据类型泛化为 `Tensor<T>`；学习 Smart Pointer 后，再用明确的 Ownership 替换示例中的裸 `Layer*`。
