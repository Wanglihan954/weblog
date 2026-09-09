---
title: C++ Day 4 - Classes & Const Correctness
categories:
  - 学习笔记
  - C++
tags:
  - C++
  - Modern C++
  - Class
  - const
description: 通过 Tensor 小类理解封装、构造与析构、初始化列表、this 以及 const overload。
readmore: true
abbrlink: 4c84084c
date: 2026-08-29 10:00:00
updated: 2026-09-09 23:41:00
---
> **学习信息**
> **学习日期：** 2026-08-29（周六，提前完成）
> **重点：** Class、Constructor、Destructor、`this`、Const Member Function、Const Overload
> **所属计划：** 14 天 C++ 学习计划 · Day 4
> **前置笔记：** C++ Day 2 - Const Correctness, Lifetime & Dynamic Memory、C++ Day 3 - Containers & Iterator


<!-- more -->
## 今日目标

- [x] 理解 Class 如何封装数据、操作和访问规则
- [x] 掌握 Constructor、Destructor 与 Member Initializer List
- [x] 理解 `this` 指向当前对象
- [x] 理解 Const Member Function
- [x] 实现可写与只读的 Const Overload

## 1. Class：数据、操作与不变量

![Class 中的 public、private、Constructor 与 Destructor](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@main/img/20260829-cs106l-l08-class-anatomy.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 8 — Inheritance（Classes Recap）](https://web.stanford.edu/class/cs106l/lectures/2026Spring-08-Inheritance.pdf)

Class 通过数据、操作和访问控制维护对象不变量：`public` 暴露稳定接口，`private` 隐藏实现细节。封装的目的不是单纯隐藏成员，而是防止对象进入非法状态。

## 2. Constructor、Destructor 与 Lifetime

Constructor 在对象初始化时调用：

```cpp
class Tensor {
public:
    explicit Tensor(std::size_t size) {
        // 初始化对象
    }
};

Tensor tensor(100);
```

Constructor：

- 与类同名；
- 没有返回类型；
- 可以重载；
- 负责让对象从生命周期开始就处于合法状态。

`explicit` 可以阻止某些意外的隐式类型转换：

```cpp
// Tensor tensor = 100; // explicit 时不允许
Tensor tensor(100);     // 明确构造
```

Destructor 在对象生命周期结束时自动调用：

```cpp
class Tensor {
public:
    ~Tensor() {
        // 清理对象拥有的资源
    }
};
```

```cpp
{
    Tensor tensor(100);
} // 在这里自动调用 ~Tensor()
```

对象生命周期：

```text
存储可用
   ↓
Constructor
   ↓
对象 Lifetime 开始
   ↓
正常使用
   ↓
Destructor
   ↓
对象 Lifetime 结束
```

> Constructor 建立不变量；Destructor 释放对象拥有的资源。若成员本身是 RAII 类型，通常不必手写 Destructor。

## 3. Member Initializer List

```cpp
class Tensor {
public:
    explicit Tensor(std::size_t size)
        : data_(size) {
    }

private:
    std::vector<float> data_;
};
```

`: data_(size)` 是 Member Initializer List。它在成员初始化阶段直接构造 `data_`，不是先默认构造再赋值。

成员实际初始化顺序由它们在 Class 中的声明顺序决定，而不是 Initializer List 中书写的顺序。

```text
成员声明顺序 → 决定初始化顺序
Initializer List → 决定每个成员如何初始化
```

它对以下成员尤其必要或重要：

- Class 类型成员；
- Reference 成员；
- Const 成员；
- 没有默认 Constructor 的成员。

## 4. `this`

`this` 是指向当前对象的 Pointer：

```cpp
class Tensor {
public:
    void setSize(std::size_t size) {
        this->size_ = size;
    }

private:
    std::size_t size_ = 0;
};
```

```cpp
Tensor a(10);
a.setSize(20);
```

调用 `a.setSize(20)` 时，函数内部的 `this` 指向 `a`。

在不存在名称冲突时，可以省略 `this->`：

```cpp
size_ = size;
```

## 5. Const Member Function

```cpp
std::size_t size() const {
    return data_.size();
}
```

函数末尾的 `const` 表示：

> 该函数承诺不通过 `this` 修改当前对象的非 `mutable` 状态。

在 Const Member Function 中，`this` 指向 Const 对象，可近似理解为：

```cpp
Tensor const* this;
```

因此：

```cpp
std::size_t size() const {
    // data_.push_back(1.0f); // 编译错误
    return data_.size();
}
```

Const 对象只能调用 Const Member Function：

```cpp
const Tensor tensor(10);
tensor.size(); // 合法，因为 size() 是 const
```

> **> 这是 Shallow Const：它限制通过当前对象修改非 `mutable` 成员，但不会递归冻结 Pointer 指向的外部对象。**

## 6. Const Overload

经典接口：

```cpp
class Tensor {
public:
    float& at(std::size_t index) {
        return data_.at(index);
    }

    const float& at(std::size_t index) const {
        return data_.at(index);
    }

private:
    std::vector<float> data_;
};
```

普通对象调用非 Const 版本：

```cpp
Tensor tensor(3);
tensor.at(0) = 3.14f; // 合法
```

Const 对象调用 Const 版本：

```cpp
const Tensor tensor(3);
std::cout << tensor.at(0); // 合法
// tensor.at(0) = 3.14f;  // 编译错误
```

接口根据对象是否为 Const，自动保留正确的可修改性：

```text
普通对象 → float&       → 可读、可写
Const 对象 → const float& → 只读
```

![Const Overload 为 const 与非 const 对象提供不同返回类型](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@img/img/cs106l-2026-day4-const-overload.png?rev=20260909)

> 图源：[Stanford CS106L Spring 2026 · Lecture 9](https://web.stanford.edu/class/cs106l/lectures/2026Spring-09-TemplateClasses.pdf)，第 75 页。

Const Overload 让同一逻辑接口同时服务可写与只读对象，并把对象的可修改性正确传递到返回值。

## 7. 完整的 `Tensor` 示例

```cpp
#include <cstddef>
#include <vector>

class Tensor {
public:
    explicit Tensor(std::size_t size)
        : data_(size) {
    }

    std::size_t size() const {
        return data_.size();
    }

    float& at(std::size_t index) {
        return data_.at(index);
    }

    const float& at(std::size_t index) const {
        return data_.at(index);
    }

private:
    std::vector<float> data_;
};
```

成员资源由 `vector` 管理，因此该类遵循 Rule of Zero，不需要手写 Destructor、Copy 或 Move 操作。

## 8. 代码实验

### Experiment 1｜Constructor 与 Destructor 顺序

```cpp
#include <iostream>

class Resource {
public:
    Resource() {
        std::cout << "construct\n";
    }

    ~Resource() {
        std::cout << "destroy\n";
    }
};

int main() {
    std::cout << "before\n";
    {
        Resource resource;
        std::cout << "inside\n";
    }
    std::cout << "after\n";
}
```

输出顺序应为 `before → construct → inside → destroy → after`。

### Experiment 2｜Tensor Const Overload

```cpp
#include <iostream>

int main() {
    Tensor tensor(3);
    tensor.at(0) = 1.5f;

    const Tensor& view = tensor;
    std::cout << view.at(0) << '\n';

    // view.at(0) = 2.0f; // 编译错误
}
```

同一个对象经普通或 Const 访问路径会选择不同 Overload，并返回相应可修改性的 Reference。

## 9. 易错点

1. Constructor 没有返回类型，连 `void` 也不能写。
2. Member Initializer List 是直接初始化，不是先构造再赋值。
3. 成员按声明顺序初始化，不按 Initializer List 的书写顺序。
4. 函数末尾的 `const` 修饰 Member Function，不是返回值。
5. Const 对象不能调用普通非 Const Member Function。
6. `float& at(...)` 与 `const float& at(...) const` 的差异既包括返回类型，也包括成员函数的 Const 限定。
7. 使用标准容器作为成员时，通常不需要手写 Destructor。

## 10. 自测

- [x] Class 的封装包含哪三个部分？
- [x] Constructor 与 Destructor 分别在什么时候调用？
- [x] Member Initializer List 与函数体内赋值有什么区别？
- [x] `this` 指向谁？
- [x] `size() const` 中的 `const` 限制什么？
- [x] 为什么 `at()` 经常提供 Const 与非 Const 两个版本？

> **一句话总结**
> Class 通过封装、生命周期与 Const Correctness 建立可靠的对象边界；优先让 RAII 成员承担资源管理。

## 下一步

> C++ Day 5 - Inheritance & Polymorphism：通过 Base Interface 统一管理具有不同行为的 Derived 对象。
