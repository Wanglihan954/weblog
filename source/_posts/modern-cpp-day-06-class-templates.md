---
title: C++ Day 6 - Class Templates
categories:
  - 学习笔记
  - C++
tags:
  - C++
  - Modern C++
  - Template
  - Tensor
description: 以 Tensor<T> 为例理解类模板、实例化、函数模板与模板定义通常放在 header 的原因。
readmore: true
abbrlink: aca26b45
date: 2026-08-30 09:00:00
updated: 2026-09-09 23:41:00
---
> **学习信息**
> **学习日期：** 2026-08-30（周日）
> **重点：** Class Template、类型参数、模板实例化、Function Template、模板定义的可见性
> **所属计划：** 14 天 C++ 学习计划 · Day 6
> **前置笔记：** C++ Day 4 - Classes & Const Correctness、C++ Day 3 - Containers & Iterator


<!-- more -->
## 今日目标

- [x] 解释 Template 与具体类型的关系；
- [x] 写出最小的 `Tensor<T>`；
- [x] 创建 `Tensor<float>`、`Tensor<int>` 和 `Tensor<double>`；
- [x] 理解 Function Template 的参数推导和显式指定；
- [x] 解释模板实现为什么通常放在 Header；
- [x] 保留普通对象与 Const 对象的双重访问接口。

## 1. Template 不是一个具体类型

模板可以先理解成“根据类型参数生成代码的蓝图”：

```cpp
template <typename T>
class Box {
public:
    explicit Box(T value) : value_(value) {}

    const T& value() const {
        return value_;
    }

private:
    T value_;
};
```

这里的 `Box` 还是模板，不是可以直接定义对象的完整类型。使用时指定参数：

```cpp
Box<int> integer_box(10);
Box<std::string> string_box("tensor");
```

![Template 像工厂一样，根据类型参数产生不同具体类型](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@img/img/cs106l-2026-day6-template-factory-v2.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 9](https://web.stanford.edu/class/cs106l/lectures/2026Spring-09-TemplateClasses.pdf)，第 32 页。

编译器会根据 `T` 生成对应的具体类。`Box<int>` 和 `Box<std::string>` 是两个不同的具体类型，但来自同一个模板定义。

> `typename T` 与 `class T` 在“类型模板参数”这个位置基本等价。现代代码通常使用 `typename` 表达“这里传入的是类型”。

## 2. 用 Class Template 泛化 `Tensor`

Day 4 的 `Tensor` 如果固定保存 `float`，只能处理一种元素类型。把元素类型提升为模板参数：

```cpp
#include <cstddef>
#include <vector>

template <typename T>
class Tensor {
public:
    explicit Tensor(std::size_t size)
        : data_(size) {}

    std::size_t size() const {
        return data_.size();
    }

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

使用同一个模板：

```cpp
Tensor<float> features(4);
Tensor<int> labels(4);
Tensor<double> weights(4);

features.at(0) = 1.5f;
labels.at(0) = 1;
weights.at(0) = 0.25;
```

### 从内向外拆解 `Tensor<float>`

```text
float
  ↓ 作为类型参数
Tensor<float>
  ↓ 具体实例化类型
features
  ↓ 对象
features.at(0)
  ↓ 通过成员函数访问内部 vector
```

模板参数是类型，不是运行时变量。`Tensor<float>` 不会在运行时根据字符串再决定元素类型；具体类型在编译期已经确定。

## 3. Const Overload 仍然适用

模板不会改变 Class 的 Const Correctness 规则：

```cpp
Tensor<int> mutable_tensor(2);
mutable_tensor.at(0) = 7;       // 调用 T& at(...)

const Tensor<int> readonly_tensor(2);
readonly_tensor.at(0);           // 调用 const T& at(...) const
// readonly_tensor.at(0) = 7;    // 编译错误
```

普通对象得到可写引用，Const 对象得到只读引用。这个模式在 `Tensor<T>`、`std::vector<T>` 和推理框架的输入输出接口里都很常见。

## 4. Function Template

Class Template 把类型参数用于类；Function Template 把类型参数用于函数：

```cpp
template <typename T>
T add(T lhs, T rhs) {
    return lhs + rhs;
}

int a = add(1, 2);             // 推导 T = int
double b = add(1.0, 2.5);      // 推导 T = double
double c = add<double>(1, 2.5); // 显式指定 T = double
```

多个参数参与同一个 `T` 的推导时，类型必须能形成一致结果：

```cpp
// add(1, 2.5);                // 通常推导失败：int 与 double 冲突
add<double>(1, 2.5);           // 明确要求统一转换为 double
```

模板参数推导不是“把所有参数自动转换成最宽类型”。当推导不明确时，显式写出模板参数或设计不同的参数类型更清晰。

## 5. 为什么模板实现通常放在 Header

普通函数可以分离声明和实现：

```cpp
// math.hpp
int add(int lhs, int rhs);

// math.cpp
int add(int lhs, int rhs) {
    return lhs + rhs;
}
```

每个 `.cpp` 单独编译后，Linker 可以把声明与实现连接起来。

模板实例化则通常需要编译器在看到调用时生成具体代码：

```cpp
// add.hpp
template <typename T>
T add(T lhs, T rhs) {
    return lhs + rhs;
}

// main.cpp
#include "add.hpp"
auto result = add<int>(1, 2);
```

编译 `main.cpp` 时，编译器必须看到 `add` 的完整定义，才能生成 `add<int>`。因此模板实现经常放在 `.h`、`.hpp` 或 `.tpp` 文件中。

如果确实希望把实现放在 `.cpp`，可以在实现文件中显式实例化指定类型，但这样只能提供那些已经显式实例化的类型：

```cpp
template int add<int>(int, int);
```

第一轮学习阶段优先采用“模板声明和实现一起放在 Header”的简单规则。

## 6. Template 与 AI/CV 工程

在推理框架中，模板常用于表达“数据类型可以变化，但容器逻辑相同”：

```text
Tensor<float>   → 模型计算
Tensor<int>     → 标签或索引
Tensor<double>  → 高精度实验
```

模板减少重复代码，同时保留静态类型检查和通常的零运行时分派成本。它和 Runtime Polymorphism 解决的问题不同：

| 机制 | 主要决定时间 | 典型问题 |
|---|---|---|
| Template | 编译期 | 同一套逻辑如何支持多种类型 |
| Virtual | 运行时 | 同一接口如何调用不同 Derived 行为 |

## 7. 最小代码实验

```cpp
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

template <typename T>
class Tensor {
public:
    explicit Tensor(std::size_t size)
        : data_(size) {}

    std::size_t size() const {
        return data_.size();
    }

    T& at(std::size_t index) {
        return data_.at(index);
    }

    const T& at(std::size_t index) const {
        return data_.at(index);
    }

private:
    std::vector<T> data_;
};

template <typename T>
T add(T lhs, T rhs) {
    return lhs + rhs;
}

int main() {
    Tensor<float> image_values(3);
    image_values.at(0) = 0.5f;

    const Tensor<float>& view = image_values;
    std::cout << view.at(0) << '\n';
    std::cout << add(2, 3) << '\n';
    std::cout << add<double>(2, 0.5) << '\n';
}
```

## 8. 易错点速查

| 易错认识 | 正确判断 |
|---|---|
| `Tensor` 本身就是一个完整类型 | `Tensor` 是模板；`Tensor<float>` 才是具体类型 |
| 模板参数是运行时变量 | 类型模板参数通常在编译期参与实例化 |
| `add(1, 2.5)` 一定自动变成 `double` | 同一模板参数的推导可能冲突，应显式指定或改接口 |
| 模板实现可以像普通函数一样只放 `.cpp` | 编译实例化时通常需要看到完整定义 |
| `const Tensor<T>` 不能调用任何成员函数 | 可以调用标记为 `const` 的成员函数 |
| 模板必然有运行时开销 | 很多模板实例化是编译期生成，具体开销取决于代码和优化 |

## 9. 过关自测

- [x] 能从内向外解释 `Tensor<float>`；
- [x] 能写出最小的 `template <typename T> class Tensor`；
- [x] 能说明 `add(1, 2)` 如何推导 `T`；
- [x] 能解释为什么 `add<double>(1, 2.5)` 可以明确解决类型问题；
- [x] 能解释模板实现通常放在 Header 的原因；
- [x] 能写出 `T&` / `const T&` 的 Const Overload。

> **Day 6 完成**
> 对话整理记录中的 Day 6 测试为 **23 / 25（92%）**。当前重点从“会写模板”转向“在 Tensor 和工程 Header 中识别模板实例化”。

### 后续复习重点

1. `Tensor<T>`
2. 模板中的 Const Interface
3. 模板与编译单元
