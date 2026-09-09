---
title: 'C++ Day 1 - Pointer, Reference & Iterator'
categories:
  - 学习笔记
  - C++
tags:
  - C++
  - Modern C++
  - Pointer
  - Reference
  - Iterator
description: 从值、地址和解引用出发，建立 Pointer、Reference、参数传递与 Iterator 的统一心智模型。
readmore: true
abbrlink: 6d8b75a2
date: 2026-08-27 09:00:00
updated: 2026-09-09 23:41:00
---
> **学习信息**
> **学习日期：** 2026-08-27（周四）
> **课程：** Stanford CS106L 2026 L3、L6
> **重点：** Reference、Pointer、参数传递、二级指针、Iterator
> **所属计划：** 14 天 C++ 学习计划 · Day 1


<!-- more -->
## 本节目标

- 区分值、地址与解引用；
- 解释 Reference 为什么是别名而不是复制；
- 比较 Pass by Value、Reference、Pointer 与 `const T&`；
- 沿着 `T**` 完成两次解引用；
- 使用 Iterator 理解 `begin()`、`end()` 与 `[begin, end)`。

## 1. 值、地址与 Pointer

```cpp
int x = 10;
int* p = &x;
```

| 表达式 | 含义 |
|---|---|
| `x` | 对象的值 |
| `&x` | `x` 的地址 |
| `p` | `p` 保存的地址，此处等于 `&x` |
| `*p` | 沿地址访问对象，此处就是 `x` |
| `&p` | Pointer 变量 `p` 自己的地址 |

```cpp
*p = 20;           // 等价于 x = 20
std::cout << x;    // 20
```

![Pointer 保存变量地址](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@img/img/cs106l-day1-pointer.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 6](https://web.stanford.edu/class/cs106l/lectures/2026Spring-06-Iterators.pdf)，第 78 页。

> Pointer 保存地址，`*` 根据地址访问对象。注意：`p` 与 `&p` 是两个不同地址。

## 2. Reference：同一对象的另一个名字

```cpp
int x = 10;
int& r = x;
r = 20;            // x 也变为 20
```

Reference 不创建新对象；`x` 与 `r` 是同一对象的两个名字。

![Reference 是同一对象的另一个名字](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@img/img/cs106l-day1-reference.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 3](https://web.stanford.edu/class/cs106l/lectures/2026Spring-03-InitializationAndReferences.pdf)，第 30 页。

### Pointer 与 Reference

| Pointer | Reference |
|---|---|
| 保存对象地址 | 是已有对象的别名 |
| 可以改变指向 | 初始化后不能重新绑定 |
| 可为 `nullptr` | 正常 Reference 必须绑定对象 |
| 用 `*p` 访问对象 | 直接用 `r` 访问对象 |

```cpp
int x = 10;
int y = 20;
int& r = x;
r = y;              // 把 y 的值赋给 x；r 仍是 x 的别名
```

## 3. 函数参数传递

| 写法 | 发生什么 | 能否修改原对象 | 调用形式 |
|---|---|:---:|---|
| `void f(T x)` | 创建参数副本 | 否 | `f(obj)` |
| `void f(T& x)` | 参数是原对象的别名 | 是 | `f(obj)` |
| `void f(T* x)` | 复制地址，再通过 `*x` 访问 | 是 | `f(&obj)` |
| `void f(const T& x)` | 不复制，只读访问 | 否 | `f(obj)` |

```cpp
void by_value(int x)      { x = 20; }
void by_reference(int& x) { x = 30; }
void by_pointer(int* x)   { *x = 40; }
```

> **> 大型只读参数常用 `const T&`，例如 `const std::vector<T>&`、`const std::string&`、`const Tensor&`。小型标量通常直接按值传递。**

`const T&` 的 `const` 限制的是这条访问路径：

```cpp
int x = 10;
const int& r = x;
x = 20;      // 合法
// r = 30;   // 非法
```

## 4. 二级指针 `T**`

```cpp
int x = 10;
int* p = &x;
int** pp = &p;
```

| 表达式 | 结果 |
|---|---|
| `pp` | `p` 的地址 |
| `*pp` | `p` |
| `**pp` | `x` |

```cpp
**pp = 50;   // 等价于 x = 50
```

> 有几层 Pointer，访问最终对象时通常就需要对应几次 dereference。

## 5. Iterator：容器访问的统一接口

Container 决定数据如何存；Iterator 决定如何访问；Algorithm 决定如何处理。算法因此不必了解 `vector`、`list`、`map` 的内部结构。

![Iterator 从 begin 逐步移动到 end](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@img/img/cs106l-day1-iterator.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 6](https://web.stanford.edu/class/cs106l/lectures/2026Spring-06-Iterators.pdf)，第 23 页。

```cpp
auto it = container.begin();
++it;                       // 移到下一个元素
auto& elem = *it;           // 访问当前元素
bool done = it == container.end();
```

Iterator 的接口类似 Pointer，但 Iterator 不一定就是 Pointer；不同容器的 `++it` 可以对应数组移动、链表跳转或树节点遍历。

## 6. `begin()`、`end()` 与遍历

`begin()` 指向第一个元素；`end()` 指向最后一个元素之后的位置（past-the-end），不能被解引用。

![end 指向容器最后一个元素之后的位置](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@img/img/cs106l-day1-begin-end.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 6](https://web.stanford.edu/class/cs106l/lectures/2026Spring-06-Iterators.pdf)，第 27 页。

```cpp
for (auto it = v.begin(); it != v.end(); ++it) {
    std::cout << *it << '\n';
}
```

这就是 `[begin, end)`：包含 `begin`，不包含 `end`。空容器统一表示为 `begin() == end()`。

## 7. 易错点速查

| 易错认识 | 正确判断 |
|---|---|
| `p` 与 `&p` 是同一个地址 | `p` 是目标地址；`&p` 是 Pointer 自己的地址 |
| `r = y` 会让 Reference 改绑到 `y` | 它把 `y` 的值赋给 `r` 所引用的对象 |
| `it` 就是元素 | `it` 是位置；`*it` 才是元素 |
| `end()` 是最后一个元素 | 它是最后一个元素之后的位置 |
| `*end()` 可取得最后一个元素 | 解引用 `end()` 是未定义行为 |
| Iterator 就是 Pointer | 两者接口相似，但 Iterator 是更通用的抽象 |

## 8. 最小代码测试

```cpp
#include <iostream>
#include <vector>

void by_value(int x) { x = 20; }
void by_reference(int& x) { x = 30; }
void by_pointer(int* x) { *x = 40; }

int main() {
    int x = 10;
    int* p = &x;
    int& r = x;
    int** pp = &p;

    std::cout << x << ' ' << &x << ' ' << p << ' '
              << *p << ' ' << &p << ' ' << **pp << '\n';

    by_value(x);      // 10
    by_reference(x);  // 30
    by_pointer(&x);   // 40

    std::vector<int> nums{10, 20, 30};
    for (auto it = nums.begin(); it != nums.end(); ++it) {
        std::cout << *it << '\n';
    }
}
```

## 9. 自测

- [x] `p`、`*p`、`&p` 分别是什么？
- [x] Reference 为什么不是复制？
- [x] `r = y` 为什么不是重新绑定？
- [x] 四种参数传递分别会不会复制或修改原对象？
- [x] `pp`、`*pp`、`**pp` 分别是什么？
- [x] 为什么不能解引用 `end()`？
- [x] Iterator 为什么只是“类似 Pointer”？

> **Day 1 完成**
> **小测结果：100 / 100**

### 后续复习重点

1. Pointer 与 Reference
2. 函数参数传递
3. 二级指针
4. Iterator
