---
title: C++ Day 3 - Containers & Iterator
categories:
  - 学习笔记
  - C++
tags:
  - C++
  - Modern C++
  - STL
  - vector
  - Iterator
description: 梳理 vector、string、容量扩张与 iterator/reference/pointer 失效规则。
readmore: true
abbrlink: a8615ab1
date: 2026-08-29 09:00:00
updated: 2026-09-09 23:41:00
---
> **学习信息**
> **学习日期：** 2026-08-29（周六，压缩完成）
> **重点：** `std::vector`、`std::string`、`size/capacity`、扩容与访问路径失效
> **所属计划：** 14 天 C++ 学习计划 · Day 3
> **前置笔记：** C++ Day 1 - Pointer, Reference & Iterator、C++ Day 2 - Const Correctness, Lifetime & Dynamic Memory


<!-- more -->
## 今日目标

- [x] 理解 STL Container 的作用和常见底层结构
- [x] 掌握 `vector` 的常用接口
- [x] 掌握 `std::string` 的基本使用
- [x] 区分 `size()`、`capacity()` 与 `reserve()`
- [x] 理解扩容及 Pointer、Reference、Iterator 失效
- [x] 完成 scores 最大值、最小值与平均值统计

## 1. Container：用数据结构表达操作需求

STL Container 管理一组对象，并提供相对统一的访问接口。选择容器，本质是在权衡存储方式、访问顺序及插入、删除和查找成本。

| Container | 典型结构 | 主要特点 |
|---|---|---|
| `vector` | 连续内存 | 随机访问快，扩容可能搬迁元素 |
| `list` | 双向链表 | 节点式存储，不支持随机访问 |
| `map/set` | 通常为平衡树 | Key 有序，操作通常为 `O(log n)` |
| `unordered_map` | 哈希表 | Key 无序，平均查找 `O(1)` |

## 2. `std::vector` 与 `std::string`

![Vector 是可动态调整大小的连续数组](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@main/img/20260829-cs106l-l05-vector-contiguous.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 5 — Containers](https://web.stanford.edu/class/cs106l/lectures/2026Spring-05-Containers.pdf)

`vector` 自动管理可动态调整大小的连续数组，因此支持高效的随机访问：

```cpp
nums[1];    // 不检查越界
nums.at(1); // 检查越界，失败时抛出异常
```

常用接口：

```cpp
nums.push_back(10);
nums.emplace_back(20);

nums.size();
nums.empty();

nums[0];
nums.at(0);

nums.begin();
nums.end();
```

`emplace_back()` 用给定参数直接构造尾部元素，但不保证总比 `push_back()` 快。对于 `int` 等简单类型，两者通常没有实质差异。

`std::string` 管理字符序列，并共享许多容器式接口：

```cpp
#include <string>

std::string framework = "KuiperInfer";
framework += " C++";

std::cout << framework.size() << '\n';
std::cout << framework[0] << '\n';
std::cout << framework.at(1) << '\n';
```

## 3. `size`、`capacity`、`reserve` 与 `resize`

- `size()`：已经构造的元素数量；
- `capacity()`：当前存储无需重新分配即可容纳的元素数量；
- `reserve(n)`：保证 Capacity 至少为 `n`，不创建元素；
- `resize(n)`：把 Size 改为 `n`，必要时创建或销毁元素。

```text
size = 3
capacity = 4

[10][20][30][预留]
```

始终有 `size() <= capacity()`；预留位置不是可访问元素。Capacity 的增长策略由实现决定，不能依赖它一定翻倍。

```cpp
std::vector<int> values;
values.reserve(100); // size == 0，capacity >= 100
values.resize(100);  // size == 100
```

## 4. Reallocation 与访问路径失效

假设：

```text
size = 4
capacity = 4

[1][2][3][4]
```

执行：

```cpp
v.push_back(5);
```

若当前空间不足，`vector` 会：

```text
申请更大的连续存储
        ↓
复制或移动旧元素
        ↓
销毁旧元素并释放旧存储
        ↓
构造新元素
```

```cpp
std::vector<int> values{10};

int* pointer = &values[0];
int& reference = values[0];
auto iterator = values.begin();
```

若后续操作触发 Reallocation：

```cpp
values.push_back(20);
```

若发生 Reallocation，元素被搬到新地址，原有 Pointer、Reference 和 Iterator 全部失效；继续使用会产生 Undefined Behavior。

规则要点：

- 若发生 Reallocation，所有指向该 `vector` 元素的 Pointer、Reference 和 Iterator 都失效；
- 若 `push_back()` 未发生 Reallocation，已有元素的 Pointer 和 Reference 通常仍有效，但原来的 `end()` Iterator 会失效；
- 不同修改操作有各自的失效规则，使用前应查对应接口文档。

> `vector` 搬家后，旧地址不会自动更新。失效的本质是访问路径仍在，但它已不再指向原来的有效位置。

## 5. `begin()`、`end()` 与 Iterator

![begin 指向首元素，end 指向尾后位置](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@main/img/20260829-cs106l-l06-begin-end.png)

> 图源：[Stanford CS106L Spring 2026 · Lecture 6 — Iterators](https://web.stanford.edu/class/cs106l/lectures/2026Spring-06-Iterators.pdf)

Iterator 的基础见 Day 1；本节只需把它与 `vector` 的 Reallocation 联系起来。Algorithm 应用见 C++ Day 7 - Lambda, Algorithms & Associative Containers。

## 6. 代码实验

### Experiment 1｜观察 Size、Capacity 与 Reallocation

连续插入元素，观察 Capacity 和底层地址何时变化：

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<int> values;

    for (int i = 0; i < 20; ++i) {
        const auto old_capacity = values.capacity();
        values.push_back(i);

        std::cout << "size=" << values.size()
                  << ", capacity=" << values.capacity()
                  << ", reallocated="
                  << (old_capacity != values.capacity())
                  << '\n';
    }
}
```

`push_back()` 后 Capacity 改变，说明发生了 Reallocation；旧存储中的 Pointer、Reference 和 Iterator 已不能继续使用。

### Experiment 2｜Scores 统计程序

使用 `vector<float>` 完成遍历与基本统计：

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<float> scores{
        82, 91, 76, 88, 95, 67, 84, 90, 73, 86,
        79, 92, 81, 87, 69, 94, 78, 85, 89, 80
    };

    float minimum = scores.at(0);
    float maximum = scores.at(0);
    float sum = 0.0f;

    for (std::size_t index = 0; index < scores.size(); ++index) {
        const float score = scores.at(index);
        std::cout << score << ' ';
        sum += score;

        if (score < minimum) {
            minimum = score;
        }
        if (score > maximum) {
            maximum = score;
        }
    }

    const float average = sum / static_cast<float>(scores.size());

    std::cout << "\nmin=" << minimum
              << "\nmax=" << maximum
              << "\naverage=" << average << '\n';
}
```

## 7. 易错点

1. `capacity()` 表示预留空间，不表示已经存在的元素数量。
2. `reserve(100)` 不等于 `resize(100)`。
3. 不要假设 Capacity 一定按两倍增长。
4. `push_back()` 触发 Reallocation 后，旧 Pointer、Reference、Iterator 全部失效。
5. `operator[]` 不检查越界，`at()` 会进行边界检查。
6. `string` 与 `vector` 管理的元素类型不同，但共享许多容器式接口。

## 8. 自测

- [x] `size()` 与 `capacity()` 有什么区别？
- [x] `reserve(100)` 会不会创建 100 个元素？
- [x] Vector 扩容时旧元素经历了什么？
- [x] Reallocation 为什么会使 Iterator 失效？
- [x] `[]` 与 `at()` 的越界行为有什么区别？
- [x] `string` 与 `vector` 有哪些相似接口？
- [x] Scores 程序如何计算最小值、最大值与平均值？

> **一句话总结**
> `vector` 自动管理连续动态存储，但 Reallocation 可能搬迁元素，因此必须同时关注 Iterator、Reference 和 Pointer 的有效性。

## 下一步

> C++ Day 4 - Classes & Const Correctness：把 Container 封装成具有清晰生命周期与 Const Interface 的对象。
