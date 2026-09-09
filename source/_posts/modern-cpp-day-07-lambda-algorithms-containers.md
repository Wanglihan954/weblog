---
title: 'C++ Day 7 - Lambda, Algorithms & Associative Containers'
categories:
  - 学习笔记
  - C++
tags:
  - C++
  - Modern C++
  - Lambda
  - Algorithm
  - STL
description: 结合 sort、find、transform 与 map、unordered_map、set，掌握 STL 的协作方式。
readmore: true
abbrlink: 9a90668a
date: 2026-08-31 09:00:00
updated: 2026-09-09 23:41:00
---
> **学习信息**
> **学习日期：** 2026-08-31（周一）
> **重点：** Lambda、Capture、`sort/find/transform`、`map/unordered_map/set`
> **所属计划：** 14 天 C++ 学习计划 · Day 7
> **前置笔记：** C++ Day 3 - Containers & Iterator、C++ Day 6 - Class Templates

![STL 的四个协作部分：Containers、Iterators、Functors 与 Algorithms](https://cdn.jsdelivr.net/gh/Wanglihan954/Picture-bed@main/img/cs106l-2026/day7-stl-lambda.png)

> 图源：Stanford CS106L Spring 2026，[Functions & Lambdas Slides](https://web.stanford.edu/class/cs106l/lectures/2026Spring-11-LambdasAndFunctors.pdf) 第 63 页。Lambda 是 Algorithm 接收的可调用规则；它与 Container、Iterator 一起构成 STL 的协作关系。


<!-- more -->
## 今日目标

- [x] 拆解 Lambda 的 capture、参数和 body。
- [x] 用 Lambda 作为 STL Algorithm 的可调用对象。
- [x] 使用 sort、find、transform 处理 iterator range。
- [x] 区分 map、unordered_map 与 set。
- [x] 识别 operator[] 查询时可能插入默认值。

## 1. Algorithm 连接 Container 与行为

Day 3 的 Container 提供数据与 Iterator；Algorithm 接收 `[begin, end)` 与规则：

```text
vector / map
    ↓ begin(), end()
iterator range
    ↓ + predicate / lambda
sort / find / transform
```

这就是 STL 的分工：容器不需要知道算法，算法也不需要知道具体底层存储。

## 2. Lambda：现场创建的可调用对象

```cpp
auto double_value = [](int x) {
    return x * 2;
};
```

语法：

```cpp
[capture](parameters) {
    body
}
```

它可概念化为编译器生成的匿名 Functor：

```cpp
struct DoubleValue {
    int operator()(int x) const {
        return x * 2;
    }
};
```

因此 Lambda 不只是匿名函数，更准确地说是“行为 + 可选捕获状态”的对象。

### Capture

| 写法 | 含义 | 修改外部对象？ |
| --- | --- | --- |
| [] | 不捕获 | 不可以 |
| [x] / [=] | 按值捕获副本 | 默认不可以，mutable 修改的是副本 |
| [&x] / [&] | 按引用捕获 | 可以 |
| [=, &x] | 默认按值，x 按引用 | 仅 x 可改 |

```cpp
int threshold = 5;
auto less_than = [threshold](int x) {
    return x < threshold;
};
```

> **> [&] 很方便，但 Lambda 若比外部变量活得更久，引用可能悬空。短生命周期的局部 Algorithm 调用通常安全；异步或延迟执行时要格外谨慎。**

## 3. 三个常用 Algorithm

### sort

```cpp
std::sort(scores.begin(), scores.end());  // 升序

std::sort(scores.begin(), scores.end(),
          [](float a, float b) {
              return a > b;               // 降序
          });
```

比较器返回 true 表示“左元素应排在右元素之前”。

### find

```cpp
auto it = std::find(scores.begin(), scores.end(), 20.f);
if (it == scores.end()) {
    // 没找到
}
```

返回的是 iterator，而不是布尔值；end() 表示“查找失败后的 past-the-end 位置”。

### transform

```cpp
std::transform(scores.begin(), scores.end(), normalized.begin(),
               [max_score](float x) {
                   return x / max_score;
               });
```

它按元素应用变换规则。输出 range 必须已经有足够空间；若想自动增长，可用 back_inserter。

## 4. Associative Container 补丁

| 容器 | Key 是否有序 | 查找直觉 | 适合场景 |
| --- | --- | --- | --- |
| `map<K,V>` | 有序 | 树，通常 `O(log n)` | 需要按 Key 排序遍历 |
| `unordered_map<K,V>` | 无序 | 哈希，平均 `O(1)` | Registry、名称到对象/ID |
| `set<T>` | 有序且去重 | 树，通常 `O(log n)` | 唯一集合 |

```cpp
std::unordered_map<std::string, int> class_ids{
    {"person", 0}, {"car", 1}, {"ship", 2}
};

if (auto it = class_ids.find("car"); it != class_ids.end()) {
    std::cout << it->second;
}
```

> **> `table["missing"]` 不是纯查询：Key 不存在时会插入默认构造的 Value。只查询时优先使用 `find()`、`contains()` 或 `at()`。**

## 5. 最小 Operator Registry

```cpp
std::unordered_map<std::string, int> operators{
    {"conv", 0}, {"relu", 1}, {"pool", 2}, {"linear", 3}
};
```

它是第 5 课“注册器”思想的 STL 最小版本：字符串 key 映射到某个实现或创建函数。

## 6. 代码实验

```cpp
std::vector<float> scores{72.f, 95.f, 61.f, 88.f};

std::sort(scores.begin(), scores.end(),
          [](float a, float b) { return a > b; });

std::vector<float> ratios(scores.size());
std::transform(scores.begin(), scores.end(), ratios.begin(),
               [top = scores.front()](float x) { return x / top; });
```

这里的初始化捕获在 Lambda 创建时保存当前最高分副本。

## 7. 易错点

| 易错认识 | 正确理解 |
| --- | --- |
| Lambda 是普通函数指针 | 有 capture 的 Lambda 是带状态的闭包对象 |
| [=] 可以修改外部变量 | 它只复制外部变量；mutable 也只改副本 |
| find 失败返回 nullptr | 返回 end() iterator |
| unordered_map 一定比 map 快 | 平均更快，但无序且受 hash、冲突和数据规模影响 |
| operator[] 只是读取 | 缺失 key 时会插入默认值 |

## 8. 过关自测

- [x] 能拆解 Lambda 语法。
- [x] 能解释 [=]、[&] 与 mutable。
- [x] 能从 find 的返回 iterator 判断是否命中。
- [x] 能写出 sort 的降序 Lambda。
- [x] 能说明 vector 与 unordered_map 的选择依据。

> **Day 7 完成**
> 对话整理记录中的阶段测试为 **27 / 28（96%）**。重点已从语法记忆转为：在真实工程里阅读 Lambda、比较器与 Registry。

## 下一步

> C++ Day 8 - Special Member Functions & Copy Semantics：对象拥有资源时，复制究竟意味着复制什么？
