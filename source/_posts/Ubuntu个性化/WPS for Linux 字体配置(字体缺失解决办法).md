---
title: WPS for Linux 字体配置(字体缺失解决办法)
categories:
  - Ubuntu个性化
tags:
  - Ubuntu
readmore: true
hideTime: true
abbrlink: 5cd3f1bd
date: 2025-12-26 10:48:39
---

## WPS for Linux 字体配置(字体缺失解决办法)

### 1. 背景

> 有些linux装完wps后提示“部分字体无法显示”或“some formula symbols might be not display”。这是因为缺少某些字体导致，主要是特殊符号或公式字体等等，而这些字体其实是在windows中可以找到的。有兴趣的自己去研究。

### 2. 解决方案

2.1. 下载字体库

[点击这里下载字体库（鼠标悬浮查看密码）](https://pan.baidu.com/s/1AhdMyXPbYsEnP0PYlLbVtQ "点击这里下载字体库（鼠标悬浮查看密码）")(g9ci)

2.2. 添加字体（使下载的字体库生效）

方法一：解压到 [WPS](https://so.csdn.net/so/search?q=WPS&spm=1001.2101.3001.7020)的默认字体文件夹中(wps-office)，然后重启WPS即可

```cobol
sudo unzip wps_symbol_fonts.zip -d /usr/share/fonts/wps-office
```

tips：如果`/usr/share/fonts/`目录下没有`wps-office`文件夹，则使用方法二。

方法二：

- 将字体库文件解压

```python
sudo unzip wps_symbol_fonts.zip		# 解压zip文件
```

- 更新字体

```csharp
# 生成字体索引sudo mkfontscalesudo mkfontdir# 更新字体缓存sudo fc-cache
```

- 重新打开WPS，提示不再出现。

方法三：图形化操作

- 点击zip文件，鼠标右键 -> 解压，解压后会得到一些tff文件
- 双击点开tff文件，点击安装（全部的tff文件都操作一遍即可）