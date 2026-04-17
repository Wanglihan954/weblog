---
title: <% tp.file.title %>
date: <% tp.file.creation_date("YYYY-MM-DD HH:mm:ss") %>
author: 宁翰
email: 314375980@qq.com
categories:
  - <% await tp.system.suggester(["🧠 人工智能", "💻 编程开发", "🛠️ 框架与部署", "💡 算法与基础", "☕ 生活随笔", "🐧 Ubuntu个性化", "🐍 Mamba"], ["人工智能", "编程开发", "框架与部署", "算法与基础", "生活随笔", "Ubuntu个性化", "Mamba"]) %>
tags:
  - <% tp.file.cursor(1) %>
readmore: true
hideTime: true
---
