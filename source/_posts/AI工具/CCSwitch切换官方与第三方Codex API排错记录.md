---
title: CC Switch 切换官方与第三方 Codex API 排错记录
categories:
  - AI工具
tags:
  - Codex
  - CC Switch
  - API
  - Windows
  - PowerShell
readmore: true
hideTime: true
abbrlink: 50fec4d3
date: 2026-08-19 23:00:00
updated: 2026-08-19 23:00:00
---

# CC Switch 切换官方与第三方 Codex API 排错记录

> **摘要 · 先说结论**
> 在 Windows 上用 CC Switch 在官方 OpenAI API 与第三方 Codex API 之间切换时，最容易出现的不是单一的“Key 无效”，而是 **API Key、Base URL、Codex 配置文件、GUI 会话和 CLI 版本没有保持同一套状态**。本文记录一次完整排查过程，并给出一套不暴露密钥、方便复用的检查方法。

> **安全提醒**
> 本文所有 API Key 都使用占位符。真实密钥不要发到聊天、截图、命令输出、Git 仓库或博客中；已经暴露过的密钥应立即撤销并重新生成。

<!-- more -->

## 一、问题背景

我的目标是同时保留两套 Codex API：

| 用途 | API 地址 | 鉴权来源 |
| --- | --- | --- |
| 官方 Codex | `https://api.openai.com/v1` | 官方 OpenAI API Key，或 Codex 的 ChatGPT 登录 |
| 第三方 Codex | `https://openapi.example.com/v1` | 第三方服务商提供的 Codex API Key |

实际使用的是 Windows、Codex GUI、Codex CLI 和 CC Switch。CC Switch 负责保存供应商配置并切换当前 provider，Codex 则根据 `auth.json`、`config.toml`、环境变量以及自身的会话状态发起请求。

这里有一个很重要的前提：第三方服务商可能分别提供 Claude 配置和 Codex 配置。两者的 Key、模型列表、协议格式甚至 Base URL 都可能不同。Claude 配置能正常使用，并不代表同一个 Key 可以直接放进 Codex。

## 二、第一次遇到的 401

最初的错误类似这样：

```text
401 Unauthorized: Incorrect API key provided
url: https://api.openai.com/v1/responses
auth error code: invalid_api_key
```

错误里最有价值的不是 Key 的内容，而是请求 URL：请求已经到达了官方 `api.openai.com`，但携带的却是第三方 Key。因此不能只检查“Key 是否正确”，还要同时检查下面这组配对关系：

```text
官方 Key      + https://api.openai.com/v1
第三方 Key    + https://openapi.example.com/v1
```

如果出现下面这种组合，就一定会有问题：

```text
第三方 Key    + https://api.openai.com/v1
官方 Key      + https://openapi.example.com/v1
```

这类错误通常来自两个原因：

1. CC Switch 改写了 `config.toml` 的 Base URL，但没有同步 Codex 实际读取的鉴权文件。
2. Codex GUI 或旧进程仍然保留着切换前的配置，而 CC Switch 已经写入了另一套配置。

## 三、Codex 的几个配置入口

排查时先把各个配置入口分开，不要把 Claude Code 的配置和 Codex 的配置混在一起。

### 1. `auth.json`

Codex 的认证文件通常位于：

```text
%USERPROFILE%\.codex\auth.json
```

它可能包含 API Key，也可能包含 ChatGPT 登录产生的 OAuth 信息。桌面版 Codex 在切换登录模式时可能重写这个文件；在本次环境中，曾观察到它把 `OPENAI_API_KEY` 清空并切回 ChatGPT 登录状态。

因此，不要假设“我刚写入 `auth.json` 的 Key 永远会留在那里”。切换后应重新检查文件结构，但不要直接打印完整密钥：

```powershell
$auth = Get-Content "$HOME\.codex\auth.json" -Raw | ConvertFrom-Json
if ($auth.OPENAI_API_KEY) {
    "OPENAI_API_KEY 已设置，长度：$($auth.OPENAI_API_KEY.Length)"
} else {
    "OPENAI_API_KEY 未设置"
}
```

### 2. `config.toml`

Codex 的 provider、模型、协议和 Base URL 通常由下面这类配置控制：

```toml
model_provider = "custom"
model = "gpt-5.5"

[model_providers.custom]
name = "第三方 Codex"
base_url = "https://openapi.example.com/v1"
wire_api = "responses"
env_key = "THIRD_PARTY_CODEX_API_KEY"
```

其中最关键的是：

- `base_url` 必须指向当前供应商，并确认是否需要 `/v1`。
- `wire_api = "responses"` 要与服务商支持的协议一致。
- `env_key` 的值只是环境变量名称，不是 API Key 本身。
- 配置写成 `env_key = "THIRD_PARTY_CODEX_API_KEY"` 后，Codex 要求当前进程中真的存在这个环境变量。

可以只查看非敏感字段：

```powershell
Get-Content "$HOME\.codex\config.toml" |
    Select-String "model_provider|model|base_url|wire_api|env_key|service_tier"
```

### 3. CC Switch 的供应商数据库

CC Switch 不只是一个临时环境变量切换器。它会保存供应商配置，并在切换时写入 Codex 的配置文件。因而可能出现这种现象：手动修改 `config.toml` 后立即生效，但下一次在 CC Switch 中切换供应商时又被重新覆盖。

排错时要区分：

- 当前文件里的配置是什么；
- CC Switch 数据库里保存的模板是什么；
- Codex 进程启动时实际读到的配置是什么。

如果每次切换都复现同一个错误，优先修正 CC Switch 中对应供应商或公共配置的模板，而不是反复手改生成文件。

## 四、`requires_openai_auth` 与环境变量问题

排查过程中曾出现这样的配置：

```toml
[model_providers.custom]
base_url = "https://openapi.example.com/v1"
wire_api = "responses"
requires_openai_auth = true
```

`requires_openai_auth = true` 会让 Codex 按 OpenAI 官方鉴权路径处理该 provider。在第三方 provider 上保留这个字段，可能导致请求绕回 `api.openai.com`，最终把第三方 Key 发给官方端点。

另一个问题是把 `env_key` 设置成了某个环境变量名，却没有设置该变量。例如：

```toml
env_key = "THIRD_PARTY_CODEX_API_KEY"
```

如果进程中没有这个变量，Codex 会报：

```text
Missing environment variable: THIRD_PARTY_CODEX_API_KEY
```

PowerShell 中可以只检查变量是否存在，不显示其值：

```powershell
if ($env:THIRD_PARTY_CODEX_API_KEY) {
    "第三方 Key 已注入当前进程，长度：$($env:THIRD_PARTY_CODEX_API_KEY.Length)"
} else {
    "第三方 Key 未注入当前进程"
}
```

如果确实要使用持久化用户环境变量，可以手动执行：

```powershell
setx THIRD_PARTY_CODEX_API_KEY "<你的第三方 Codex API Key>"
```

`setx` 不会更新已经打开的 PowerShell 或 GUI 进程。执行后必须关闭旧终端和 Codex GUI，再启动新的进程。需要注意，`setx` 会把密钥写入用户环境配置，安全性低于由专用启动器在内存中注入密钥；个人电脑上应根据安全要求选择方案。

## 五、为什么 GUI 的旧对话会 401，而新对话正常

这是本次排查中最容易误判的一点。

在切换到第三方 API 后，Codex GUI 的新对话可以正常使用，但某些已经使用过官方 API 的旧对话仍然请求：

```text
https://api.openai.com/v1/responses
```

同时携带第三方 Key，于是得到 401。

现象可以概括为：

```text
旧官方对话
    -> 继续沿用创建时的 provider 或请求路由
    -> 请求仍发往 api.openai.com
    -> 当前认证文件已变成第三方 Key
    -> 401 invalid_api_key

新对话
    -> 读取当前 CC Switch 配置
    -> 使用第三方 Base URL 与第三方 Key
    -> 正常请求
```

这说明 GUI 的会话状态可能不完全等同于全局配置。切换供应商后，建议新建对话；官方和第三方对话也最好分开管理。如果必须继续旧对话，先尝试在 GUI 中重新选择 provider，或完全退出 GUI 后重新打开该对话。

这里的结论是根据实际现象得到的经验判断。不同 Codex GUI 版本的会话持久化实现可能不同，不应把它当作所有版本都固定如此的公开协议。

## 六、为什么第三方模式看不到官方 Codex 的宠物和其他数据

切换到第三方 API 后，除了旧对话可能出现 401，还可能发现界面没有原来使用的宠物，或者看不到官方模式下的会话、同步状态和账号相关数据。这并不表示第三方 API 把本地宠物删除了。

原因在于“模型请求”和“Codex 应用数据”是两条不同的链路：

```text
Codex GUI / CLI
    ├─ 本地界面、宠物资源、缓存与会话索引
    └─ 当前 provider -> API 请求 -> 模型服务
```

第三方 API 只负责接收请求并返回模型响应，它不会拥有 OpenAI 账号中的官方会话、云端同步状态、宠物配置或其他 GUI 账户数据。CC Switch 改变的是 provider、Base URL 和鉴权来源，并不会把官方账号的数据迁移到第三方服务商。

本次环境中，本地宠物资源仍然存在于：

```text
%USERPROFILE%\.codex\pets\xiaolemi
```

因此，如果官方模式能看到宠物而第三方模式看不到，优先考虑以下情况：

1. 当前启动的是 Codex CLI，而不是带宠物界面的 Codex GUI。
2. GUI 切换 API Key 后退出了 ChatGPT/OAuth 账号上下文，进入了纯 API Key 模式。
3. 第三方启动器使用了不同的 `CODEX_HOME`、不同的运行时或不同的本地数据目录。
4. 当前 GUI 版本把宠物显示或账号同步功能限制在官方登录模式中。

可以先确认是否设置了自定义数据目录：

```powershell
if ($env:CODEX_HOME) {
    "CODEX_HOME=$env:CODEX_HOME"
} else {
    "CODEX_HOME 未设置，通常使用默认目录 $HOME\.codex"
}
```

如果目标是保留完整的官方 GUI 体验，最稳妥的方式是让官方 Codex GUI 继续使用官方登录，并把第三方 API 放到独立的 Codex CLI、独立 profile 或受控的本地路由中。不要指望把第三方 Key 填入官方端点后仍然共享官方账号数据，也不要为了“恢复宠物”直接复制或覆盖整个 `.codex` 目录。

宠物资源存在于本地，只能证明文件没有被删除；它是否显示还取决于当前 GUI 运行时、认证模式和功能开关。这和 API Key 是否能成功调用模型是两个独立问题。

## 七、为什么只有 Codex CLI 报 `service_tier` 错误

切回官方 provider 后，Codex CLI 出现：

```text
Error loading config.toml: unknown variant `default`, expected `fast` or `flex`
in `service_tier`
```

这个错误发生在读取配置阶段，甚至还没有开始调用 API。它与 401 鉴权错误是两个独立问题。

本次环境中检查到的版本是：

| 程序 | 版本 |
| --- | --- |
| PowerShell 中的 npm Codex CLI | `0.130.0` |
| Codex GUI 内置 CLI | `0.148.0-alpha.9` |

因此，CLI 与 GUI 并不是同一个 Codex 构建，也不一定使用同一个配置解析器。CC Switch 或较新的 GUI 写入了：

```toml
service_tier = "default"
```

而旧版 CLI 的枚举只接受 `fast` 或 `flex`，所以直接拒绝加载配置。这个现象不是官方 API 导致的，也不是前面配置第三方 API 时降级了 Codex；更准确地说，是两个程序版本不同，以及生成的配置字段超出了旧 CLI 的 schema。

### 推荐修复

先升级 CLI：

```powershell
npm install -g @openai/codex@latest
```

关闭当前 PowerShell，重新打开后确认实际调用的是哪个文件：

```powershell
codex --version
Get-Command codex -All
where.exe codex
& "$env:APPDATA\npm\codex.cmd" --version
```

如果暂时不能升级，也可以在实际被 CLI 读取的配置中删除：

```toml
service_tier = "default"
```

对于官方 API，通常不需要手动设置 `service_tier`。如果确实需要保留，应使用当前 CLI 支持的值，但不要把 `fast` 或 `flex` 误认为“官方 API 开关”；它们只是服务层配置，和 provider、Base URL、鉴权方式是不同维度。

如果每次通过 CC Switch 切换官方配置都会重新写回 `default`，就要修正 CC Switch 的公共配置或供应商模板。只改生成后的 `config.toml` 只能暂时解决。

## 八、官方与第三方配置的安全切换原则

### 1. 不要共用一个含义模糊的全局变量

更清晰的做法是给第三方 Key 使用独立变量：

```text
OPENAI_API_KEY              -> 官方 OpenAI API
THIRD_PARTY_CODEX_API_KEY   -> 第三方 Codex API
```

第三方 provider 使用：

```toml
env_key = "THIRD_PARTY_CODEX_API_KEY"
```

官方 provider 则使用官方认证机制。这样可以减少“第三方 Key 留在 `OPENAI_API_KEY`，但 Base URL 已切回官方”的混用风险。

### 2. 官方 API Key 与 ChatGPT 登录不是一回事

Codex 可能支持两种不同的官方使用方式：

- ChatGPT 账号登录或 OAuth；
- `platform.openai.com` 创建的 OpenAI API Key。

ChatGPT 订阅不自动等于 API Key 余额，API Key 模式也不等同于 ChatGPT 登录。切换官方 provider 时，要明确自己使用的是哪种认证方式，不要把一个无效的 `sk-...` 当成 OAuth 登录结果。

### 3. 切换后按三项配对检查

每次切换供应商后，至少确认：

```text
provider      当前是不是目标供应商
base_url      是否与 provider 对应
credential    当前进程实际能读取到对应 Key
```

不要只看 CC Switch 界面显示“已切换”。真正可靠的依据是 Codex 进程读取的 `config.toml`、认证文件和环境变量，以及请求错误中显示的目标 URL。

## 九、一套不泄露密钥的排查流程

### 第一步：完全退出 Codex GUI

关闭窗口还不一定够，确认托盘和后台进程也退出，避免 GUI 持有旧配置或重新写回 `auth.json`。

### 第二步：在 CC Switch 中切换完整供应商

供应商配置至少要包含正确的 API 地址、协议格式、模型名和认证来源。不要只修改 Base URL。

### 第三步：检查当前 Codex 配置

```powershell
Get-Content "$HOME\.codex\config.toml" |
    Select-String "model_provider|model|base_url|wire_api|env_key|service_tier"
```

### 第四步：只检查认证是否存在

```powershell
$authPath = "$HOME\.codex\auth.json"
if (Test-Path $authPath) {
    $auth = Get-Content $authPath -Raw | ConvertFrom-Json
    if ($auth.OPENAI_API_KEY) {
        "auth.json 中存在 API Key，长度：$($auth.OPENAI_API_KEY.Length)"
    } else {
        "auth.json 中没有 API Key"
    }
}
```

### 第五步：确认 CLI 版本与调用路径

```powershell
codex --version
Get-Command codex -All
where.exe codex
```

### 第六步：新建最小测试对话

不要直接拿一个已经绑定另一套 provider 的旧 GUI 对话测试。先新建会话，发送一条最小请求，再根据错误中的 URL 判断请求是否走到了预期端点。

## 十、常见错误与判断方法

| 错误 | 优先检查 |
| --- | --- |
| `401 invalid_api_key`，URL 是官方端点 | 是否把第三方 Key 发给了官方 URL，或官方 Key 已失效 |
| `401 invalid_api_key`，URL 是第三方端点 | 第三方 Key 是否被撤销、复制错误，或第三方服务商要求特殊鉴权头 |
| `Missing environment variable` | `config.toml` 中的 `env_key` 是否对应当前进程中的变量 |
| `unknown variant default` | CLI 版本与 `service_tier` 字段不兼容 |
| 新对话正常、旧对话 401 | 旧会话可能保留了创建时的 provider 或路由状态 |
| 切换后手动修改又被覆盖 | CC Switch 数据库中的供应商模板或公共配置仍是旧值 |
| 模型不存在或 404 | 当前 Key 所属供应商的模型列表与 `model` 配置不匹配 |

## 十一、最终结论

这次问题不是一个单独的“API Key 错误”，而是多个状态叠加造成的：

1. 第三方 Claude Key 与第三方 Codex Key 不是同一个配置，不能混用。
2. `auth.json`、`config.toml` 和环境变量可能分别来自不同次切换。
3. GUI 旧对话可能继续使用创建时的 provider，新对话才读取当前配置。
4. Codex GUI 与 npm CLI 使用了不同版本，CLI `0.130.0` 无法解析较新配置中的 `service_tier = "default"`。
5. 升级 CLI 或删除不兼容字段，可以解决配置解析错误；但这不会自动修复 API Key 与 Base URL 的错配。

以后排查类似问题时，按“请求 URL → provider → Base URL → 鉴权来源 → 模型 → 客户端版本”的顺序检查，通常比反复更换 API Key 更快定位根因。

最后再次提醒：真实 API Key 一旦出现在错误日志、终端输出、截图或博客草稿中，就应当视为已经泄露，立即撤销并重新生成。
