---
title: Git使用ssh连接
categories:
  - Ubuntu个性化
tags:
  - Ubuntu
readmore: true
hideTime: true
abbrlink: 9b6d1847
date: 2026-01-21 10:48:39
---

## 一、什么是SSH key，为什么要配置？

- **原理说明：**  
    SSH key像是一把“虚拟钥匙”，由一对密钥（公钥和私钥）组成。公钥（放在GitHub上），私钥（保存在你的电脑上）。  
    配置以后，你每次连接GitHub，不用再输入用户名密码，系统自动用“钥匙”验证是不是你本人，更安全也更方便！

---

## 二、步骤详解

### 1. 检查你电脑上有没有现成的SSH key

- 打开命令行（Windows可以用Git Bash，Mac直接用终端Terminal）
  
- 输入下面命令：
  
    ```
    ls ~/.ssh
    ```
    
- 如果看到有`id_rsa`和`id_rsa.pub`或者`id_ed25519`和`id_ed25519.pub`，说明已经有key了，可以直接用；没有就继续下一步。
  

> `.pub`是公钥，没后缀的是私钥。

---

### 2. 生成新的SSH key（如果没有）

- 输入：
  
    ```
    ssh-keygen -t ed25519 -C "你的邮箱地址"
    ```
    
    - 这里`-t ed25519`是选择一种更安全的算法，`-C`是加备注（通常写你的邮箱）。
- 提示你输入保存文件名，直接回车（默认就好）。
  
- 让你设置密码，直接回车跳过（有需要可以设置，方便一般不用）。
  
- 完成后，会生成类似`id_ed25519`和`id_ed25519.pub`这两个文件。
  

**原理说明：**

- 公钥（.pub）可以分享，别人拿它验证你是谁。
- 私钥（不带.pub的）只有你自己保存，千万别泄露。

---

### 3. 把公钥添加到GitHub账号

- 打开命令行，输入下面命令，把公钥内容复制出来：
  
    ```
    cat ~/.ssh/id_rsa.pub
    ```
    
- 拷贝输出的整行内容！
  
- 登录你的GitHub账号，进入：  
    右上角头像 -> Settings -> SSH and GPG keys -> New SSH key
    
- 粘贴刚才复制的内容到“Key”那里，起个名字（随便，比如“My Laptop”），点击“Add SSH key”。
  

**原理说明：**

- 给GitHub你的公钥，相当于让GitHub认你的钥匙。
- 以后你用私钥打开门（登录），GitHub用公钥验证是不是你。

---

### 4. 测试一下是否成功

- 在命令行输入：
  
    ```
    ssh -T git@github.com
    ```
    
- 第一次连接，可能会提示你“Are you sure you want to continue connecting”，输入`yes`回车。
  
- 如果看到`Hi xxx! You've successfully authenticated...`，说明配置成功！
## 三、解决“ssh:connect to host github.com port 22: Connection timed out”
有时候使用 SSH 连接 GitHub 时，出现如下报错：

> ssh: connect to host github.com port 22: Connection timed out

无论尝试多少次，甚至重启电脑都没用。很多情况下，这是因为防火墙或网络环境把 22 端口封掉了。

### 常见解决方法

#### 方法一：改用 HTTPS 连接方式

1. 输入命令：
   
    ```
    git config --local -e
    ```
    
2. 将配置文件的 `url = git@github.com:username/repo.git` 一行改为  
    `url = https://github.com/username/repo.git`
3. 保存退出即可。

![git配置示意](https://i-blog.csdnimg.cn/blog_migrate/45a2698e9da4ed694f18b4389dc3e008.png)

---

#### 方法二：更改 SSH 端口（推荐）

1. 进入 ~/.ssh 目录
   
    ```
    cd ~/.ssh
    ```
    
2. 新建或编辑 config 文件（如用 vim 编辑器）：
   
    ```
    vim config
    ```
    
3. 写入以下内容（如果用的是 id_rsa，留意 IdentityFile 的文件名；注意 gitlab 配置可选）：
   
    ```
    Host github.com
      User git
      Hostname ssh.github.com
      PreferredAuthentications publickey
      IdentityFile ~/.ssh/id_rsa
      Port 443
    
    Host gitlab.com
      Hostname altssh.gitlab.com
      User git
      Port 443
      PreferredAuthentications publickey
      IdentityFile ~/.ssh/id_rsa
    ```
    
4. 保存并退出。
   
5. 检查配置是否生效：
   
    ```
    ssh -T git@github.com
    ```
    
    按提示操作（一般会要你输入 `yes`）。
    
1. 再次尝试 `git push` 操作，应该就可以顺利提交了！
## 四、小结

- SSH Key 是 Git 安全高效访问 GitHub 的推荐方案。
- 若因端口封锁连不上，可通过更换端口解决，不必放弃 SSH 方式。
- 只要保护好自己的私钥，GitHub 账号安全系数非常高！