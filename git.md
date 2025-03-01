## git基本操作

1、初始化仓库

配置ssh：ssh-keygen  -t rsa -b 4096 -C "szh"

git init

2、下载代码

git clone .git  --depth 10

> 大公司可能会用到
>
> git submodule update --init --recursive 对于一些很久的分支，要用这个更新一下子模块
>
> git status 查看状态,确定一下是否有没有track的子模块

3、指定作者和邮箱

git config --global user.name "单章辉"

 git config --global user.email "ci8278489@163.com"



