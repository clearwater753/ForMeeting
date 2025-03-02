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

4、创建新的分支

git checkout -b szh/0301_test

5、放到暂存区和提交

git add xxx.py(vscode不需要用这个命令)

git commit -m "[planner]: message"

6、本地git仓库上传github

清除代理

git config --global --unset http.proxy 

git config --global --unset https.proxy 

添加代理

git config --global http.proxy http://127.0.0.1:10809

7、本地与远程仓库的交互

比如开始时只有本地仓库，那么需要建立远程仓库，然后把远程仓库的内容pull到本地
建立联系

git remote add origin <远程仓库的URL或者git>

拉取远程仓库main分支的内容

git pull origin main

推送main分支(第一次加-u第二次不用加)

git push -u origin main

git push origin main

推送当前分支(两种方式)

git push --set-upstream origin <branch-name>

git push -u origin HEAD

删除远程分支

git push origin --delete <远程分支名>

查看远程分支

git branch -r

查看当前分支

git branch

8、切换分支与创建分支

git checkout main

在当前分支上创建新分支

git checkout -b szh/0301_test

9、撤销提交

撤销当前分支最后一次提交

git reset --soft HEAD^

回撤之后远程仓库报错，那就强制提交（多人项目时要小心使用）

git push -f

10、合并分支

切换到主分支

git checkout main

合并

git merge <branch-name>

11、列出对应的远程仓库

git remote -v

12、git fetch和git pull的区别

git fetch origin --depth 10

git fetch: 是一个相对“保守”的命令，它用于从远程仓库获取最新的分支信息和提交记录，但不会自动合并这些更改到本地分支

git pull origin main

git pull: 是一个更“激进”的命令，它实际上是 git fetch 和 git merge 的组合。它不仅会从远程仓库获取最新的更改，还会立即将这些更改合并到当前的本地分支。

13、合并冲突和合并分支

git merge：将一个分支的更改合并到另一个分支，同时保留分支的历史记录。

git rebase：将一个分支的更改重新应用到另一个分支的顶部，同时“重写”分支的历史记录。

git cherry-pick <comit-hash>: 将某个特定的提交提交到当前分支
