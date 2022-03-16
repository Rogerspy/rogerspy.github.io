---
layout: page
title: 关于小站
body: [article, grid, comments]
gitalk:
  placeholder: 有什么想对我说的呢？
sidebar: false
---

> 本站搭建过程参考了很多大佬的博客，在此向大佬们表示感谢！这里介绍一下本站搭建的一些配置与主题的优化。

# 1. 安装Node.js

首先下载稳定版[Node.js](https://nodejs.org/dist/v9.11.1/node-v9.11.1-x64.msi)，我这里给的是64位的。

安装选项全部默认，一路点击`Next`。

最后安装好之后，按`Win+R`打开命令提示符，输入`node -v`和`npm -v`，如果出现版本号，那么就安装成功了。

如果没有梯子的话，可以使用阿里的国内镜像进行加速。

```bash
npm config set registry https://registry.npm.taobao.org
```

# 2. 安装Git

为了把本地的网页文件上传到github上面去，我们需要用到分布式版本控制工具————Git[[下载地址\]](https://git-scm.com/download/win)。

安装选项还是全部默认，只不过最后一步添加路径时选择`Use Git from the Windows Command Prompt`，这样我们就可以直接在命令提示符里打开git了。

安装完成后在命令提示符中输入`git --version`验证是否安装成功。

# 3. 配置Github Pages

首先是要注册一个github账号，链接：https://github.com/，按照步骤填写一些信息完成注册即可。

接下来新建一个项目：

![](https://img.vim-cn.com/97/eb87e38f2627cf30c29d7ca8533acf4a622be7.png)

然后输入项目名字，注意：后面一定要加上`.github.io`后缀，否可能引起不可知的bug。

![](https://img.vim-cn.com/68/7707eb2df7ca9d572725acdabe6ffbae503c63.png)

点击`create repository`这样项目就建好了。

项目建好以后，点击`Settings`，向下拉到最后有个`GitHub Pages`，点击`Choose a theme`选择一个主题。

![](https://img.vim-cn.com/10/5f5db98b80df4dea30cafc8c615f399b1557b9.png)

然后等一会儿，再回到`GitHub Pages`，会变成下面这样：

![](https://img.vim-cn.com/27/f20529fa20afe05d15643f443ae2924d90e6de.png)



#  4. 安装Hexo

在合适的地方新建一个文件夹，用来存放自己的博客文件，比如我的博客文件都存放在`D:\blog`目录下。

在该目录下右键点击`Git Bash Here`，打开git的控制台窗口

定位到该目录下，输入`npm i hexo-cli -g`安装Hexo

安装完后输入`hexo -v`验证是否安装成功。

然后就要初始化我们的网站，输入`hexo init`初始化文件夹，接着输入`npm install`安装必备的组件。

这样本地的网站配置也弄好啦，输入`hexo g`生成静态网页，然后输入`hexo s`打开本地服务器，然后浏览器打开http://localhost:4000/，就可以看到我们的博客啦，效果如下：

![åå®¢é»è®¤é¦é¡µé¡µé¢](https://img-blog.csdnimg.cn/20190118141145968.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1V0b3BpYU9mQXJ0b3JpYQ==,size_16,color_FFFFFF,t_70)



# 5. 连接Github与本地

首先右键打开git bash，然后输入下面命令：

```bash
git config --global user.name "yourname"
git config --global user.email "your-email"
```

然后生成密钥SSH key：

```bash
ssh-keygen -t rsa -C "your-email"
```

打开[github](https://github.com/)，在头像下面点击`settings`，再点击`SSH and GPG keys`，新建一个SSH，名字随便。

git bash中输入

```bash
cat ~/.ssh/id_rsa.pub
```

如果是 windows用户，`id_rsa.pub`一般会在`c:/user`下面。

将输出的内容复制到框中，点击确定保存。

打开博客根目录下的`_config.yml`文件，这是博客的配置文件，在这里你可以修改与博客相关的各种信息。

修改最后一行的配置：

```bash
deploy:
  type: git
  repository: https://github.com/yourname/abc.github.io
  branch: master
```

repository修改为你自己的github项目地址。



# 6. 写/发布文章

- 安装`hexo-deployer-git`

```
npm i hexo-deployer-git
```

- 新建文章

```
hexo new post "article title"
```

- 打开博客根目录下的`source/_post`目录，可以发现多了一个`.md`文件，这个文件就是你刚刚新建的文章，然后就可以往里面写入你的文章
- 写完`.md`文件以后

```
hexo g  # 生成静态网页
hexo s  # 启动本地服务
```

- 打开`localhost:4000`就可以预览刚刚写的文章
- 如果没有问题

```
hexo d  # 上传到github
```

这样就可以在刚刚生成abc.github.io上看到 你的文章，到这里文章就发布完毕了。

因为我没有购买私有化域名，所以就不介绍了。



#  7. 个性化配置

本博客用的主题是`Material-X`，

作者主页：https://xaoxuu.com/wiki/material-x/

Github地址: https://github.com/xaoxuu/hexo-theme-material-x

##  7.1 更换主题

作者提供了两种方式：

 **A. 使用脚本全自动安装（目前仅支持macOS）**

1. 打开终端输入下面命令安装脚本，脚本文档见[#hexo.sh](https://xaoxuu.com/wiki/hexo.sh/)。

   ```
   curl -s https://xaoxuu.com/install | sh -s hexo.sh
   ```

2. 安装成功后，在你的博客路径打开终端，输入下面命令即可安装主题和依赖包。

   ```
   hexo.sh i x
   ```

**B. 手动安装**

1. 下载主题到 `themes/` 文件夹

   ```
   git clone https://github.com/xaoxuu/hexo-theme-material-x themes/material-x
   ```

2. 然后安装必要的依赖包

   ```
   npm i -S hexo-generator-search hexo-generator-json-content hexo-renderer-less
   ```

## 7.2 文章头设置

为了新建文章方便，建议把博客根目录下`/scaffolds/post.md`修改为：

```
---
type: blog
title: {{ title }}
date: {{ date }}
top: false
cover: false
toc: true
mathjax: true
summary:
tags:
categories:
body: [article, comments]
gitalk:
  id: /wiki/material-x/c
---
```

这样新建文章后不用你自己补充了，修改信息就行。至于其中每一个字段的意义可以参考主题作者的说明文档。

## 7.3 添加404页面

原来的主题没有404页面，加一个也不是什么难事。首先在`/source/`目录下新建一个`404.md`，内容如下：

```
---
layout: page
title: 404 Not Found
body: [article, comments]
meta:
  header: false
  footer: false
sidebar: false
gitalk:
  path: /404.html
  placeholder: 请留言告诉我您要访问哪个页面找不到了
---

# <center>404 Not Found</center>

<br>

<center>很抱歉，您访问的页面不存在</center>
<center>可能是输入地址有误或该地址已被删除</center>

<br>
<br>
```

## 7.4 Mathjax数学公式渲染优化

这部分的优化主要参考了[hexo中的mathjax数学公式渲染优化](https://wxwoo.top/2019/05/15/hexo-mathjax-renderer-optimization/)这篇文章。

### mathjax渲染修改

#### 1. 修改渲染引擎

更改hexo的默认渲染引擎，使其支持mathjax

打开cmd，cd到hexo博客文件夹下，输入

```
npm uninstall hexo-renderer-marked --save
npm install hexo-renderer-kramed --save
```

#### 2. 更改配置

找到`/node_modules/hexo-renderer-kramed/lib/renderer.js`，将

```
function formatText(text) {
    // Fit kramed's rule:  $$ + \1 + $$ 
    return text.replace(/`\ $(.*?)\$ `/g, ' $$$$$ 1 $$$$ ');
}
```

改为

```
function formatText(text) {
    return text;
}
```

#### 3. 修改数学包

在cmd中输入

```
npm uninstall hexo-math --save
npm install hexo-renderer-mathjax --save
```

#### 4. 更新mathjax配置文件

找到`/node_modules/hexo-renderer-mathjax/mathjax.html`，将最下面那一行`<script>`注释掉，改成

```
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>
```

#### 5. 修改转义规则

因为markdown和mathjax语法有冲突，我们修改转义规则以避免冲突

找到`\node_modules\kramed\lib\rules\inline.js`，将`escape`和`em`这两行注释掉，改成

```
escape: /^\\([`*\[\]()# +\-.!_>])/,
em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
```

#### 6. 开始使用

找到`\_config.yml`，加上一行`mathjax: true`就可以了

### mathjax渲染优化

其实这应该就结束了

但不知什么原因，部分情况直接渲染出来会有一个灰色的框，十分难看

所以我们还要进行优化

打开`/node_modules/hexo-renderer-mathjax/mathjax.html`，在`MathJax.Hub.Config`中加上

```
extensions: ["tex2jax.js"],
jax: ["input/TeX", "output/HTML-CSS"],
```

将刚刚第4步修改的链接修改为

```
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_HTMLCSS"></script>
```

就可以了

但是仍然有一些不尽人意的地方，比如：

### 部分公式渲染出错

比如这个公式：

```
$\dfrac{1}{2}$
```

显示`Undefined control sequence \dfrac`

不仅是这个，还有很多会出错，比如`\geqslant`

这是由于更改链接后，缺少了`amsmath`包

在`MathJax.Hub.Config`中加上

```
TeX: { 
    equationNumbers: { autoNumber: "AMS" },
    extensions: ["AMSmath.js", "AMSsymbols.js"]
},
```

即可解决

### 修改字体

在`MathJax.Hub.Config`中加上

```
"HTML-CSS": {
	preferredFont: "TeX", 
    availableFonts: ["STIX","TeX"]
}
```

### 关闭右下角加载信息

在`MathJax.Hub.Config`中加上

```
showProcessingMessages: false,
messageStyle: "none",
```

### 关闭右键菜单

在`"HTML-CSS"`中加上

```
showMathMenu: false
```

### 代码整合

可以根据需要自行修改

```
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        TeX: { 
            equationNumbers: { autoNumber: "AMS" },
            extensions: ["AMSmath.js", "AMSsymbols.js"]
        },
        "HTML-CSS": {
            preferredFont: "TeX", 
            availableFonts: ["STIX", "TeX"],
            showMathMenu: false
        },
        showProcessingMessages: false,
		messageStyle: "none",
        extensions: ["tex2jax.js"],
        jax: ["input/TeX", "output/HTML-CSS"],
        tex2jax: { 
            inlineMath: [ ["$","$"] ],  
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'a']
        }
    });
    MathJax.Hub.Queue(function() {
        var all = MathJax.Hub.getAllJax();
        for (var i = 0; i < all.length; ++i)
            all[i].SourceElement().parentNode.className += ' has-jax';
    });
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_HTMLCSS"></script>
```

## 7.5 其他美化

其他的美化包括，添加血小板，鼠标单击效果，输入特效等都是参考的[Hexo博客Material-X主题个性化](https://blog.treelo.xin/2019/05/12/themes-custom/)这篇文章。

### 添加血小板

**添加方式**

1. `live2d`资源目录添加至主目录`/source`下
2. 在主题目录`/layout/_partial/head.ejs`文件中引入`live2.css`文件

```
<!-- 血小板-->
<link rel="stylesheet" href="/live2d/css/live2d.css" />
```

3. 在主题目录`/layout/_partial/footer.ejs`文件中添加以下代码

```
<!-- 血小板 -->
<div id="landlord">
  <div class="message" style="opacity:0"></div>
  <canvas id="live2d" width="560" height="500" class="live2d"></canvas>
  <div class="hide-button">隐藏</div>
</div>
<!-- 血小板-->
<script type="text/javascript">
  var message_Path = '/live2d/'
</script>
<script type="text/javascript" src="/live2d/js/live2d.js"></script>
<script type="text/javascript" src="/live2d/js/message.js"></script>
<script type="text/javascript">
  loadlive2d("live2d", "/live2d/model/xiaoban/model.json");
</script>
```

4. 为了移动端更好的阅读效果，请将以下代码添加至主题目录/source/less/_footer.less文件下

```
@media (max-width: @on_phone) {
  #footer{
      background-color:transparent;
      padding-bottom: 180px ;
  }
  #landlord{
      width: 200px;
      height: 170px;
      .message{
        width: 200px;
        left: 43px;
        top: 15px;
    }
  }
  #live2d{
      width: 200px;
      height: 170px;
      bottom: -80px;
      left: 43px;
  }
}
```

### 添加页面点击小心心特效,文本输入特效、运行时间

**添加方式**
注意：由于博客使用了防盗链，请将脚本另存为
1.将 `https://blog.treelo.xin/cool/clicklove.js` 另存在主题目录/source/下
2.将 `https://blog.treelo.xin/cool/cooltext.js` 另存在主题目录/source/下
3.将 `https://blog.treelo.xin/cool/sitetime.js` 另存在主题目录/source/下
4.修改`sitetime.js`参数
5.在主题目录`/layout/_partial/footer.ejs`文件中引入

```
<!-- 点击特效，输入特效 运行时间 -->
<script type="text/javascript" src="/cool/cooltext.js"></script>
<script type="text/javascript" src="/cool/clicklove.js"></script>
<script type="text/javascript" src="/cool/sitetime.js"></script>
```

6.在主题目录`/layout/_partial/footer.ejs`文件中上方添加

```
<div id="sitetime"></div>
```

## 8. 2021.8 附加一：图床搭建

## 8.1 下载 picgo

这里是最新版本的项目发布地址：

```
https://github.com/Molunerfinn/PicGo/releases
```

如果是`macos`系统，请选择`dmg`版本下载，`windows`系统请选择`exe`版本下载。

## 8.2 配置 picgo

1. 首先，我们要在 github 中新建一个仓库，这个仓库可以随意命名，建议名字短一点，这样最后生成的链接也更短。（注：一定要创建公共仓库，私有仓库是无法通过外链显示的)
2. 创建好后，我们需要在 github 上生成一个 token，这样 PicGo 才能直接对 github 仓库内容进行操作。
   - 点击头像–> setting –> Developer settings
   - 在侧边栏选择 Personal access tokens
   - 点击 Generate new token，输入密码，进入创建 token 的界面
   - Note 可以随便填，建议填 PicGo
   - 将repo这一栏的权限全部勾选，其他的不用动。
   - 拖到最底下，点击 Generate token
3. 创建好 token 后，我们需要记录下这个 token，因为它只会出现一次。
4. 接下来，我们进入 PicGo 的主程序中，找到 Github 图床。
5. 按照以下内容填入：
   ![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9jZG4uanNkZWxpdnIubmV0L2doL3NjaHJvZGluZ2VyY2F0c3MvRmlndXJlQmVkQG1hc3Rlci9pbWcvVElNJUU2JTg4JUFBJUU1JTlCJUJFMjAyMDA2MTUxNjU0NTcucG5n?x-oss-process=image/format,png)
   - 仓库名：github用户名/仓库名
   - 分支：填 master
   - Token:填你之前创建的那个Token
   - 指定存储路径：这个可以随便填，也可以不填
   - 设定自定义域名：https://cdn.jsdelivr.net/gh/github用户名/仓库名@master

## 8.3 jsdelivr 加速

因为github的服务器在国外，所以用来当图床的时候，国内的速度非常感人，但是我们可以使用`jsdelivr`提供的CDN 服务，速度非常的快，基本不大的图片可以秒开。

所以上述自定义域名实际上是使用的`jsdelivr`的CDN服务。

## 8.4 得到图片的直链

这样，我们就可以直接在PicGo的上传区将要传入的图片拖入即可。

然后，在相册区，我们可以复制对应图片的外链。

还可以选择相册顶部的小倒三角，选择链接的各种，有`markdown`,`html`，`URL`等方式供选择。

# 9. 2021.8 附加二：Excalidraw-Claymate

1. 下载

   ```
   git clone https://github.com/dai-shi/excalidraw-claymate
   ```

2. 新建一个仓库

3. 在新仓库里启动 github actions

4. 将刚刚拉下来的代码 push 到新仓库里

5. 在新仓库 settings 中找到 pages，选择部署的文件。

# 10. 2021.8 附加三：字数统计

先在博客目录下执行以下命令安装 hexo-wordcount 插件：

```
$ npm i --save hexo-wordcount
```

注意：在 [Material X](https://xaoxuu.com/wiki/material-x/) 主题中，字数统计和阅读时长的功能我已提交 PR，在最新版本中，只需要安装插件后，在主题 `config.yml` 配置文件里，将 `word_count` 关键字设置为 `true` 即可，对于旧版本，可以通过以下方法实现：

以 [Material X](https://xaoxuu.com/wiki/material-x/) 主题（版本 1.2.1）为例，在 \themes\material-x\layout\_meta 目录下创建 word.ejs 文件，在 word.ejs 文件中写入以下代码:

```
<% if(isPostList || !isPostList){ %>
  <% if (theme.word_count && !post.no_word_count) { %>
    <div style="margin-right: 10px;">
      <span class="post-time">
        <span class="post-meta-item-icon">
          <i class="fa fa-keyboard"></i>
          <span class="post-meta-item-text">  字数统计: </span>
          <span class="post-count"><%= wordcount(post.content) %>字</span>
        </span>
      </span>
      &nbsp; | &nbsp;
      <span class="post-time">
        <span class="post-meta-item-icon">
          <i class="fa fa-hourglass-half"></i>
          <span class="post-meta-item-text">  阅读时长≈</span>
          <span class="post-count"><%= min2read(post.content) %>分</span>
        </span>
      </span>
    </div>
  <% } %>
<% } %>
```

然后在主题的配置文件 _config.yml 找到 meta 关键字，将 word 填入 header 中：

```
meta:
  header: [title, author, date, categories, tags, counter, word, top]
  footer: [updated, share]
```

最后在主题目录下的 _config.yml 添加以下配置即可

```
word_count: true
```

# 11. hexo中的mathjax数学公式渲染优化

> 转载自 [hexo中的mathjax数学公式渲染优化](https://wxwoo.top/2019/05/15/hexo-mathjax-renderer-optimization/)

在使用hexo博客和material-x等博客主题时，难免会遇到mathjax数学公式渲染失败或者与markdown渲染冲突的问题。

xaoxuu给出了解决方案，只需在`_config.yml`里加入`mathjax: true`即可解决，可以解决大量的mathjax公式渲染，但仍有部分复杂的公式渲染出现问题。

我在这里给出一种解决方案。

*注：部分资料来自互联网*

## 11.1 mathjax渲染修改

### 11.1.1 修改渲染引擎

更改hexo的默认渲染引擎，使其支持mathjax

打开cmd，cd到hexo博客文件夹下，输入：

```bash
npm uninstall hexo-renderer-marked --save
npm install hexo-renderer-kramed --save
```

### 11.1.2 更改配置

找到`/node_modules/hexo-renderer-kramed/lib/renderer.js`，将

```js
function formatText(text) {
    // Fit kramed's rule:  $$ + \1 + $$ 
    return text.replace(/`\ $(.*?)\$ `/g, ' $$$$$ 1 $$$$ ');
}
```

改为

```js
function formatText(text) {
    return text;
}
```

### 11.1.3 修改数学包

在cmd中输入

```bash
npm uninstall hexo-math --save
npm install hexo-renderer-mathjax --save
```

### 11.1.4 更新mathjax配置文件

找到`/node_modules/hexo-renderer-mathjax/mathjax.html`，将最下面那一行`<script>`注释掉，改成

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML"></script>
```

### 11.1.5 修改转义规则

因为 markdown 和 mathjax 语法有冲突，我们修改转义规则以避免冲突

找到`\node_modules\kramed\lib\rules\inline.js`，将`escape`和`em`这两行注释掉，改成

```js
escape: /^\\([`*\[\]()# +\-.!_>])/,

em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
```

## 6. 开始使用

找到`\_config.yml`，加上一行`mathjax: true`就可以了。

## 11.2 mathjax渲染优化

其实这应该就结束了

但不知什么原因，部分情况直接渲染出来会有一个灰色的框，十分难看

所以我们还要进行优化

打开`/node_modules/hexo-renderer-mathjax/mathjax.html`，在`MathJax.Hub.Config`中加上

```html
extensions: ["tex2jax.js"],
jax: ["input/TeX", "output/HTML-CSS"],
```

将刚刚第4步修改的链接修改为

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_HTMLCSS"></script>
```

就可以了

但是仍然有一些不尽人意的地方，比如：

### 11.2.1 部分公式渲染出错

比如这个公式：

```markdown
$\dfrac{1}{2}$
```

显示`Undefined control sequence \dfrac`

不仅是这个，还有很多会出错，比如`\geqslant`

这是由于更改链接后，缺少了`amsmath`包

在`MathJax.Hub.Config`中加上

```js
TeX: { 
    equationNumbers: { autoNumber: "AMS" },
    extensions: ["AMSmath.js", "AMSsymbols.js"]
},
```

即可解决。

### 11.2.2 修改字体

在`MathJax.Hub.Config`中加上

```js
"HTML-CSS": {
	preferredFont: "TeX", 
    availableFonts: ["STIX","TeX"]
}
```

~~其实这没什么用~~。

### 11.2.3 关闭右下角加载信息

在`MathJax.Hub.Config`中加上

```js
showProcessingMessages: false,
messageStyle: "none",
```

### 11.2.4 关闭右键菜单

在`"HTML-CSS"`中加上

```js
showMathMenu: false
```

### 11.2.5 代码整合

可以根据需要自行修改

```html
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        TeX: { 
            equationNumbers: { autoNumber: "AMS" },
            extensions: ["AMSmath.js", "AMSsymbols.js"]
        },
        "HTML-CSS": {
            preferredFont: "TeX", 
            availableFonts: ["STIX", "TeX"],
            showMathMenu: false
        },
        showProcessingMessages: false,
		messageStyle: "none",
        extensions: ["tex2jax.js"],
        jax: ["input/TeX", "output/HTML-CSS"],
        tex2jax: { 
            inlineMath: [ ["$","$"] ],  
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'a']
        }
    });
    MathJax.Hub.Queue(function() {
        var all = MathJax.Hub.getAllJax();
        for (var i = 0; i < all.length; ++i)
            all[i].SourceElement().parentNode.className += ' has-jax';
    });
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_HTMLCSS"></script>
```

