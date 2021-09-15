---
layout: false
---

{% raw %}

<!DOCTYPE html>
	<head>		
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<title>李宏毅-机器学习及其深层与结构化2018</title>
		<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/node-waves@0.7.6/dist/waves.min.css">
		<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@5.10.1/css/all.min.css">
		<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@5.10.1/css/all.min.css">
		<link rel="shortcut icon" type="image/x-icon" href="https://cdn.jsdelivr.net/gh/xaoxuu/assets@master/favicon/favicon.ico">
		<link rel="icon" type="image/x-icon" sizes="32x32" href="https://cdn.jsdelivr.net/gh/xaoxuu/assets@master/favicon/favicons/favicon-32x32.png">
		<link rel="apple-touch-icon" type="image/png" sizes="180x180" href="https://cdn.jsdelivr.net/gh/xaoxuu/assets@master/favicon/favicons/apple-touch-icon.png">
		<link rel="mask-icon" color="#1BC3FB" href="https://cdn.jsdelivr.net/gh/xaoxuu/assets@master/favicon/favicons/safari-pinned-tab.svg">
		<link rel="manifest" href="https://cdn.jsdelivr.net/gh/xaoxuu/assets@master/favicon/favicons/site.webmanifest">
		<link href="../css/style.css" rel="stylesheet" type="text/css">
		<link href="../css/sidebar.css" rel="stylesheet" type="text/css">
		<link href="../css/header.css" rel="stylesheet" type="text/css">
		<link href="../css/footer.css" rel="stylesheet" type="text/css">
	</head>
	<body>
	    <div class="cover-wrapper">
            <cover class="cover half" style="position: relative; z-index: 0; background: none;">
                <h1 class='title'>Rogerspy's Home</h1>
                <div class="m_search">
                    <form name="searchform" class="form u-search-form">
                        <input type="text" class="input u-search-input" placeholder="" />
                        <i class="icon fas fa-search fa-fw"></i>
                    </form>
                </div>
				<div class='menu navgation'>
					<ul class='h-list'>
						<li>
							<a class="nav home" href="/" id="home">
								<i class='fas fa-edit fa-fw'></i>&nbsp;博文
							</a>
						</li>
						<li>
							<a class="nav home active" href="/video/" id="video">
								<i class='fas fa-film fa-fw'></i>&nbsp;视频
							</a>
						</li>
						<li>
							<a class="nav home" href="/material/" rel="nofollow" id="material">
								<i class='fas fa-briefcase fa-fw'></i>&nbsp;资料
							</a>
						</li>
						<li>
							<a class="nav home" href="/about/" rel="nofollow" id="about">
								<i class='fas fa-info-circle fa-fw'></i>&nbsp;关于
							</a>
						</li>
					</ul>
				</div>
				<div class="backstretch" style="left: 0px; top: 0px; overflow: hidden; margin: 0px; padding: 0px; height: 412px; width: 100%; z-index: -999998; position: absolute;">
				    <img src="https://img.vim-cn.com/6d/a0c9e6f9efad8b731cb7376504bd10d79d2053.jpg" style="position: absolute; margin: 0px; padding: 0px; border: none; width: 100%; height: 100%; max-height: none; max-width: none; z-index: -999999; left: 0px; top: 0px;">
				</div>
            </cover>
            <header class="l_header pure">
	            <div class='wrapper'>
					<div class="nav-main container container--flex">
				        <a class="logo flat-box waves-effect waves-block" href='/' >
					        Rogerspy's Home
					    </a>
						<div class='menu navgation'>
							<ul class='h-list'>
								<li>
									<a class="nav flat-box" href="/blog/" id="blog">
										<i class='fas fa-edit fa-fw'></i>&nbsp;博客
									</a>
								</li>
								<li>
									<a class="nav flat-box active" href="/video/" id="video">
										<i class='fas fa-film fa-fw'></i>&nbsp;视频小站
									</a>
								</li>
								<li>
									<a class="nav flat-box" href="/material/" id="material">
										<i class='fas fa-briefcase fa-fw'></i>&nbsp;学习资料
									</a>
								</li>
								<li>
									<a class="nav flat-box" href="/diary/" id="diary">
										<i class='fas fa-book fa-fw'></i>&nbsp;随心记
									</a>
								</li>
								<li>
									<a class="nav flat-box" href="/categories/" rel="nofollow" id="categories">
										<i class='fas fa-folder-open fa-fw'></i>&nbsp;分类
									</a>
								</li>
								<li>
									<a class="nav flat-box" href="" rel="nofollow" id="tags">
										<i class='fas fa-hashtag fa-fw'></i>&nbsp;标签
									</a>
								</li>
								<li>
									<a class="nav flat-box" href="/tags/" rel="nofollow" id="blogarchives">
										<i class='fas fa-archive fa-fw'></i>&nbsp;归档
									</a>
								</li>
							</ul>
						</div>
						<div class="m_search">
							<form name="searchform" class="form u-search-form">
								<input type="text" class="input u-search-input" placeholder="Search" />
								<i class="icon fas fa-search fa-fw"></i>
							</form>
						</div>
		            </div>
	            </div>
	        </header>
		</div>
		<div class="container-x", id="box">
			<div class="scrollbar" id="style-1">
				<div class="navbox">
					<ul class="nav">
						<li><a href="index.html">为什么需要深层网络？</a></li>
						<li><a href="potential_deep.html">深层结构的潜力</a></li>
						<li><a href="is_deep_better.html">深层结构比浅层的好吗？</a></li>
						<li><a href="gd_is_zero.html">当梯度为零时</a></li>
						<li><a href="deep_linear.html">深度线性网络</a></li>
						<li><a href="local_min.html">深层网络有局部最小化吗？</a></li>
						<li><a href="geometry_lossi.html">损失表面几何（猜想）</a></li>
						<li><a href="geometry_lossii.html">损失表面几何（经验）</a></li>
						<li><a href="generalization.html">深度学习泛化能力</a></li>
						<li><a href="indicator.html">泛化能力指示器</a></li>
						<li><a href="computional_graph.html">计算图</a></li>
						<li><a href="seq2seq.html">序列到序列学习</a></li>
						<li><a href="pointer_network.html">指针网络</a></li>
						<li><a href="resursive_network.html">递归网络</a></li>
						<li><a href="attention_based.html">基于注意力的模型</a></li>
						<li><a href="bn.html">Batch Normalization</a></li>
						<li><a href="l2l.html">learing2learn</a></li>
						<li><a href="intro.html">GAN-简介</a></li>
						<li><a href="cond_gen.html">GAN—条件生成</a></li>
						<li><a href="unsupervise_cond_gen.html">GAN-无监督条件生成</a></li>
						<li><a href="theory.html">GAN-基础理论</a></li>
						<li><a href="general_framework.html">GAN-通用框架</a></li>
						<li><a href="wgan.html">GAN-WGAN, EBGAN</a></li>
						<li><a href="infogan.html">GAN-InfoGAN, VAE-GAN, BiGAN</a></li>
						<li><a href="photo_editiing.html">GAN-照片编辑</a></li>
						<li><a href="seq_gen.html">GAN-序列生成</a></li>
						<li><a href="evaluation.html">GAN-评估</a></li>
						<li><a href="ppo.html">DRL-Proximal Policy Optimization (PPO)</a></li>
						<li><a href="qlearning.html">DRL-Q-Learning</a></li>
						<li><a href="actor_critic.html">DRL-Actor-critic</a></li>
						<li><a href="sparse_reward.html">DRL-Sparse Reward</a></li>
						<li><a href="imitation_learn.html">DRL-Imitation Learning</a></li>
					</ul>
				</div>
			</div>
			<section class="video-section video-section--light" id="demo">
			    <h4 class="title-video heading-video">深层结构的潜力</h4>
				<div class="container-video">
				    <iframe src="https://player.bilibili.com/player.html?aid=24382437&cid=40967909&page=2" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
			    </div>
			</section>
		</div>
		<footer id="footer" class="clearfix">
		    <div id="sitetime"></div>
			<div class="social-wrapper">
				<a href="/atom.xml" class="social fas fa-rss flat-btn" target="_blank" rel="external nofollow noopener noreferrer"></a>
				<a href="mailto:rogerspy@163.com" class="social fas fa-envelope flat-btn" target="_blank" rel="external nofollow noopener noreferrer"> </a>
				<a href="https://github.com/rogerspy" class="social fab fa-github flat-btn" target="_blank" rel="external nofollow noopener noreferrer"></a>
				<a href="https://music.163.com/#/user/home?id=1960721923" class="social fas fa-headphones-alt flat-btn" target="_blank" rel="external nofollow noopener noreferrer"> </a>
			</div>
		    <br>
		    <div>
			    <p>Blog content follows the <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en">Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License</a></p>
		    </div>
		    <div>
			    Use <a href="https://xaoxuu.com/wiki/material-x/" target="_blank" class="codename">Material X</a>
			    as theme, total visits <span id="busuanzi_value_site_pv"><i class="fas fa-spinner fa-spin fa-fw" aria-hidden="true"></i></span> times. 
		    </div>
	    </footer>
		<script type="text/javascript" src="/cool/cooltext.js"></script>
		<script type="text/javascript" src="/cool/clicklove.js"></script>
		<script type="text/javascript" src="/cool/sitetime.js"></script>
		<script src="https://cdn.jsdelivr.net/npm/jquery@3.3.1/dist/jquery.min.js"></script>
		<script async src="https://cdn.jsdelivr.net/gh/xaoxuu/cdn-busuanzi@2.3/js/busuanzi.pure.mini.js"></script>
		<script src="https://cdn.jsdelivr.net/gh/xaoxuu/cdn-material-x@19.5/js/app.js"></script>
        <script src="https://cdn.jsdelivr.net/gh/xaoxuu/cdn-material-x@19.5/js/search.js"></script>
	</body>
</html>

{% endraw %}