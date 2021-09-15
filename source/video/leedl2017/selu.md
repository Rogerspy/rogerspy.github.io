---
layout: false
---

{% raw %}

<!DOCTYPE html>
	<head>		
		<meta charset="utf-8">
		<meta name="viewport" content="width=device-width, initial-scale=1.0">
		<title>李宏毅-深度学习2017</title>
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
						<li><a href="index.html">深度学习模型基本结构（1）</a></li>
						<li><a href="dl_structure2.html">深度学习模型基本结构（2）</a></li>
						<li><a href="computional_graph_bp.html">计算图和后向传播</a></li>
						<li><a href="dllm.html">深度学习语言模型</a></li>
						<li><a href="transformer.html">Transformer</a></li>
						<li><a href="highway.html">High-way Network & Grid LSTM</a></li>
						<li><a href="rnn.html">Recursive Network</a></li>
						<li><a href="rnn_generation.html">RNN和Attention条件生成</a></li>
						<li><a href="pointer_network.html">指针网络</a></li>
						<li><a href="bm.html">Batch Normalization</a></li>
						<li><a href="selu.html">SELU</a></li>
						<li><a href="capsule.html">Capsule</a></li>
						<li><a href="hyperparameter.html">超参数微调</a></li>
						<li><a href="interesting.html">深度学习中一些有趣的事</a></li>
						<li><a href="gan.html">GAN</a></li>
						<li><a href="imporved_gan.html">Improved GAN</a></li>
						<li><a href="rl_gan_generateion.html">RL和GAN用于句子生成和对话系统</a></li>
						<li><a href="ml_beauty.html">机器学习美少女</a></li>
						<li><a href="imitation_learning.html">模仿学习</a></li>
						<li><a href="evaluation.html">生成模型评估</a></li>
						<li><a href="ensembel_gan.html">GAN集成</a></li>
						<li><a href="energy_gan.html">基于能量的GAN</a></li>
						<li><a href="video_gan.html">GAN生成视频</a></li>
						<li><a href="a3c.html">A3C</a></li>
						<li><a href="gated_rnn.html">门RNN和序列生成</a></li>
					</ul>
				</div>
			</div>
			<section class="video-section video-section--light" id="demo">
			    <h4 class="title-video heading-video">SELU</h4>
				<div class="container-video">
				    <iframe src="https://player.bilibili.com/player.html?aid=9770302&cid=16150508&page=11" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
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