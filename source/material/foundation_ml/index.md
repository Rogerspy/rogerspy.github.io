---
layout: false
---

{% raw %}

<!DOCTYPE html>
<head>		
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title>机器学习基础</title>
	<link rel="stylesheet" href="https://cdn.bootcss.com/font-awesome/4.6.3/css/font-awesome.min.css" >
	<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/node-waves@0.7.6/dist/waves.min.css">
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
	<link href="../css/list.css" rel="stylesheet" type="text/css">
	<link href="../css/bootstrap.css" rel="stylesheet" type="text/css">
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
							<a class="nav home active" href="/material/" rel="nofollow" id="material">
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
									<a class="nav flat-box" href="/video/" id="video">
										<i class='fas fa-film fa-fw'></i>&nbsp;视频小站
									</a>
								</li>
								<li>
									<a class="nav flat-box active" href="/material/" id="material">
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
									<a class="nav flat-box" href="/tags/" rel="nofollow" id="tags">
										<i class='fas fa-hashtag fa-fw'></i>&nbsp;标签
									</a>
								</li>
								<li>
									<a class="nav flat-box" href="/blog/archives/" rel="nofollow" id="blogarchives">
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
		<div class="container-x" id="box">
			<div class="container">
				<div class="row">
					<div class="col-md-offset-3 col-md-6">
					    <h2 class="title-list heading-list">Foundations for Machine Learning</h2>
						<div class="panel-group" id="accordion" role="tablist" aria-multiselectable="true">
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingOne">
									<h4 class="panel-title">
										<a class= "collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
											1. Introduction
										</a>
									</h4>
								</div>
								<div id="collapseOne" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingOne">
									<div class="panel-body">
									    <a href="">1.1 What is machine learning?</a>
     								</div>
									<div class="panel-body">
									    <a href="">1.2 What kind of problems can be tackled using machine learning?</a>
     								</div>
									<div class="panel-body">
									    <a href="">1.3 Some standard learning tasks</a>
     								</div>
									<div class="panel-body">
									    <a href="">1.4 Learning stages</a>
     								</div>
									<div class="panel-body">
									    <a href="">1.5 Learning scenarios</a>
     								</div>
									<div class="panel-body">
									    <a href="">1.6 Generalization</a>
     								</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingTwo">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
											2. The PAC Learning Framework
										</a>
									</h4>
								</div>
								<div id="collapseTwo" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingTwo">
									<div class="panel-body">
										<a href="">2.1 The PCA learning model</a>
									</div>
									<div class="panel-body">
										<a href="">2.2 Guarantees for finite hyperthesissets — consistent case</a>
									</div>
									<div class="panel-body">
										<a href="">2.3 Guarantees for finite hypothesis sets — inconsistent case</a>
									</div>
									<div class="panel-body">
										<a href="">2.4 Generalities</a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingThree">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseThree" aria-expanded="false" aria-controls="collapseThree">
											3. Rademacher Complexity and VC-Dimension
										</a>
									</h4>
								</div>
								<div id="collapseThree" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingThree">
									<div class="panel-body">
										<a href="">3.1 Rademacher complexity</a>
									</div>
									<div class="panel-body">
										<a href="">3.2 Growth function</a>
									</div>
									<div class="panel-body">
										<a href="">3.3 VC-dimension</a>
									</div>
									<div class="panel-body">
										<a href="">3.4 Lower bounds</a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingFour">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseFour" aria-expanded="false" aria-controls="collapseFour">
											4. Model Selection
										</a>
									</h4>
								</div>
								<div id="collapseFour" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingFour">
									<div class="panel-body">
										<a href="">4.1 Estimation and approximation errors</a>
									</div>
									<div class="panel-body">
										<a href="">4.2 Empirical risk minimization (ERM)</a>
									</div>
									<div class="panel-body">
										<a href="">4.3 Structural risk minimization (SRM)</a>
									</div>
									<div class="panel-body">
										<a href="">4.4 Cross-validation</a>
									</div>
									<div class="panel-body">
										<a href="">4.5 n-Fold cross-validation</a>
									</div>
									<div class="panel-body">
										<a href="">4.6 Regularization-based algorithms</a>
									</div>
									<div class="panel-body">
										<a href="">4.7 Convex surrogate losses</a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingFive">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseFive" aria-expanded="false" aria-controls="collapseFive">
											5. Support Vector Machines 
										</a>
									</h4>
								</div>
								<div id="collapseFive" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingFive">
									<div class="panel-body">
										<a href="">5.1 Linear classification</a>
									</div>
									<div class="panel-body">
										<a href="">5.2 Separable case</a>
									</div>
									<div class="panel-body">
										<a href="">5.3 Non-separable case</a>
									</div>
									<div class="panel-body">
										<a href="">5.4 Margin theory</a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingSix">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseSix" aria-expanded="false" aria-controls="collapseSix">
											6. Kernel Methods 
										</a>
									</h4>
								</div>
								<div id="collapseSix" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingSix">
									<div class="panel-body">
										<a href="">6.1 Introduction</a>
									</div>
									<div class="panel-body">
										<a href="">6.2 Positive definite symmetric kernels</a>
									</div>
									<div class="panel-body">
										<a href="">6.3 Kernel-based algorithms</a>
									</div>
									<div class="panel-body">
										<a href="">6.4 Negative definite symmetric kernels</a>
									</div>
									<div class="panel-body">
										<a href="">6.5 Sequence kernels</a>
									</div>
									<div class="panel-body">
										<a href="">6.6 Approximate kernel feature maps</a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingSeven">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseSeven" aria-expanded="false" aria-controls="collapseSeven">
											7. Boosting 
										</a>
									</h4>
								</div>
								<div id="collapseSeven" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingSeven">
									<div class="panel-body">
										<a href="">7.1 Introduction</a>
									</div>
									<div class="panel-body">
										<a href="">7.2 AdaBoost</a>
									</div>
									<div class="panel-body">
										<a href="">7.3 Theoretical results</a>
									</div>
									<div class="panel-body">
										<a href="">7.4 L1-regularization</a>
									</div>
									<div class="panel-body">
										<a href="">7.5 Discussion</a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingEight">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseEight" aria-expanded="false" aria-controls="collapseEight">
											8. On-Line Learning
										</a>
									</h4>
								</div>
								<div id="collapseEight" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingEight">
									<div class="panel-body">
										<a href="">8.1 Introduction</a>
									</div>
									<div class="panel-body">
										<a href="">8.2 Prediction with expert advice </a>
									</div>
									<div class="panel-body">
										<a href="">8.3 Linear classification </a>
									</div>
									<div class="panel-body">
										<a href="">8.4 On-line to batch conversion </a>
									</div>
									<div class="panel-body">
										<a href="">8.5 Game-theoretic connection</a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingNine">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseNine" aria-expanded="false" aria-controls="collapseNine">
											9. Multi-Class Classification 
										</a>
									</h4>
								</div>
								<div id="collapseNine" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingNine">
									<div class="panel-body">
										<a href="">9.1 Multi-class classification problem </a>
									</div>
									<div class="panel-body">
										<a href="">9.2 Generalization bounds </a>
									</div>
									<div class="panel-body">
										<a href="">9.3 Uncombined multi-class algorithms </a>
									</div>
									<div class="panel-body">
										<a href="">9.4 Aggregated multi-class algorithms </a>
									</div>
									<div class="panel-body">
										<a href="">9.5 Structured prediction algorithms </a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingTen">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseTen" aria-expanded="false" aria-controls="collapseTen">
											10. Ranking 
										</a>
									</h4>
								</div>
								<div id="collapseTen" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingTen">
									<div class="panel-body">
										<a href="">10.1 The problem of ranking</a>
									</div>
									<div class="panel-body">
										<a href="">10.2 Generalization bound</a>
									</div>
									<div class="panel-body">
										<a href="">10.3 Ranking with SVMs</a>
									</div>
									<div class="panel-body">
										<a href="">10.4 RankBoost</a>
									</div>
									<div class="panel-body">
										<a href="">10.5 Bipartite ranking </a>
									</div>
									<div class="panel-body">
										<a href="">10.6 Preference-based setting  </a>
									</div>
									<div class="panel-body">
										<a href="">10.7 Other ranking criteria  </a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingEleven">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseEleven" aria-expanded="false" aria-controls="collapseEleven">
											11. Regression 
										</a>
									</h4>
								</div>
								<div id="collapseEleven" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingEleven">
									<div class="panel-body">
										<a href="">11.1 The problem of regression </a>
									</div>
									<div class="panel-body">
										<a href="">11.2 Generalization bounds </a>
									</div>
									<div class="panel-body">
										<a href="">11.3 Regression algorithms</a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingTwelve">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseTwelve" aria-expanded="false" aria-controls="collapseTwelve">
											12. Maximum Entropy Models 
										</a>
									</h4>
								</div>
								<div id="collapseTwelve" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingTwelve">
									<div class="panel-body">
										<a href="">12.1 Density estimation problem </a>
									</div>
									<div class="panel-body">
										<a href="">12.2 Density estimation problem augmented with features</a>
									</div>
									<div class="panel-body">
										<a href="">12.3 Maxent principle </a>
									</div>
									<div class="panel-body">
										<a href="">12.4 Maxent models </a>
									</div>
									<div class="panel-body">
										<a href="">12.5 Dual problem </a>
									</div>
									<div class="panel-body">
										<a href="">12.6 Generalization bound </a>
									</div>
									<div class="panel-body">
										<a href="">12.7 Coordinate descent algorithm</a>
									</div>
									<div class="panel-body">
										<a href="">12.8 Extensions </a>
									</div>
									<div class="panel-body">
										<a href="">12.9 L2-regularization</a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingThirteen">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseThirteen" aria-expanded="false" aria-controls="collapseThirteen">
											13. Conditional Maximum Entropy Models 
										</a>
									</h4>
								</div>
								<div id="collapseThirteen" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingThirteen">
									<div class="panel-body">
										<a href="">13.1 Learning problem</a>
									</div>
									<div class="panel-body">
										<a href="">13.2 Conditional Maxent principle</a>
									</div>
									<div class="panel-body">
										<a href="">13.3 Conditional Maxent models</a>
									</div>
									<div class="panel-body">
										<a href="">13.4 Dual problem</a>
									</div>
									<div class="panel-body">
										<a href="">13.5 Properties</a>
									</div>
									<div class="panel-body">
										<a href="">13.6 Generalization bound</a>
									</div>
									<div class="panel-body">
										<a href="">13.7 Logistic regression</a>
									</div>
									<div class="panel-body">
										<a href="">13.8 L2-regularization</a>
									</div>
									<div class="panel-body">
										<a href="">13.9 Proof of the duality theorem</a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingFourteen">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseFourteen" aria-expanded="false" aria-controls="collapseFourteen">
											14. Algorithmic Stability  
										</a>
									</h4>
								</div>
								<div id="collapseFourteen" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingFourteen">
									<div class="panel-body">
										<a href="">14.1 Definitions </a>
									</div>
									<div class="panel-body">
										<a href="">14.2 Stability-based generalization guarantee</a>
									</div>
									<div class="panel-body">
										<a href="">14.3 Stability of kernel-based regularization algorithms</a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingFifteen">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseFifteen" aria-expanded="false" aria-controls="collapseFifteen">
											15. Dimensionality Reduction 
										</a>
									</h4>
								</div>
								<div id="collapseFifteen" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingFifteen">
									<div class="panel-body">
										<a href="">15.1 Principal component analysis </a>
									</div>
									<div class="panel-body">
										<a href="">15.2 Kernel principal component analysis (KPCA)</a>
									</div>
									<div class="panel-body">
										<a href="">15.3 KPCA and manifold learning</a>
									</div>
									<div class="panel-body">
										<a href="">15.4 Johnson-Lindenstrauss lemma</a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingSixteen">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseSixteen" aria-expanded="false" aria-controls="collapseSixteen">
											16. Learning Automata and Languages
										</a>
									</h4>
								</div>
								<div id="collapseSixteen" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingSixteen">
									<div class="panel-body">
										<a href="">16.1 Introduction</a>
									</div>
									<div class="panel-body">
										<a href="">16.2 Finite automata </a>
									</div>
									<div class="panel-body">
										<a href="">16.3 Efficient exact learning </a>
									</div>
									<div class="panel-body">
										<a href="">16.4 Identification in the limit</a>
									</div>
								</div>
							</div>
							<div class="panel panel-default">
								<div class="panel-heading" role="tab" id="headingSeventeen">
									<h4 class="panel-title">
										<a class="collapsed" role="button" data-toggle="collapse" data-parent="#accordion" href="#collapseSeventeen" aria-expanded="false" aria-controls="collapseSeventeen">
											17. Reinforcement Learning 
										</a>
									</h4>
								</div>
								<div id="collapseSeventeen" class="panel-collapse collapse" role="tabpanel" aria-labelledby="headingSeventeen">
									<div class="panel-body">
										<a href="">17.1 Learning scenario </a>
									</div>
									<div class="panel-body">
										<a href="">17.2 Markov decision process model</a>
									</div>
									<div class="panel-body">
										<a href="">17.3 Policy</a>
									</div>
									<div class="panel-body">
										<a href="">17.4 Planning algorithms</a>
									</div>
									<div class="panel-body">
										<a href="">17.5 Learning algorithms</a>
									</div>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div> <!-- container-x -->
		<!--footer-->
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
		<!-- 点击特效，输入特效 运行时间 -->
		<script type="text/javascript" src="/cool/cooltext.js"></script>
		<script type="text/javascript" src="/cool/clicklove.js"></script>
		<script type="text/javascript" src="/cool/sitetime.js"></script>
		<!--js-->
		<script src="https://cdn.bootcss.com/jquery/1.11.0/jquery.min.js"></script>
		<script src="https://cdn.jsdelivr.net/npm/jquery@3.3.1/dist/jquery.min.js"></script>
		<script async src="https://cdn.jsdelivr.net/gh/xaoxuu/cdn-busuanzi@2.3/js/busuanzi.pure.mini.js"></script>
		<script src="https://cdn.jsdelivr.net/gh/xaoxuu/cdn-material-x@19.5/js/app.js"></script>
        <script src="https://cdn.jsdelivr.net/gh/xaoxuu/cdn-material-x@19.5/js/search.js"></script>
		<script src="https://cdn.bootcss.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
	</body>
</html>



{% endraw %}