---
layout: post
comments: true
title: 大规模社交用户的标签自动生成技术研究
category: 技术
tags: MachineLearning
keywords: 推荐系统,图嵌入式表示,大数据
description: 
---



## 摘要
   社会标签(Social Tagging)作为一种新型网络信息组织方式,由网络信息的提供者或者用户自发为某类信息赋予一定数量的标签,
选用自由词对感兴趣的网络信息资源进行描述来实现网络信息的分类,正在改变着传统的网络信息组织模式,
被广泛应用在社交网络、商业以及科研教育领域。社会标签较传统的主题词具有更大的灵活性、易用性,同时资源描述性关键词的增加也便于
对资源进行准确查找,尤其是对多媒体资源来说,用户所标注的标签信息就更为重要。
  
   Word2Vec从提出至今，已经广泛应用在自然语言处理、机器学习、数据挖掘等领域，从它延伸出的Doc2Vec、Graph2Vec也具有广泛的应用前景，
它已经成为了深度学习在自然语言处理中的基础部件，本文尝试利用大规模微博社交网络数据，提出将异构网络（微博用户关系网络、用户标签网络、用户文本网络）
进行联合训练的方法并得到用户与标签的嵌入式表示。这项技术可用于建模用户兴趣，推荐用户感兴趣的商品、信息和好友，进行用户画像，以及寻找目标用户进行商品推广。
本文提出的联合训练方法产生的用户与标签的嵌入式表示优于分别训练用户关系网络、用户标签网络产生的用户和标签的嵌入式表示，并在预测用户标签的准确率上优于同类方法。

关键字：推荐系统，图嵌入表示，大数据




## Abstract

   Social Tagging is a new type of organizing the information of Internet. The users of Internet will provide some tags to
some objects spontaneously.We can use these tags to sort resources that we are interested in. Social Tagging is changing
the the method of the organization of network information resources and widely used in the social network service,business
and education. Social tagging is more flexible and convenient than traditional key words. With the increase of the number of 
social tags，it is more convenient to locate resource. Especially for multi-media resource，social tagging plays an increasingly important role.
   
   Word2vec has been widely applied in the fields of NLP，machine learning and data mining since it was first proposed by Mikolov. Doc2vec
and Graph2Vec，which are all originated from Word2vec，also have wide application foreground. Word2vec has become fundamental component in 
the field of NLP. This paper tries to use huge amounts of social networks data. In this paper, a new algorithm is presented, which can be
use to train heterogeneous network(user relationship network and user tags network and user text network). Through this algorithm,we can 
get embedding of both user and tag and then we can use these embedding in the fields，such as user interests modeling，Commodity recommendation and User portrait. The embedding produced by our algorithm is better than that produced by separately training each network.
Besides,in the case of predicting user tags，our algorithm is better than other similar methods.

keywords:Recommendation System,Graph Embedding,Big Data



##  1. 引言
### 1.1 研究背景 

  微博作为一种新型媒体，是一种基于草根用户的关系构建个性化用户信息的即时传递、共享和获取平台。它具有信息实时性，内容简洁性，用户交互性等特点。
微博之所以可以成为当今国内外主流的社交媒体，主要是因为其具有强大的用户实时交互性，用户在使用微博的过程中，会在微博的网络空间中结成种种关系，
比如用户之间的关注关系，社区中的好友和亲情关系，实时交互过程中因共同购买或评论产品而结成的共同评论关系等，有效分析和挖掘微博中复杂的用户关系不仅可以激发、
助推和引导社会事件的发展趋势，还可以准确高效的为关注某一兴趣和爱好的微博群体进行个性化推荐，甚至可以大大降低企业和消费者的交易成本，推动企业营销模式的不断创新。
此外，微博在凝聚民心，降低事件危害以及政务互动等方面也发挥着不可替代的积极作用。由此可见，微博的兴起赋予了社会经济活动前所未有的大众化和网络化的内涵，
极大提升了社交媒体的社会服务效能。但是急剧增长的微博用户数量和海量用户下的交互行为增加了社会、经济与生产的复杂性，使一些社会实践变得更加不可预测，难以控制，
从而为分析社会化效应带来了新的挑战，因此如何正确理解微博用户之间的关系以及用户在关系交互中所产生的行为，成为学者迫切需要研究的新方向。

 社会化标签（Social Tagging） 也称为collaborative tagging，指的是用户在网络中自发得分配电子标签或关键词来描述网络上的资源，并且在网络用户群体中共享这些标签。这种方法允许用户使用自己的语言、以“标签”的形式对信息资源的内外部特征进行标注,以实现资源的查找和共享。社会化标签系统与预先定义网络资源类别
 来对网络资源分类的机制相比，它可以使用户自发产生和分配标签，这对网络资源分类有积极意义。随着社会化标签的广泛应用，公众分类法（folksonomy）应运生，这种分类方法改变了传统利用专家来对网络资源分类的方式，它是从公众的角度来分类资源。公众分类法是互联网所推崇的共享与协作精神的体现,是新的互联网信息环境中一种独具特色的信息组织工具。它的产生为互联网信息组织与检索的改进提供了新方向。随着社交网络，图片分享，视频分享业务的蓬勃发展，社会化标签系统被广泛应用在互联网的各个领域，它可以对社交网络和数据挖掘提供技术支持，而且有助于改进搜索结果，提高广告投放的准确率，同时利用所有用户产生的标签数据可以挖掘出其他有价值的信息，比如用户社群，潜在目标客户等。社会化标签系统将用户产生的大量通过网络传播而聚合起来，可以实现对网络资源的合作标记和公众分类，体现网络用户的群体智慧。

   在web2.0中，用户不仅可以通过豆瓣来分享图书,通过优酷来分享视频，通过微博来发表博文，通过Flicker来发布照片，通过youtube上传视频等方式来创造内容，
用户这些行为有一个共同的特征，即用户会自由的选择一些词(Term)或者短语(Phrase)来标注相关网络信息资源，
我们称用户的这种行为为标注(Tagging),用户所选择的词或者词语为标签(Tag),提供标注行为的系统为社会标签系统，
本文研究的就是利用微博用户关系与用户已有标签来为每个用户推荐相关的标签，便于建模用户兴趣，为用户之间的衔接赋予更丰富的信息，
推荐用户感兴趣的商品、信息和好友，进行用户画像，以及寻找目标用户进行商品推广。

### 1.2 问题的提出
在Web2.0环境下，互联网已经成为全球最大的知识库，它在给人类的生活和工作带来革命性变化的同时，也引发了“信息泛滥”，“信息迷航”等问题，社会化标签推荐能够根据用户的需求主动的将合适的信息、商品、知识提供给用户，可以有效缓解这些问题。同时，作为由用户产生的元数据，社会化标签能够独特反应用户的需求及其变化{% cite mathes2004folksonomies%}，而且“用户-资源-社会化标签”之间的关系网络能够为个性化信息推荐系统提供十分有价值的基础数据，由此部分学者对基于社会化标签的个性化知识推荐进行了密切的关注，并从以下三个方面进行了探索。

* 基于矩阵的方法，即通过构建“用户-资源”矩阵、“用户-社会化标签”矩阵，“社会化标签-资源”矩阵实现知识推荐。Ji AT等人依据“用户-资源”矩阵，“用户-标签”矩阵，“标签-资源”矩阵构建Naive Bayesian分类器，以此为基础实现协同过滤推荐{% 
cite JiAT2007Collaborative%}

* 基于聚类的方法,主要包括用户、资源、社会化标签三种对象的聚类，其中基于社会化标签的聚类是当前研究的重点。一个重要的研究思路就是一句社会化标签之前的共现频率，利用k-means聚类算法、马尔科夫聚类算法等方法对社会化标签进行聚类，进而依据聚类的结果为用户提供个性化推荐。

* 基于图论的方法。其中社会网络分析是学者们关注的重点，如Shiratsuchi等人依据用户使用的标签之间的相似性
  建立用户社会网络{% cite shiratsuchi2006finding%}，并利用Clauset提出的local midularity算法{% cite clauset2005finding%}划分网络社区，进而实现协同过滤推荐。

显然，学者们提出的三类方法都有其优势，但是都面临各自的问题。首先第一类方法，面临一个重要的问题就是社会化标签使用量的“幂率分布”规律，排序在前几位的社会化标签具有较大的使用量，而大量的社会化标签都处于“长尾”区域，由此相关的矩阵可能非常不规则，从而严重制约个性化推荐算法。其次第二类方法中基于社会化标签的聚类方法属于基于内容的推荐思想，难以发现用户新的兴趣，而且社会化标签使用量的“幂律分布”问题同样会制约推荐效果。最后，第三类方法中基于社会网络分析的方法属于协同过滤思想，虽然能够发现用户新兴趣，但是个性化推荐需要依据用户的特定知识需求，在一般的社会网络中，个性间的“关系互动”并不意味着在特定的知识情境下必然能产生“知识互动”由此，当前学者们提出的基于社会化标签的各种推荐方法都有自身无法克服的劣势，如何利用社会化标签是实现精准的推荐是学者研究的重要问题，本文中提出一种有监督训练社会化标签的方法，我们对海量微博数据构建用户关系网络，用户转发网络，用户文本网络，用户标签网络，其中用户转发网络与用户关系网络刻画的是用户之间的交互关系，我们通过一阶相似性和二阶相似性找到相似用户{% cite tang2015line%}，在此基础上利用用户标签网络和用户文本网络为用户注入标签信息，利用这种方式我们可以得到用户与标签的嵌入式表示，在预测用户标签任务中，效果优于同类方法。
 

### 1.3 研究内容与研究方法

#### 1.3.1 研究内容

传统的无监督文本嵌入方法，比如Skip-gram，Paragraph Vectors可以学习到比较通用微博用户向量表示，但是这种方法对指定的任务不会产生很高的准确率，我们提出一种有监督的学习方法，它可以利用打标签的数据(用户-标签网络)和无标签数据(用户-用户网络)来产生对指定分类任务更有针对性的标签，从而提升用户分类的准确率。为了达到这个目标，我们需要对有标记数据和无标记数据进行统一表示。
下面来定义五个网络:

* User-User Network:用户关注关系网络,记为\\(G\_{uu}=(V,E\_{uu})\\)这个网络记录了用户之间的关注关系，\\(V\\)是爬取的微博用户的集合，\\(E_{uu}\\)是边的集合，表现的是用户之间的关注关系，如果两个用户之间有关注关系，这两个用户之间有一条边。用户-用户网络捕获了用户之间的关注关系，这个是用户嵌入表示，比如Skip-Gram，所需要的重要信息。

* User-Forward Network：用户转发网络,记为\\(G\_{uf} = (V\_1,E\_{uf})\\),这个网络记录了微博用户的转发关系，其中的转发特征是根据微博文本中的`//@`来确定的，
\\(V\_1\\)是爬取数据微博文本中含有转发关系的用户集合，\\(V\_1\\)是\\(V\\)的子集，\\(E\_{uf}\\)是用户与转发用户之间边的集合,这个网络隐含用户之间的关系，为了有监督的捕获用户标签的信息，我们需要定义用户标签网络,用户微博文本网络。

* User-Tag Network：用户标签网络，记为\\(G\_{ut}=(V\_2\bigcup T,E\_{ut})\\)。这个二部图网络记录了用户与用户的Tag之间的信息，包含的标记信息，用于作为标记数据。\\(V\_2\\)是爬取的微博数据中带有标签的微博用户的集合，\\(V\_2\\)是\\(V\\)的子集。\\(T\\)是爬取微博用户的标签的集合，\\(E\_{ut}\\)是用户集合和标签集合之间边的集合。

* User-Microblog Network: 用户微博文本网络,记为\\(G\_{um}=(V\_3\bigcup T\_1,E\_{um})\\)这个网络记录了微博文本中的标签信息,我们通过对微博文本分词并提取标签信息得到，其中\\(V\_3\\)是通过分析用户发的微博得到用户集合，\\(V\_3\\)是\\(V\\)的子集，
其中\\(T\_1\\)是通过分析用户发的微博得到标签集合，它是\\(T\\)的子集,这个网络具有标记信息，可用于有监督学习。

* Heterogeneous User Tag Network：这个网络整合了无标记网络(User-User Network、User-Forward Network)和有标记网络(User-Tag Network、User-Microblog Network),它包含不同维度的用户信息，包含有标记数据和无标记数据两部分。

下面来引入我们的研究内容：  
Predictive User Tag embedding:针对海量微博数据(无标记的用户关系数据和有标记用户标签数据),我们通过Heterogeneous User Tag Network去得到用户以及Tag的低维表示。通过用户和Tag的低维表示，我们可以利用\\( \vec{u}\bullet\vec{v}\\)来排序来计算与用户最相关的Tag。


#### 1.3.2 研究方法
下面介绍本文的训练方法：

(1)	Bipartite Network Embedding
LINE Model是Graph embedding的比较常用的方法，它可以训练大规模网络从而得到节点的嵌入式表示。LINE Model主要是解决同构网络的节点嵌入式表示，由于异构网络之间边的权重没有可比性，所以LINE Model不能直接应用于异构网路。我们利用了LINE Model 二阶相似的思想，有相似邻居的两个顶点是相似的，这两个顶点在低维空间中距离很近。给定一个二部图网络\\(G=(V\_A\bigcup{V\_B},E)\\)，其中\\(V_A\\)，\\(V\_B\\)是两个不同类型的不相交的顶点集合，E是连接两个顶点集合之间边的集合。我们首先定义由在集合\\(V\_B\\)中的顶点\\(v\_j\\)生成\\(V\_A\\)中\\(v\_i\\)顶点的条件概率为：

\begin{equation}
  p(v\_i|v\_j) = {e^{\vec{u\_i}\bullet\vec{u\_j}}\over \sum\_{i^{\prime} \in V\_A} e^{\vec{u\_i^{\prime}}\bullet \vec{u\_j}}} 
  \label{eq:eq1}
\end{equation}


其中\\(\vec u\_i\\)是\\(v\_i\\)的嵌入式表示，\\(\vec u\_j\\)是\\(v\_j\\)的嵌入式表示，对于每一个在\\(V\_B\\)中的顶点\\(v\_j\\),方程\eqref{eq:eq1}定义了在`\(V_A\)`中所有顶点上的一个条件分布`\(p(\bullet|v_j)\)`。对于任意一对顶点`\(v_j\)`和`\(v_{j^{\prime}}\)`,为了保留二阶相似性，我们可以使条件分布`\(p(\bullet|v_j)\)`接近于
`\(\hat p(\bullet|v_j)\)`,所以我们可以通过最小化下面这个目标函数达到目标:
\begin{equation}
O = \sum\_{j \in V\_B} \lambda\_j d(\hat p(\bullet|v\_j),p(\bullet|v\_j))
\label{eq:eq2}
\end{equation}

其中`\(d(\bullet,\bullet)\)`是两个分布的KL距离,`\(\lambda_j\)`用来表示顶点`\(v_j\)`在网络中的重要性，它可以定义为:`\(deg_j=\sum_i w_{ij}\)`,经验分布
`\(\hat p(\bullet|v_j) = {w_{ij}\over deg_j}\)`忽略一些常数，目标函数\eqref{eq:eq2}可简化为：
\begin{equation}
O = - \sum\_{(i,j) \in E} w\_{ij} \log {p(v\_j|v\_i)}
\label{eq:eq3}
\end{equation}

目标函数\eqref{eq:eq3}可以利用边采样[5]或者负采样[11]的随机梯度下降法进行优化求解。
我们可以将上文提到的四种单一网络中的无向边看成是两条有向边，然后`\(V_A\)`可以看做源节点的集合，`\(V_B\)`可以看做目的节点的集合，通过这样处理，我们可以将上文提到的四种单一网络视为二部图来处理，从而可以利用改模型进行求解

(2)	Heterogeneous User Tag Network由两个四个部图网络构成，其中无标记网络为User-User Network、User-Forward Network和有标记网络为User-Tag Network、User-Microblog Network。其中User节点集合被四个网络所共享，为了学习到四个网络结构的嵌入式表示，我们直觉上的方法是整体训练这四个二部图网络，即最小化下面的目标函数:
\begin{equation}
O\_{total} = O\_{uu} + O\_{ut} + O\_{uf} + O\_ {um}
\label{eq:eq4}
\end{equation}

其中：
\begin{equation}
O\_{uu} = - \sum\_{(i,j) \in E\_{uu}} \log p(v\_i|v\_j)
\label{eq:eq5}
\end{equation}

\begin{equation}
O\_{ut} = - \sum\_{(i,j) \in E\_{ut}} \log p(v\_i|t\_j)
\label{eq:eq6}
\end{equation}

\begin{equation}
O\_{uf} = - \sum\_{(i,j) \in E\_{uf}} \log p(v\_i|f\_j)
\label{eq:eq7}
\end{equation}

\begin{equation}
O\_{um} = - \sum\_{(i,j) \in E\_{um}} \log p(v\_i|m\_j)
\label{eq:eq8}
\end{equation}


目标函数\eqref{eq:eq4}有多种优化方法，一种解决方式是同时训练有标记数据和无标记数据，我们称这种方式为联合训练(Joint training)；另一种方式是先训练无标记数据，得到用户的嵌入式表示，然后利用有标记数据进行调优(Pre-training + Fine-tuning)[12],下面是具体的训练过程：

![algorithm](/public/img/pte-algorithm-1-2.png "pte-algorithm")

在联合训练中，上面的四种网络（用户关注关系网络、用户转发网络、用户标签网络、用户微博文本网络）均被训练，优化\eqref{eq:eq4}的一个方案是将\\(G\_{uu}，G\_{uf}，G\_{ut}，G\_{um}\\)中的所有边聚集在一起，然后使用边采样来更新模型，边采样的概率正比于边的权重，然而当网络是异构的，不同的网络结构之间边的权重是不兼容的，一个更好的解决方案是从四个边的集合中交替选择边采样，如上图中的算法\\(1\\)所示，相似的先训练后优化的算法细节如上图中的算法\\(2\\)所示。

## 2. 国内外研究现状
本文对微博平台的使用主体也就是微博用户间的用户关系，用户标签以及用户发微博情况，以及微博用户转发情况展开相关研究，针对微博用户方面的内容也有很多人进行了研究，主要从以下几个角度进行研究:

标签是由用户认为自由，不受约束环境下创造出来的，因此具有自由性和低限度的特点，当然标签系统的优点也往往正是它的缺点，标签具有一定的社会性和含糊性，也同时存在着例如同义词，多义词等一词多义，甚至拼写错误的情况，所以导致了标签系统存在大量重复、不规范、无效的标签，我们称之为噪音。当用户对其感兴趣的资源进行标注标签行为的时候，规范、有效、质量高的标签则会创造出标签系统的循环性，促进系统良性循环。

面向微博用户关系模式信息推荐的基本思想是首先建立用户和信息源之间以及用户和用户之间的对应关系，然后进行用户社群分析，建立相似用户群或兴趣共同体。当相同用户群的某个用户或者某几个用户对某信息或者商品感兴趣时，可以预测共同体的其他成员也感兴趣，从而将该信息推荐给其他成员。常用的分析方法时，通过分析用户社交网络中用户之间的相互关系情况，然后根据用户的不同关系进行基于内容的协同过滤推荐。在微博使用实践中，用户积极选择并参与构建个性化关系，与一些具有相似特征的用户自发的聚集到一起形成群体，用户社群分析作为用户关系挖掘的主要技术手段，他在常规复杂系统的研究中比较成熟。


社会化标签系统是Web用户利用社会化标签对Web资源进行标注的环境，它包括三个基本的实体，分别是Web用户，Web资源和社会化标签。另外，还包括一个关系集合。社会化标签系统的模型可以用一个四元组来表示\\(F=(U,T,R,A)\\),其中，\\(U\\)是Web用户的有限集合，\\(T\\)是社会标签的有限集合，\\(R\\)是Web资源的有限集合，\\(A \subseteq U \times T \times R\\)是一个三元关系集合，元素\\(a=(u,t,r) \in A\\)表示用户\\(a\\)使用标签\\(t\\)标注了资源\\(r\\)。标签共现分析是揭示标签语义关系的重要途径，Michlmayr和Cayzer {% cite michlmayr2007learning %}指出，如果两个标签被某一用户结合或共同使用去标注某一个书签，那么这两个标签之间一定存在着某种语义关系。Szomszor等人{% cite ecs14007 %}通过实验表明标签共现
关系的重要本质是能够用来揭示标签之间的语义关系,并利用Jaccard系数来衡量标签之间的共现关系。Kipp和Campbell {% cite kipp2006patterns %}利用共词分析来抽取社会化书签服务系统delicious中的标签模型，他们发现标签的数量和使用频率之间遵循幂律分布，即只有少量的标签被经常用来标注资源而大部分标签被使用的次数较少。Begelman，Keller和Smadja等人{% cite begelman2006automated %}使用标签聚类技术，提出了基于标签共现分布相似性的算法，并利用谱聚类实现了标签的聚类分析。王萍和张际平{% cite 王萍2010一种社会性标签聚类算法 %}把标签共现定义为两个标签用来标注同一资源,并设计了一种基于标签相似性的聚类算法对标签共现网络进行分割,来建立标签聚类簇。

随着互联网技术的发展，用户的日常生活和互联网建立起了紧密的联系，与此同时，互联网上产生了海量用户数据，海量数据为个性化推荐系统创造了独一无二的优势，近几年，个性化推荐技术逐渐成为众多研究者的研究热点{% cite Xu:2006:CAS:2114193.2114262%}{% cite yin2013connecting%}。文献{% cite 基于社会化标签的协同过滤推荐策略研究 %}和文献{% cite lacic2014recommending%}均介绍了各种类型的成熟推荐技术，这些推荐技术各有利弊，分别适用于不同类型不同场景下的推荐系统。如Alexandrin Popescul等提出概率框架，合并基于内容和基于协同过滤的方法，加上EM算法学习的二次内容信息用于解决稀疏问题，辅助混合模型推荐。

微博是Web2.0的重要应用，其中包含了丰富的网络和用户信息，在微博中标签是一种表示用户兴趣和属性的有效方式，一个用户的兴趣也通常隐藏在他/她的文本和网络中，Zhiyuan Liu提出一种概率模型，网络正则化的概率标签模型NTDM{% cite 涂存超:24%},用来进行微博用户标签推荐，NTDM用来对微博个人介绍中的词和语义关系进行建模，同时将其所在的网络结构信息通过正则化的方式考虑进来，产生了很好的效果。
社会化标签技术对个性化推荐的精确、高效起到了推进作用{% cite kumar2014exploiting%}，Chatti等在个人学习环境(PLE)设置不同的标签来研究基于标签的协同过滤算法，Godoy等实现以标签为基础的分类结果证明基于标签的分类优于那些使用文本的文档以及其他内容相关的来源的推荐效果{% cite godoy2012one%}；Yoshida等使用通过结合标签的排名和基于内容的过滤得出标签的相关性水平排名，从而提高项目推荐性能{% cite yoshida2012improving%}。

关于微微博用户具有结构差异特性，不同用户因其自身属性及其所在关系位置的不同，所以其在关系中所处的位置和交互方式也各不相同，并逐步形成一定的影响力，用户影响力分析主要研究如何基于用户的交互活动水平来研究用户与用户是如何相互影响以及研究用户在社交网络中影响力的大小。在社区中影响力大的用户是关键用户（或称意见领袖），能在一定程度上引导舆论，影响用户行为和政治观点等。




## 3. 介绍模型和大数据处理的相关技术

### 3.1 分布式向量表示

#### 3.1.1 统计语言模型
word2vec 是 Google 于 2013 年开源推出的一个用于获取 word vector 的工具包{% cite mikolov2013distributed%}，它简单、高效，因此引起了很多人的关注，word2vec是用来生成词向量的工具，而词向量和语言模型有着密切的联系。当今的互联网迅猛发展，每天都在产生大量的文本，图片，语言和视频数据，要从这些数据处理并挖掘出有价值的信息，离不开自然语言处理（Nature Language Processing，NLP）技术，其中统计语言模型（Statistical Language Model）就是很重要的一环，它是所有NLP的基础，被广泛应用于语音识别，机器翻译，分词，词性标注和信息检索等任务。统计语言模型是用来计算一个句子的概率的概率模型，他通常基于一个语料库来构建。假设\\(W = w\_1^T :=(w\_1,w\_2,\cdots,w\_T)\\)表示由T个词\\(w\_1,w\_2,\cdots,w\_T\\)按顺序构成一个句子，则\\(w\_1,w\_2,\cdots,w\_T\\)的联合概率

\begin{equation}
p(W) = p(w\_1^T) = p(w\_1,w\_2,\cdots,w\_T)
\label{eq:eq9}
\end{equation}
就是这个句子的概率，利用Bayes公式，上式可以被链式分解为

\begin{equation}
p(w\_1^T) = p(w\_1)\bullet p(w\_2|w\_1) \bullet p(w\_3|w\_1^2) \cdots p(w\_T|w\_1^{T-1})
\label{eq:eq10}
\end{equation}
其中的条件概率\\(p(w\_1)\bullet p(w\_2|w\_1) \bullet p(w\_3|w\_1^2) \cdots p(w\_T|w\_1^{T-1})\\)就是语言模型的参数，那么给定一个句子\\(w\_1^T\\)就可以很快地算出相应的\\(p(w\_1^T)\\)了。
刚才我们考虑了一个给定长度为\\(T\\)的句子，就需要计算\\(T\\)个参数，假设对应词典\\(D\\)的大小(即词汇量)为\\(N\\),那么如果考虑长度为T的任意句子，总过就需要\\(T \bullet N^T\\)个参数，这些参数的量级是很大的。

#### 3.1.2 n-gram模型
n-gram模型可以用来计算上述的参数，考虑\\(p(w\_k|w\_1^{k-1})(k > 1)\\)的近似计算。利用Bayes公式有

\begin{equation}
p(w\_k|w\_1^{k-1}) = {p(w\_1^k) \over p(w\_1^{k-1})}
\label{eq:eq11}
\end{equation}
根据大数定理，当语料库足够大时，
\begin{equation}
p(w\_k|w\_1^{k-1}) \approx {count(w\_1^k) \over count(w\_1^{k-1})}
\label{eq:eq12}
\end{equation}
其中\\(count(w\_1^k)\\)和\\(count(w\_1^{k-1})\\)在表示词串\\(w\_1^k\\)和\\(w\_1^{k-1}\\)语料中出现的次数,当k很大时，统计会很耗时。从公式\eqref{eq:eq10}可以看出，一个词出现的概率与它前面的所有词都相关。n-gram模型的基本思想是一个词出现的概率和它前面固定数目的词相关，它做了一个\\(n-1\\)阶的Markov假设，认为一个词出现的概率只与它前面的\\(n-1\\)个词相关，即
$$p(w\_k|w\_1^{k-1}) \approx p(w\_k|w\_{k-n+1}^{k-1})$$
于是公式\eqref{eq:eq12}可以简化为
\begin{equation}
p(w\_k|w\_1^{k-1}) \approx {count(w\_{k-n+1}^k) \over count(w\_{k-n+1}^{k-1})}
\label{eq:eq13}
\end{equation}
这样简化，不仅使得参数统计变得更容易，也使得参数总数变得更少了。

#### 3.1.3 神经概率语言模型
本小节介绍Bengio等人提出的一种神经概率语言模型{% cite bengio2003neural%}，该模型用到了一个重要的工具---词向量。对词典\\(D\\)中的任意词\\(w\\)指定一个任意长度的实值向量\\(v(w) \in \Bbb R^m\\),\\(v(w)\\)就称为\\(w\\)的词向量，m为词向量的长度。
图1给出了神经网络的结构示意图，它包括四个层：输入层(Input)，投影层(Projection)，隐藏层(Hidden)和输出层(Output)，其中\\(W,U\\)分别为投影层与隐藏层以及隐藏层和输出层之间的权值矩阵，\\(p,q\\)分别为隐藏层和输出层上的偏置向量。
{% include  image.html
            img="/public/img/neuro-network.png"
            title="title for image"
            caption="图1 神经网络结构图" 
%}
对于语料\\(C\\)中的任意一个词\\(w\\)，将Context(w)取为前\\(n-1\\)个词(类似于n-gram)，这样二元对\\(Context(w),w\\)就是一个训练样本了，接下来将讨论
样本\\(Context(w),w\\)经过如图1所示的神经网络时是如何参与运算的。一旦语料\\(C\\)和词向量的长度\\(m\\)给定后，投影层和输出层的规模就确定了，前者为\\((n-1)m\\),后者为\\(N=\|D\|\\),即语料C的词汇量大小，而隐藏层的规模\\(n\_n\\)是可调参数，由用户指定。将输入层的\\(n-1\\)个词向量按顺序首尾相接地拼起来形成了一个长向量，其长度是\\((n-1)m\\),有了\\(x\_w\\)了接下来的计算过程就很平凡了，具体为




$$
\left\{  
\begin{array}  
{l l}  
z_w &=tanh(Wx_w + p)\\
y_w &=Uz_w+q 
\end{array}  
\right.
$$


其中tanh为双曲正切函数，用来做隐藏层的激活函数，上式中，tanh作用在向量上表示它作用在向量的每一分量上。经过上述两步计算得到的\\(y\_w =(y\_{w,1},y\_{w,2},\cdots,y\_{w,N})\\)只是一个长度为N的向量，其分量不能表示概率，如果想要\\(y\_w\\)的分量\\(y\_{w,i}\\)表示当上下文为\\(Context(w)\\)时下一个词恰为词典\\(D\\)中第i个词的概率，则还需要做一个softmax归一化，归一化后，\\(p(w\|Context(w))\\)就可以表示为

\begin{equation}
p(w|Context(w)) = {e^{y\_{w,i\_w}} \over \sum_{i=1}^N e^{y\_{w,i}}}
\label{eq:eq14}
\end{equation}
其中\\(i\_w\\)表示词\\(w\\)在词典\\(D\\)中的索引。
公式\eqref{eq:eq14}给出了概率\\(p(w\|Context(w))\\)的函数表示，即找到了上一节中提到的函数\\(F(w,Context(w),\theta)\\),其中\\(\theta\\)是待确定的参数。\\(theta\\)有两部分：

* 词向量：\\(v(w) \in \Re^m \\) , \\(w \in D\\) 以及填充向量
* 神经网络参数：\\(W \in \Bbb R^{n\_h \times (n-1)m},p \in \Bbb R^{n\_h};U \in R^{N \times n\_h},q \in \Bbb R^N\\)

这些参数均通过训练算法得到.值得一提的是，通常的机器学习算法中，输入都是已知的，而在上述神经概率语言模型中，输入\\(v(w)\\)也需要通过训练才能得到。

#### 3.1.4 词向量的理解
在NLP任务中，我们将自然语言交给机器学习算法来处理，但机器无法直接理解人类的语言，因此首先要做的事情就是将语言数字化，词向量提供了一种很好的方式。一种最简单的词向量是one-hot representation，他就是用一个很长的向量来表示一个词，向量的长度为词典\\(D\\)的大小N,向量的分量只有一个1，其余全为0,1的位置对应该词在词典中的索引。但是这种词向量表示又有一些缺点，容易受维数灾难的困扰，尤其是将其应用到Deep Learning的场景时。另一种词向量是Distributed Representation，它最早是Hinton于1986年提出的{% cite rumelhart1988learning%}，可以克服one-hot representation的上述缺点，其基本思想是：通过训练将某种语言中的每一个词映射成一个固定长度的短向量，所有这些向量构成一个词向量空间，而每一个则可以视为该空间中的一个点，在这个空间上引入“距离”，就可以根据词之间的距离来判断他们之间的相似性了。Word2vec采用的就是这种Distributed Representation的词向量。

### 3.2 基于 Hierarchical Softmax 的模型
word2vec中用到了两个重要模型 - CBOW(Continuous Bag-of-Words Model)模型和Skip-gram模型(Continuous Skip-gram Model)，关于这两个模型，作者Tomas Mikolov在文{%cite mikolov2013distributed%}给出了如图2和图3所示的模型
{% include  image.html
            img="/public/img/cbow.png"
            title="cbow"
            caption="图2 CBOW模型" 
%}

{% include  image.html
            img="/public/img/skip-gram.png"
            title="skip-gram"
            caption="图3 Skip-gram模型" 
%}
对于CBOW和Skip-gram两个模型，word2vec给出了两套框架，他们分别是基于Hierarchical Softmax和Negative Sampling来进行设计，本节介绍基于Hierarchical Softmax的CBOW和Skip-gram模型。在3.1节中我们提到基于神经网络的语言模型的目标函数通常取为如下对数似然函数

\begin{equation}
L = \sum\_{w \in C} logP(w\|Context(w))
\label{eq:eq15}
\end{equation}

其中的关键是条件概率函数\\(P(w\|Context(w))\\)的构造，文{%cite bengio2003neural%}中的模型就给出了这个函数的一种构造方法，即公式\eqref{eq:eq14}。对于word2vec中基于Hierarchical Softmax的CBOW模型，优化的目标函数也形如公式\eqref{eq:eq15}；而对于基于Hierarchical Softmax的Skip-gram模型，优化的目标函数则形如：

\begin{equation}
L = \sum\_{w \in C} logP(Context(w)\|w)
\label{eq:eq16}
\end{equation}

下面将介绍\\(p(w\|Context(w))\\)或者\\(p(Context(w)\|w)\\)的构造。

#### 3.2.1 CBOW模型
本小节介绍word2vec中的第一个模型---CBOW模型。
##### 3.2.1.1 网络结构
图四给出了CBOW模型的网路结构，它包括三层:输入层、投影层、输出层。下面以样本\\(Context(w),w\\)为例(这里假设Context(w)由w前后各c个词构成)，下面对这三个层作简要说明。

1. **输入层**: 包含Context(w)中的2c个词的词向量\\(v(Context(w)\_1),v(Context(w)\_2),\cdots,v(Context(w)\_{2c}) \in R^m\\),这里,m的含义同上表示词向量的长度。

2. **投影层**: 将输入层的2c个向量做求和累加，即\\(x\_w = \sum\_{i=1}^2c v(Context(w)\_i) \in R^m\\)

3. **输出层**: 输出层对应一棵二叉树，它是以语料中出现过的词当叶子节点，以各词在语料中出现的次数当权值构造出来Huffman树，在这颗Huffman树中，叶子节点
   共N(=|D|)个，分别对应词典D中的词，非叶子节点N-1个(图中标成黄色的那些顶点)。

对比神经概率语言模型的网络图(见图2和图3)和CBOW模型的结构图(见图4)，易知它们主要有以下三处不同:

1. (从输入层到投影层的操作) 前者是通过拼接，后者通过累加求和。

2. (隐藏层) 前者有隐藏层，后者无隐藏层。

3. (输出层) 前者是线性结构，后者树形结构。  

{% include  image.html
            img="/public/img/cbow-net.png"
            title="cbow-net"
            caption="图4 CBOW模型的网络结构" 
%}

在3.1.3节介绍的神经概率语言模型中，我们指出，模型的大部分计算集中在隐藏层和输出层之间的矩阵向量运算，以及输出层上的softmax归一化运算。而从上面的对比中可见，CBOW模型对这些计算复杂度高的地方有针对性的进行了改变，首先去掉了隐藏层，其次，输出层改用Huffman树，从而为利用Hierarchical softmax技术奠定了基础。

##### 3.2.1.2 梯度计算

Hierarchical Softmax是word2vec中用于提高性能的一项关键技术，为了描述方便起见，在具体介绍这个技术之前，先引入若干相关记号。考虑Huffman树中的某个叶子节点，假设它对应词典D中的词w，记

1. \\(p^w\\)：从根结点出发到达\\(w\\)对应
2. \\(l^w\\)：路径\\(p^w\\)中包含的结点的个数
3. \\(p\_q^w,p\_2^w,\cdots,p\_{l^w}^w\\): 路径\\(p^w\\)中的\\(l^w\\)个结点,其中\\(p\_1^w\\)表示根节点,\\(p\_{l^w}^w\\)表示词w对应的结点。
4. \\(d\_2^w,d\_3^w,\cdots,d\_{l^w}^w \in {0,1}\\)：词w的Huffman编码，它由\\(l^w-1\\)位编码构成，\\(d\_j^w\\)表示路径\\(p^w\\)中第j个结点对应的编码(根节点不对应编码)。
5. \\(\theta\_1^w,\theta\_2^w,\cdots,\theta \in R^m\\)：路径\\(p^w\\)中非叶子结点对应的向量,\\(\theta\_j^w\\)表示路径\\(p^w\\)中第j个非叶子结点对应的向量。
 
下面介绍如何利用\\(x\_w \in R^m\\) 以及Huffman树来定义函数\\(p(w\|Context(w))\\),我们从二分类的角度考虑问题，那么对于每一个非叶子结点，就需要为其左右孩子结点指定一个类别，即哪一个是正类(标签为1)，哪一个是负类(标签为0)。碰巧，除了根节点以外，树中每个结点都对应了一个取值为0或1的Huffman编码。因此，一种最自然的做法就是将Huffman编码为0的结点定义为正类，编码为1的结点定义为
负类，word2vec选用的这个约定：

\begin{equation}
Lable(P\_i^w) = 1 - d\_i^w, i = 2,3,4,\cdots,l^w
\label{eq:eq17}
\end{equation}
所以根据逻辑回归，一个结点分为正类的概率是\\(\sigma(x\_w^T \theta) = {1 \over 1 + e^{-x\_w^T \theta}}\\),被分为负类的概率为\\(1-\sigma(x\_w^T \theta)\\),其中\\(\theta\\)是待定参数，非叶子结点对应的那些向量\\(\theta\_i^w\\)就可以扮演参数\\(\theta\\)的角色。对于词典D中的任意词w，Huffman树中必存在一条从根结点到词w对应结点的路径\\(p^w\\)（且这条路径是唯一的）。路径\\(p^w\\)上存在\\(l^w-1\\)个分支，将每个分支看做一次二分类，每一次分类就产生一个概率，将这些概率乘起来，就是所需的\\(P(w|Context(w))\\)。
条件概率

\begin{equation}
p(w\|Context(w)) = \prod\_{j=2}^{l^w}p(d\_j^w|X\_w,\theta\_{j-1}^w)
\label{eq:eq18}
\end{equation}

其中\\(p(d\_j^w\|X\_w,\theta\_{j-1}^w) = [\sigma(X\_w^T\theta\_{j-1}^w)]^{1-d\_j^w} \bullet [1-\sigma(X\_w^T\theta\_{j-1}^w)]^{d\_j^w}\\)
将公式\eqref{eq:eq18}带入公式\eqref{eq:eq15}得到

\begin{equation}
L = \sum\_{w \in C} \sum\_{j=2}^{l^w}{(1-d\_j^w)log[\sigma(X\_w^T\theta\_{j-1}^w)]+d\_j^wlog[1-\sigma(X\_w^T\theta\_{j-1}^w)]}
\label{eq:eq19}
\end{equation}

至此，已经推导出了对数似然函数\eqref{eq:eq19}，这个就是CBOW模型的目标函数，下面利用随机梯度下降法来优化这个目标函数，观察目标函数\eqref{eq:eq19}易知，该函数中的参数包括向量\\(X\_w,\theta\_{j-1}^w,w \in C,j = 2,\cdots,l^w\\)。通过求导可得

\begin{equation}
\theta\_{j-1}^w := \theta\_{j-1}^w + \eta[1 - d\_j^w - \sigma(X\_w^T\theta\_{j-1}^w)]X\_w
\label{eq:eq20}
\end{equation}
其中\\(\eta\\)表示学习率，下同。

\begin{equation}
v(\widetilde{w}) := v(\widetilde{w}) + \eta \sum\_{j=2}^{l^w} {\partial L(w,j) \over \partial X\_w }, \widetilde{w} \in Context(w)
\label{eq:eq21}
\end{equation}

其中
\\[
{\partial L(w,j) \over \partial X\_w } = [1-d\_j^w -\sigma(X\_w^T\theta\_{j-1}^w)]\theta\_{j-1}^w
\\]
下面以样本(Context(w),w)为例，给出了CBOW模型中采用随机梯度下降法更新各参数的伪代码
{% include  image.html
            img="/public/img/cbow-pseudocode.png"
            title="cbow-pseudocode.png"
            caption="图5 CBOW模型训练方法" 
%}

#### 3.2.2 Skip-gram模型
本小结介绍word2vec中的另一个模型---Skip-gram模型，由于推导过程与CBOW大同小异，因此会沿用上节引入的记号。
##### 3.2.2.1 网络结构

图6给出了Skip-gram模型的网络结构，同CBOW模型网络结构也一样，它也包括三层：输入层，投影层，输出层，下面以样本\\(w,Context(w)\\)为例，对这三个层做简要说明。

1. **输入层**：只含有当前样本中心词\\(w\\)的词向量\\(v(w) \in R^m\\)。
2. **投影层**: 这是个恒等投影，把\\(v(w)\\)投影到\\(v(w)\\)。因此，这个投影层其实是多余的，这里之所以保留投影层只要是方便和CBOW模型的网络结构做对比。
3. **输出层**: 和CBOW模型一样，输出层也是一颗Huffman树

{% include  image.html
            img="/public/img/skip-gram-net.png"
            title="skip-gram-net"
            caption="图6 Skip-gram模型的网络结构示意图" 
%}

##### 3.2.2.2 梯度计算
对于Skip-gram模型，已知的是当前词w，需要对其上下文Context(w)中的词进行预测，因此目标函数应该形如公式\eqref{eq:eq16},且关键是条件概率函数\\(p(Context(w)\|w)\\)的构造，
Skip-gram模型将其定义为
\\[
p(Context(w)|w) = \prod\_{u \in Context(w)} p(u\|w)
\\]
上式中的\\(p(u\|w)\\)可以按照上一小节介绍的Hierarchical Softmax思想，写为
\\[
p(u\|w) = \prod\_{j=2}^{l^u} p(d\_j^u|v(w),\theta\_{j-1}^u)
\\]
其中

\begin{equation}
p(d\_j^u|v(w),\theta\_{j-1}^u) = [\sigma(v(w)^T\theta\_{j-1}^u)]^{1-d\_j^u}[1-\sigma(v(w)^T\theta\_{j-1}^u)]^{d\_j^u}
\label{eq:eq22}
\end{equation}

将公式\eqref{eq:eq22}依次带回目标函数\eqref{eq:eq16}可以得到对数似然函数的具体表达式

\begin{equation}
L = \sum\_{w \in C} \sum\_{u \in Context(w)} \sum\_{j=2}^{l^u}  (1-d\_j^u)log[\sigma(v(w)^T\theta\_{j-1}^u)]+d\_j^ulog[1-\sigma(v(w)^T\theta\_{j-1}^u)]
\label{eq:eq23}
\end{equation}
下面对目标函数求导后可知参数的更新公式为：
\\[
\theta\_{j-1}^u := \theta\_{j-1}^u + \eta[1-d\_j^u - \sigma(v(w)^T\theta\_{j-1}^u)]v(w)
\\]

\\[
v(w) := v(w) + \eta \sum\_{u \in Context(w)} \sum\_{j=2}^{l^u} {\partial L(w,u,j) \over \partial v(w)}
\\]

其中
\\[
 {\partial L(w,u,j) \over \partial v(w)} = [1 -d\_j^u - \sigma(v(w)^T\theta\_{j-1}^u)]\theta\_{j-1}^u
\\]
下面以样本\\(w,Context(w)\\)为例，给出Skip-gram模型中采用随机梯度下降法更新各参数的伪代码
{% include  image.html
            img="/public/img/skip-gram-pseudocode.png"
            title="skip-gram-pseudocode"
            caption="图7 Skip-gram模型训练方法" 
%}

### 3.3 基于Negative Sampling的模型
本节将介绍基于Negative Sampling的CBOW和Skip-gram模型。Negative Sampling(简称NEG)是Tomas Mikolov等人在{% cite mikolov2013distributed%}中提出的，它是NCE(Noise Contrastive Estimation)的一个简化版本，目的是用来提高训练速度并改善所得词向量的质量。与Hierarchical Softmax相比，NEG不再使用复杂的Huffman树，而是利用(相对简单的)随机负采样，能大幅提高性能，因此可作为Hierarchical Softmax的一种替代。

#### 3.3.1 CBOW模型
在CBOW模型中，已知词w的上下文Context(w),需要预测w,因此对于给定的Context(w)，词w就是一个正样本，其他词就是负样本，现在假定已经选好了一个关于w的负样本子集\\(NEG(w) \neq \emptyset \\)
且对\\(\forall \widetilde{w} \in D\\),定义

$$
f(n) =  
\begin{cases}  
1, &\text{$\widetilde{w} = w$} \\[2ex]
0, &\text{$\widetilde{w} \neq w$}  
\end{cases}
$$

表示词\\(\widetilde{w}\\)的标签，即正样本标签为1，负样本标签为0.

给定一个正样本(Context(w),w),我们希望最大化

\begin{equation}
g(w) = \prod\_{u \in {w}\bigcup NEG(w)} p(u\|Context(w))
\label{eq:eq24}
\end{equation}
其中

$$
p(u|Context(w)) =  
\begin{cases}  
\sigma(X_w^T\theta^u), &\text{$L^w(u) =1$} \\[2ex]
1-\sigma(X_w^T\theta^u), &\text{$L^w(u)=0$}  
\end{cases}
$$

这里\\(X\_w\\)仍表示Context(w)中各词的词向量之和，而\\(\theta^u \in R^m\\)表示词u对应的一个辅助向量，为待训练参数。负采样的思想是增大正样本的概率同时降低负样本的概率，于是，
对于一个给定的语料库\\(C\\),函数

\\[
G = \prod \_{w \in C}g(w)
\\]
就可以作为整体优化目标，为了计算方便，对G取对数，最终的目标函数为

\begin{equation}
L = \sum\_{w \in C} \sum\_{u \in {w} \bigcup NEG(w) }L^w(u)log[\sigma(x\_w^T \theta^u)]+[1-L^w(u)]log[1- \sigma(x\_w^T \theta^u)]
\label{eq:eq25}
\end{equation}

利用随机梯度下降来计算参数的更新公式：
\\[
\theta^u := \theta^u + \eta[L^w(u) -\sigma(x\_w^T\theta^u)]x\_w
\\]

\\[
v(\widetilde{w}) := v(\widetilde{w}) + \eta \sum\_{u \in {w} \bigcup NEG(w)} {\partial L(w,u) \over \partial x\_w} ,\widetilde{w} \in Context(w)
\\]

其中
\\[
{\partial L(w,u) \over \partial x\_w} = [L^w(u) - \sigma(x\_w^T\theta^u)]\theta^u
\\]
下面以样本(Context(w),w)为例，给出基于Negtive Sampling的CBOW模型中采用随机梯度下降法更新各参数的伪代码

{% include  image.html
            img="/public/img/neg-cbow-code.png"
            title="neg-cbow-code"
            caption="图8 基于负采样的CBOW模型训练方法" 
%}
#### 3.3.2 Skip-gram模型
本小节介绍基于Negative Sampling的Skip-gram模型，将CBOW下的目标函数改写为

\begin{equation}
G = \prod\_{w \in C} \prod\_{u \in Context(w)} g(u)
\label{eq:eq26}
\end{equation}

这里\\(\prod\_{u \in Context(w)} g(u)\\)表示对于一个给定的样本(w,Context(w)),我们希望最大化的量，\\(g(u)\\)类似于上一节的\\(g(w)\\),定义为

\begin{equation}
g(u) = \prod\_ {z \in u \bigcup NEG(u)} p(z\|w)
\label{eq:eq27}
\end{equation}

其中NEG(u)表示处理词u时生成的负样本子集，条件概率

$$
p(z|w) =  
\begin{cases}  
\sigma(v(w)^T\theta^z), &\text{$L^u(z) =1;$} \\[2ex]
1-\sigma(v(w)^T\theta^z), &\text{$L^u(z)=0$}  
\end{cases}
$$

所以最终的目标函数为

\begin{equation}
L = logG = \sum\_{w \in C} \sum\_{\widetilde{w} \in Context(w)} \sum\_ {u \in {w} \bigcup NEG^{\widetilde{w}}(w)}{L^w(u)log[\sigma(v(\widetilde{w})^T \theta^u)]+[1-L^w(u)]log[1-\sigma(v(\widetilde{w})^T \theta^u)]} 
\label{eq:eq28}
\end{equation}

于是\\(\theta^u\\)的更新公式可写为
\\[
\theta^u := \theta^u + \eta[L^w(u) - \sigma(v(\widetilde{w})^T\theta^u)]v(\widetilde{w})
\\]

\\(v(\widetilde{w})\\)的更新公式可以写为

\\[
v(\widetilde{w}) := v(\widetilde{w}) + \eta \sum\_ {u \in {w} \bigcup NEG^{\widetilde{w}}(w)} {\partial L(w,\widetilde{w},u) \over \partial v(\widetilde{w})}
\\]
下面以样本(w,Context(w))为例,给出基于Negative Sampling的Skip-gram模型中采用随机梯度下降法更新各参数的伪代码

{% include  image.html
            img="/public/img/neg-skip-gram-code.png"
            title="neg-skip-gram-code"
            caption="图9 基于负采样的Skip-gram模型训练方法" 
%}

### 3.4 大规模消息网络嵌入式表示

大规模消息网路嵌入式表示(Large-scale Information Network Embeding，简记LINE)主要是用来研究大规模消息网络结点间的关系，并用低维向量空间来表示该网络结构，这项技术广泛应用在多个领域，比如：数据可视化，结点分类，链接预测。LINE相比于
当前存在的图嵌入技术{% cite tenenbaum2000global%} {%cite belkin2001laplacian%}的主要优点是它可以适用于现实世界中的真实网络(拥有上百万顶点，上千万边的网络)。LINE适用于任意类型的消息网络，比如说，无向网络，有向网络，
带权重的网络。这种方法通过优化保留了局部和全局网络结构的目标函数来刻画消息网络的特征从而得到每一个结点的低维向量表示。局部网络结构，又称一阶相似性，捕获的是网络中两个顶点的链接关系，大部分图嵌入方法均可以保留一阶相似性，比如IsoMap{% cite tenenbaum2000global%}。由于在现实世界的真实网络中，很多合理的链接并没有被捕获，仅仅通过一阶相似性不足以表示全局网络结构，文{% cite tang2015line%}中提出了结点间的二阶相似性。二阶相似性通过判断两个结点间是否共享邻居来判断这两个结点是否相似，这个思想和我们直观上的想法---“我们可以通过某人的朋友来了解一个人”。

下面举例说明，在图10中的消息网络中，边可以是有向的，无向的或者带权重的。顶点6和顶点7由于存在边直接连接，所以顶点6和顶点7存在一阶相似性。顶点5和顶点6由于共享相同的邻居(顶点1，顶点2，顶点3，顶点4),所以顶点5和顶点6具有二阶相似性。

{% include  image.html
            img="/public/img/line-2-prox.png"
            title="line-2-prox"
            caption="图10 网络中的二阶相似性" 
%}

下面具体来用数学公式来刻画一阶相似性和二阶相似性。

#### 3.4.1 一阶相似性

一阶相似性指的是网络中两个顶点的局部成对相似性，为了对一阶相似性建模，对于任意无向边\\((i,j)\\),我们定义了顶点\\(v\_i\\)和\\(v\_j\\)的联合概率如下：

\begin{equation}
p\_ 1(v\_ i,v\_ j) = {1 \over 1+exp(-\vec{u\_ i}^T \cdot \vec{u \_ j})}
\label{eq:eq29}
\end{equation}

其中\\(\vec{u\_ i} \in R^d\\)是顶点\\(v\_ i\\)的低维向量表示，\eqref{eq:eq29}定义了在\\(V \times V\\)空间上的一个概率分布\\(p(\cdot,\cdot)\\),它的经验分布为
\\(\hat{p\_ 1} = {w\_ {ij} \over W}\\),其中\\(W = \sum \_{(i,j) \in E} w\_{ij}\\),为了保留一阶相似性，一个最直接的想法就是最小化前面两个分布的距离，即目标函数为

\begin{equation}
O\_ 1 = - \sum \_ {(i,j) \in E} w\_ {ij} \log p\_ 1(v\_ i,v\_ j) 
\label{eq:eq30}
\end{equation}

一阶相似性只适合于无向图，不适合于有向图，通过找到\\( \\{\vec{u\_ i}  \\}\_{i = 1,\cdots,\|V\|}\\)来最小化目标函数\eqref{eq:eq30}，我们便可以得到每个顶点的\\(d\\)维向量表示。

#### 3.4.2 二阶相似性
二阶相似性适用于无向图和有向图，为了不损失一般性，我们考虑一个有向图网络(无向边可以看做两个具有相同权重但是方向相反的有向边)，二阶相似性认为如果两个顶点共享邻居结点的话，那么这两个结点相似。在这种情况下，每一个顶点被指定一个上下文(Context)，如果顶点在上下文上具有相似的分布，那么认为这两个顶点是相似的。
因此，一个顶点担任两个角色，第一，顶点本身；第二，其他顶点的上下文(Context)。所以我们引入两个向量\\(\vec{u\_i} 和 {\vec{u\_i}}^{\prime}\\)，其中\\(\vec{u\_i}\\)是\\(v\_i\\)的顶点表示，\\({\vec{u\_i}}^{\prime}\\)是\\(v\_i\\)的上下文表示，对于每一个有向边(i,j)，我们定义通过顶点\\(v\_i\\)生成上下文\\(v\_j\\)的概率如下

\begin{equation}
p\_2(v\_ j\|v\_ i) = {exp({\vec{u\_ j}^{\prime}}^T \cdot \vec{u\_ i} ) \over \sum \_ {k=1} ^{\|V\|} exp({\vec{u\_ k}^{\prime}}^T \cdot \vec{u\_ i} )}
\label{eq:eq31}
\end{equation}

其中\\(\|V\|\\)是顶点或者上下文的数量，对于每一个顶点\\(v\_i\\)，公式\eqref{eq:eq31}定义了在\\(v\_i\\)在上下文下的条件概率分布\\(p\_2(\cdot\|v\_ i)\\),正如上文所说，二阶相似性表示的是，如果顶点在上下文上具有相似的概率分布，那么这些顶点相似。为了保留二阶相似性，我们应使\\(p\_2(\cdot\|v\_i)\\)与经验分布\\(\hat{p\_2}(\cdot\|v\_i)\\)距离最近,因此我们定义如下目标函数。

\begin{equation}
O\_2 = \sum\_{i \in V} \lambda\_ i d(p\_2(\cdot\|v\_i),\hat{p\_2}(\cdot\|v\_i))
\label{eq:eq32}
\end{equation}
其中\\(d(\cdot,\cdot)\\)表示两个概率分布之间的距离，由于网络中顶点之间的重要性不同，\\(\lambda\_i\\)表示每个顶点的重要性，它可以用每个顶点的度或者PageRank{% cite page1999pagerank%},其中\\(\hat{p\_2}(v\_j\|v\_ i) = {w\_{ij} \over d\_ i}\\),\\(w\_{ij}\\)是边(i,j)的权重，\\(d\_i\\)是顶点i的出度，我们通过KL距离来计算\\(d(\cdot,\cdot)\\)并且\\(\lambda\_i = d\_i\\),目标函数可以简化为
\begin{equation}
O\_2 = - \sum \_{(i,j) \in E} w\_{ij}\log p\_2(v\_j\|v\_i)
\label{eq:eq33}
\end{equation}

通过学习出\\(\\{\vec{u\_ i}\\}\_{i = 1 \cdots \|V\|}\\) 和\\(\\{\vec{u\_ i}^{\prime}\\}\_{i = 1 \cdots \|V\|}\\)使目标函数达到最小，
我们便可以用一个d维向量\\(\vec{u\_ i}\\)表示每一个顶点\\(v\_i\\)。






























## 4. 主要工作

## 5. 实验结果

## 本文总结和对未来的工作展望

References
----------
{% bibliography --cited\_in\_order %}