---
layout: post
comments: true
title: 在Jekyll中集成Disqus的原理
category: 技术
tags: Jekyll
keywords: Jekyll Disqus Js
description: 
---
# Disqus
**Disqus**是一个第三方的JavaScript应用，它通过用户向自己网站(比如Blog)注入js脚本，用户通过嵌入的一小段JavaScript脚本
向Disqus服务器发送请求，用来初始化JavaScript Loader，这个Loader可以在用户网站创建所需的iframe元素，并从Disqus服务器
获取数据，渲染魔板，并将该页面所需的评论数据注入页面。

## 后端架构
Disqus后台用了很多技术来支持这看似简单的操作，后台每天需要处理上百万的读请求，主要用到的技术有Python，Django，PostgreSQL
，由于Disqus的业务绝大部分是实时业务，所以也需要用Redis缓存技术。

## 加载第三方JS的技术
> 网页中加载JS文件是一个老问题了，已经被讨论了一遍又一遍，这里不会再赘述各种经典的解决方案。JS文件可以通过来源来分为两个纬度：第一方JS和第三方JS。第一方JS是网页开发者自己使用的JS代码（内容开发者可控）。而第三方JS则是其他服务提供商提供的（内容开发者不可控），他们将自己的服务包装成JS SDK供网页开发者使用。Disqus用到了第三方JS文件的加载技术。

涉及到的文件有_includes/disqus.html, _layouts/post.html



