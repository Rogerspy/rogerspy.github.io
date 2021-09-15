import os
import re

s = open('index.md', encoding='utf8').read()
head_p = re.compile('<li><a href=".*">(.*)</a>')
heads = head_p.findall(s)

pp = re.compile('<li><a href="(.*).html"')
pl = pp.findall(s)

p_head = re.compile('<h4.*</h4>')


prelink = '<iframe src="https://player.bilibili.com/player.html?aid=9770302&cid=16150508&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>'
l = '<iframe src="https://player.bilibili.com/player.html?aid=9770302&cid=16150508&page=%s" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>'
for i, f in enumerate(heads[1:]):
    nf = open(pl[i+1]+'.md', 'w+', encoding='utf8')
    ns = re.sub(p_head, '<h4 class="title-video heading-video">'+heads[i+1]+'</h4>', s)
    ns = ns.replace(prelink, l % (i+2))
    nf.write(ns)
    nf.close()