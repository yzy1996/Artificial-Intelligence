## 改变字体大小

全局修改在开头

> \documentclass[12pt]{article}

12pt就是正文的大小，一般11或者12就比较合适了

局部的是这那一句前加 `\large`，可以选择：

`\tiny \scriptsize \footnotesize \small \normalsize \large \Large \LARGE \huge \Huge`

## 改变字体颜色

>  \textcolor{-color}{-text}

`\textcolor{red}{Main idea}`

## 插入图片以及改变图片位置、大小

最直接的用法是  `\includegraphics{figname}`

**修改大小**

`\includegraphics[scale=0.5]{figname}`  按照原图大小进行比例缩放

`\includegraphics[width=0.5\textwidth]{figname}`  按文档一行宽度进行比例缩放

**修改位置**

最常用的就是这样紧跟着文本，但如果大小不合适，文本会随着图片一起移到下一页

```
\begin{figure}[H] % H need add \usepackage{float}
\includegraphics{figname}
\centering
\end{figure}
```

## 插入参考文献

自己看 [链接](https://www.overleaf.com/learn/latex/Bibliography_management_in_LaTeX)