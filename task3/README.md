# Task2
基本文本处理技能
## 任务要求：
*    基本文本处理技能：中英文字符串处理（删除不相关的字符、去停用词）；分词（结巴分词）；词、字符频率统计。
*    语言模型；`unigram`、`bigram`、`trigram`频率统计。
*    [jiebe](https://github.com/fxsjy/jieba)分词介绍和使用


1. 中英文字符串处理（删除不相关的字符、去停用词）
   * 以保留相关字符方式删除不相关字符
   * `jieba`分词
   * 去停用词
2. 词、字符频率统计
3. 语言模型
> 统计语言模型是一个单词序列上的概率分布，对于一个给定长度为m的序列，它可以为整个序列产生一个概率，即想办法找到一个概率分布，它可以表示任意一个句子或序列出现的概率。
* `unigram`：
  >**一元文法模型**——上下文无关模型
  > 该模型只考虑当前词本身出现的概率，而不考虑当前词的上下文环境。
  
    <a href="https://www.codecogs.com/eqnedit.php?latex=P(w_1,&space;w_2,...,w_m)=P(w_1)*P(w_2)*...*P(w_m)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(w_1,&space;w_2,...,w_m)=P(w_1)*P(w_2)*...*P(w_m)" title="P(w_1, w_2,...,w_m)=P(w_1)*P(w_2)*...*P(w_m)" /></a>
  > 每个句子出现的概率为每个单词概率成绩
*  N-gram
 依赖于上下文环境的词的概率分布的统计计算机语言模型。可以理解为当前词的概率与前面的n个词有关系
   - `bigram`：当N=2时称为二元 `bigram`模型，当前词只与它前面的一个词相关，这样概率求解公式：
<a href="https://www.codecogs.com/eqnedit.php?latex=P(w_1,w_2,...,w_m)=\prod_{i=1}^{m}&space;P(w_i|w_{i-(n-1)},...,w_{i-1})=\prod_{i=1}^{m}&space;P(w_i|w_{i-1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(w_1,w_2,...,w_m)=\prod_{i=1}^{m}&space;P(w_i|w_{i-(n-1)},...,w_{i-1})=\prod_{i=1}^{m}&space;P(w_i|w_{i-1})" title="P(w_1,w_2,...,w_m)=\prod_{i=1}^{m} P(w_i|w_{i-(n-1)},...,w_{i-1})=\prod_{i=1}^{m} P(w_i|w_{i-1})" /></a>

   * `trigram`: 当N=3时称为三元`trigram`模型，同理当前词只与它前面的两个词相关
