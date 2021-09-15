---
ctype: blog
title: Transformer代码实现-Tensoflow版
top: false
cover: true
toc: true
mathjax: true
date: 2019-09-16 20:28:30
password:
summary: Transformer代码实现
tags: [Transformer, tensorflow]
categories: [NLP]
body: [article, comments]
gitalk:
  id: /wiki/material-x/
---

前面介绍了Transformer的`pytorch`版的代码实现，下面我们再介绍一下`tensorflow`版的代码实现。

<!-- more -->

本文主要参考的是`tensorflow`[官方教程](https://www.tensorflow.org/beta/tutorials/text/transformer)，使用的是`tensoflow 2.0`，因此首先还是要先搭建代码环境，可以参考这里：[简单粗暴 TensorFlow 2.0](https://tf.wiki/zh/basic/installation.html)。

# 1. 前期准备

```python
from __future__ import absolute_import, division, print_function, unicode_literals
try:
    %tensorflow_version 2.x
except Exception:
    pass
import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
```

# 2. Scaled Dot-Product Attention

![](https://img.vim-cn.com/ed/97e04d7d6067cb360e8fef1d29cf41978d353e.png)

```python
def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimension.
    k, v must have matching penultimate dimension, i.e.:seq_len_k = seq_len_v.
    The mask has different shapes depending on its type (padding or look ahead)
    but it must be broadcastable for addition.
    
    :params q: query shape == (..., seq_len_q, depth)
    :params k: key shape == (..., seq_len_k, depth)
    :params v: value shape == (..., seq_len_v, depth)
    :params mask: Float tensor with shape bradcastable to 
                  (None, seq_len_q, seq_len_k), Default is None.
    """
    # MatMul step in above Fig
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
   
    # Scale step in above Fig
    # This is done because for large values of depth, the dot product grows 
    # large in magnitude pushing the softmax function where it has small 
    # gradients resulting in a very hard softmax.
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention = matmul_qk / tf.math.sqrt(dk)
    
    # Mask step in above Fig
    # This is done because the mask is summed with the scaled matrix 
    # multiplication of Q and K and is applied immediately before a softmax. 
    # The goal is to zero out these cells, and large negative inputs to 
    # softmax are near zero in the output.
    if mask is not None:
        scaled_attention += (mask * -1e9) 
        
    # SoftMax step in above Fig
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1
    attention_weights = tf.nn.softmax(scaled_attention, axis=-1)
    
    # The last MatMul step in above Fig
    out = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    
    return out, attention_weights
```

# 3. Multi-Head Attention

![](https://img.vim-cn.com/b1/e4bc841abc55d366813340f92f6696c5d59e95.png)

*Multi-Head Attention*有四部分组成：

- 线性转换层和multi-head (Q, K, V)
- *Multi-head Scaled dot-product attention*
-  *Concatenation of heads*
- 最后的线性转换层

```python
class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Implement Multi=head attention layer.
    """
    def __init__(self, d_mode, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model= d_model
        self.num_heads = num_heads
        
        # after `Concat`, concatenated heads dimension must equal to d_model
        assert d_model % num_heads == 0
        
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
   
    def split_heads(self, x, batch_size):
        """
        Split the last dimension (word vector dimension) into (num_heads, depth).
        Transpose the result such that the shape is 
        (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]
        
        # First linear transition step in above Fig
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model
        
        # Split K, Q, V into multi-heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # Scaled Dot-Product Attention step in above Fig
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Concat step in above Fig
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # scale_attention.shape == (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        # concate_attention.shaoe == (batch_size, seq_len_q, d_model)
        
        # Final linear transition step in above Fig
        out = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return out, attention_weights
```

# 4. Point wise feed forward network

*Point wise feed forward network*由两个全连接层组成，激活函数使用*Relu*：

```python
def point_wise_feed_forward_network(d_model, d_ff):
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation='relu'),  # (batch_size, seq_len, d_ff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])
    return ffn
```

# 5. Positional encoding

$$
PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}})
$$

$$
PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})
$$

```python
def get_angles(pos, i, d_model):
    """
    Get the absolute position angle from each word.
    
    :params pos: position index
    :params i: word embedding dimension index at each position
    :params d_model: model dimension
    """
    angle_rates = 1 / np.power(10000, (2*(i//2)) / np.float32(d_model))
    return pos * angle_rates
```

```python
def positional_encoding(position, d_model):
    """
    Compute positional encoding.
    
    :params position: length of sentence
    :params d_model: model dimension
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)  # (position, d_model)
    
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    # positional encoding
    positional_encoding = angle_rads[np.newaxis, ...]
    
    return tf.cast(positional_encoding, dtype=tf.float32)
```

# 6. Masking

这里有两种Mask，一种用来mask掉输入序列中的padding，一种用来mask掉解码过程中“未来词”。

- Mask每个batch中所有序列的padding token，使得模型不会把padding token当成输入：

```python
def create_padding_mask(seq):
    """
    Mask all the pad tokens in the batch of sequence. 
    It ensures that the model does not treat padding as the input. 
    The mask indicates where pad value 0 is present: 
    it outputs a 1 at those locations, and a 0 otherwise.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
```

- Mask掉解码过程中的“未来词”：

```python
def create_look_ahead_mask(size):
    """
    The look-ahead mask is used to mask the future tokens in a sequence. 
    In other words, the mask indicates which entries should not be used.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)
```

# 7. Encoder and Decoder

![](https://img.vim-cn.com/3a/78ea12dca1ce0f99f9a9705466afc16c58c3cf.png)

Transformer和标准的*seq2seq with attention*模型一样，采用*encoder-decoder*结构，*encoder / decoder*都包含了6个结构相同的*encoder layer*和*decoder layer*。

## 7.1 Encoder

*Encoder layer*由两个sub-layer组成：

- Multi-head attention
- Point wise feed forward network

每个sub-layer后面都接一个layer normalization，使用残差连接防止梯度消失。

```python
class EncoderLayer(tf.keras.layers.Layer):
    """
    Implements Encoder Layer.
    """
    def __init__(self, d_model, num_heads, d_ff, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.fropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, padding_mask):
        # Multi-head attention sub-layer
        attention_out, _ = self.multihead_attention(x, x, x, padding_mask)  
        # attention_out.shape == (batch_size, input_seq_len, d_model)
        attention_out = self.dropout1(attention_out, training=training)
        attn_norm_out = self.layernorm1(x + attention_out)  
        # attn_norm_out.shape == (batch_size, input_seq_len, d_model)
        
        # point wise feed forward network sub-layer
        ffn_out = self.ffn(attn_norm_out)  # (batch_size, input_seq_len, d_model)
        ffn_out = self.dropout2(ffn_out, training)
        ffn_norm_out = self.layernorm2(attn_norm_out + ffn_out)  
        # ffn_norm_out.shape == (batch_size, input_seq_len, d_model)
        
        return ffn_norm_out
```

*Encoder*由三部分组成：

- 输入Embedding
- Positional Encoding
- N个*encoder layer*

```python
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, 
                 d_ff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        # Positional encoding layer
        self.pos_encoding = positional_encoding(input_vocab_size, d_model)
        
        # encoder layers
        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_ff, rate)
                               for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, padding_mask):
        """
        The input is put through an embedding which is summed with the positional encoding.
        The output of this summation is the input to the encoder layers. 
        The output of the encoder is the input to the decoder.
        """
        seq_len = tf.shape(x)[1]
        
        # adding embedding and positional encoding
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # ????
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        
        # encoder layer
        for encoder in self.encoder_layers:
            x =encoder(x, training, padding_mask)
            
        return x  # (batch_size, input_seq_len, d_model)
```

## 7.2 Decoder

*Decoder layer*由三个sub-layer组成：

- Masked multi-head attention (with look ahead mask and padding mask)
- Multi-head attention (with padding mask)。其中Q（query）来自于前一层（或者输入层）的输出， K（key）和V（value）来源于*Encoder*的输出。
- Point wise feed forward networks

与*encoder layer*类似，每个sub-layer后面会接一个layer normalization，同样使用残差连接。

```python
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.multihead_attention1 = MultiHeadAttention(d_model, num_heads)
        self.multihead_attention2 = MultiHeadAttention(d_model, num_heads)
        
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, encoder_out, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        
        # Masked multi-head attention (with look ahead mask and padding mask)
        attention_out1, attn_weights1 = self.multihead_attention1(x, x, x, padding_mask)
        # attention_out.shape == (batch_size, target_seq_len, d_model)
        attention_out1 = self.dropout1(attention_out1, training=training)
        attn_norm_out1 = self.layernorm1(attention_out1 + x)
        # attn_nor_out.shape == (batch_size, target_seq_len, d_model)
        
        # Multi-head attention (with padding mask)
        attention_out2, attn_weights2 = self.multihead_attention2(attention_out1, 
                                                                  encoder_out,
                                                                  encoder_out,
                                                                  padding_mask)
        # attention_out2.shape == (batch_size, target_seq_len, d_model)
        attention_out2 = self.dropout2(attention_out2, training=training)
        attn_norm_out2 = self.layernorm2(attention_out2 + attn_norm_out1)
        # attn_nor_out2.shape == # (batch_size, target_seq_len, d_model)
        
        # Point wise feed forward networks
        ffn_out = self.ffn(attn_norm_out2)  # (Point wise feed forward networks)
        ffn_out = self.dropout3(ffn_out, training=training)
        ffn_norm_out = self.layernorm3(ffn_out + attn_norm_out2)
        
        return ffn_norm_out, attn_weights1, attn_weights2
```

*Decoder*由三部分组成：

- Output Embedding
- Positional Encoding
- N个*decoder layer*

```python
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, d_ff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(target_vocab_size, d_model)

        self.decoder_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, encoder_out, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        
        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x, block1, block2 = self.decoder_layers[i](x, encoder_out, training, 
                                        look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
            
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
```

# 8. Create the Transformer

```python
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, d_ff, 
                 input_vocab_size, target_vocab_size, rate=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, 
                               input_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff,
                               target_vocab_size, rate)
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, inputs, targets, training, encode_padding_mask,
             look_ahead_mask, decode_padding_mask):
        encoder_output = self.encoder(inputs, training, encode_padding_mask)
        # encoder_output.shape = (batch_size, inp_seq_len, d_model)
        
        decoder_output, attention_weights = self.decoder(
            targets, encoder_output, training, look_ahead_mask, decode_padding_mask
        )
        # decoder_output.shape = (batch_size, tar_seq_len, d_model)
        
        final_output = self.final_layer(decoder_output)
        # final_output.shape = (batch_size, tar_seq_len, target_vocab_size)
        
        return final_output, attention_weights
```

# 9. 实验

我们的实验还是将Transformer用于机器翻译——葡萄牙语翻译成英语。模型训练以后，我们输入葡萄牙语，模型返回英语。

## 9.1 优化器

论文中使用的优化器是*Adam*， 使用下式自定义学习率：
$$
l_{rate} = d_{model}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
$$

```python
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
```

```python
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)c
```

示例：

```python
temp_learning_rate_schedule = CustomSchedule(d_model)

plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
```

![png](https://www.tensorflow.org/beta/tutorials/text/transformer_files/output_f33ZCgvHpPdG_1.png)

## 9.2 Loss and Metrics

由于target sentence被padding了，因此计算损失的时候使用padding mask也是至关重要的：

```python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction="none"
)
```

```python
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_mean(loss_)
```

```python
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
```

## 9.3 模型超参数设置

为了保证模型较小，训练速度相对够快，实验过程中的超参数不会和论文保持一致， *num_layers*，*d_model*，*d_ff*都会有所减小：

```python
num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer_pt.vocab_size + 2
target_vocab_size = tokenizer_en.vocab_size + 2
dropout_rate = 0.1
```

## 9.4 数据pipeline

- 数据集

数据集使用[TFDS](https://www.tensorflow.org/datasets)从[TED Talks Open Translation Project](https://www.ted.com/participate/translate)中加载 [Portugese-English translation dataset](https://github.com/neulab/word-embeddings-for-nmt)。这个数据集包含大概5万训练数据，1100验证数据和2000测试数据。

```python
examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']
```

- Tokenizer

```python
tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (en.numpy() for pt, en in train_examples), target_vocab_size=2**13)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    (pt.numpy() for pt, en in train_examples), target_vocab_size=2**13)
```

示例：

```python
sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print ('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print ('The original string: {}'.format(original_string))

assert original_string == sample_string
```

> Tokenized string is [7915, 1248, 7946, 7194, 13, 2799, 7877]
> The original string: Transformer is awesome.

Tokenizer会将不在词表中的词拆分成子字符串：

```python
for ts in tokenized_string:
    print('{}--->{}'.format(ts, tokenizer_en.decode([ts])))
```

> 7915 ----> T
> 1248 ----> ran
> 7946 ----> s
> 7194 ----> former 
> 13 ----> is 
> 2799 ----> awesome
> 7877 ----> .

```python
BUFFER_SIZE = 20000
BATCH_SIZE = 64
```

- 向输入和输出中添加开始和结束符

```python
def encode(lang1, lang2):
    lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
             lang1.numpy()) + [tokenizer_pt.vocab_size+1]

    lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
             lang2.numpy()) + [tokenizer_en.vocab_size+1]
  
    return lang1, lang2
```

```python
def tf_encode(pt, en):
    return tf.py_function(encode, [pt, en], [tf.float64, tf.float64])
```

- 为了使模型不至于太大，且实验相对较快，我们过滤掉太长的句子

```python
MAX_LEN = 40
def filter_max_len(x, y, max_len=MAX_LEN):
    return tf.logical_and(tf.size(x) <= max_len, tf.size(y) <= max_len)
```

```python
train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]))
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]))
```

```python
pt_batch, en_batch = next(iter(val_dataset))
pt_batch, en_batch
```

> (<tf.Tensor: id=207697, shape=(64, 40), dtype=int64, numpy=
>  array([[8214, 1259,    5, ...,    0,    0,    0],
>         [8214,  299,   13, ...,    0,    0,    0],
>         [8214,   59,    8, ...,    0,    0,    0],
>         ...,
>         [8214,   95,    3, ...,    0,    0,    0],
>         [8214, 5157,    1, ...,    0,    0,    0],
>         [8214, 4479, 7990, ...,    0,    0,    0]])>,
>  <tf.Tensor: id=207698, shape=(64, 40), dtype=int64, numpy=
>  array([[8087,   18,   12, ...,    0,    0,    0],
>         [8087,  634,   30, ...,    0,    0,    0],
>         [8087,   16,   13, ...,    0,    0,    0],
>         ...,
>         [8087,   12,   20, ...,    0,    0,    0],
>         [8087,   17, 4981, ...,    0,    0,    0],
>         [8087,   12, 5453, ...,    0,    0,    0]])>)

## 9.5 Training and checkpointing

```python
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)
```

```python
def create_masks(inp, tar):
    # Encoder padding mask
    encode_padding_mask = create_padding_mask(inp)
  
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    decode_padding_mask = create_padding_mask(inp)
  
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    decode_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(decode_target_padding_mask, look_ahead_mask)
  
    return encode_padding_mask, combined_mask, decode_padding_mask
```

管理checkpoint，每N轮保存一次

```python
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')
```

target被分成两份：`tar_inp`和`tar_real`。其中`tar_inp`用于传入给`decoder`，`tar_real`是和输入一样的，只是向右移动一个位置，例如：

`sentence = "SOS A lion in the jungle is sleeping EOS"`

`tar_inp = "SOS A lion in the jungle is sleeping"`

`tar_real = "A lion in the jungle is sleeping EOS"`

```python
# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
  
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                     True, 
                                     enc_padding_mask, 
                                     combined_mask, 
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
    train_loss(loss)
    train_accuracy(tar_real, predictions)
```

```python
EPOCHS = 20

for epoch in range(EPOCHS):
    start = time.time()
  
    train_loss.reset_states()
    train_accuracy.reset_states()
  
    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)
    
        if batch % 50 == 0:
          print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                  epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    
    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
```

> W0814 01:06:36.753235 140098807473920 deprecation.py:323] From /tmpfs/src/tf_docs_env/lib/python3.5/site-packages/tensorflow_core/python/keras/optimizer_v2/optimizer_v2.py:455: BaseResourceVariable.constraint (from tensorflow.python.ops.resource_variable_ops) is deprecated and will be removed in a future version.
> Instructions for updating:
> Apply a constraint manually following the optimizer update step.
>
> Epoch 1 Batch 0 Loss 4.7365 Accuracy 0.0000
> Epoch 1 Batch 50 Loss 4.3028 Accuracy 0.0033
> Epoch 1 Batch 100 Loss 4.1992 Accuracy 0.0140
> Epoch 1 Batch 150 Loss 4.1569 Accuracy 0.0182
> Epoch 1 Batch 200 Loss 4.0963 Accuracy 0.0204
> Epoch 1 Batch 250 Loss 4.0199 Accuracy 0.0217
> Epoch 1 Batch 300 Loss 3.9262 Accuracy 0.0242
> Epoch 1 Batch 350 Loss 3.8337 Accuracy 0.0278
> Epoch 1 Batch 400 Loss 3.7477 Accuracy 0.0305
> Epoch 1 Batch 450 Loss 3.6682 Accuracy 0.0332
> Epoch 1 Batch 500 Loss 3.6032 Accuracy 0.0367
> Epoch 1 Batch 550 Loss 3.5408 Accuracy 0.0405
> Epoch 1 Batch 600 Loss 3.4777 Accuracy 0.0443
> Epoch 1 Batch 650 Loss 3.4197 Accuracy 0.0479
> Epoch 1 Batch 700 Loss 3.3672 Accuracy 0.0514
> Epoch 1 Loss 3.3650 Accuracy 0.0515
> Time taken for 1 epoch: 576.2345867156982 secs
>
> Epoch 2 Batch 0 Loss 2.4194 Accuracy 0.1030
> Epoch 2 Batch 50 Loss 2.5576 Accuracy 0.1030
> Epoch 2 Batch 100 Loss 2.5341 Accuracy 0.1051
> Epoch 2 Batch 150 Loss 2.5218 Accuracy 0.1076
> Epoch 2 Batch 200 Loss 2.4960 Accuracy 0.1095
> Epoch 2 Batch 250 Loss 2.4707 Accuracy 0.1115
> Epoch 2 Batch 300 Loss 2.4528 Accuracy 0.1133
> Epoch 2 Batch 350 Loss 2.4393 Accuracy 0.1150
> Epoch 2 Batch 400 Loss 2.4268 Accuracy 0.1165
> Epoch 2 Batch 450 Loss 2.4125 Accuracy 0.1182
> Epoch 2 Batch 500 Loss 2.4002 Accuracy 0.1196
> Epoch 2 Batch 550 Loss 2.3885 Accuracy 0.1209
> Epoch 2 Batch 600 Loss 2.3758 Accuracy 0.1222
> Epoch 2 Batch 650 Loss 2.3651 Accuracy 0.1235
> Epoch 2 Batch 700 Loss 2.3557 Accuracy 0.1247
> Epoch 2 Loss 2.3552 Accuracy 0.1247
> Time taken for 1 epoch: 341.75365233421326 secs
>
> Epoch 3 Batch 0 Loss 1.8798 Accuracy 0.1347
> Epoch 3 Batch 50 Loss 2.1781 Accuracy 0.1438
> Epoch 3 Batch 100 Loss 2.1810 Accuracy 0.1444
> Epoch 3 Batch 150 Loss 2.1796 Accuracy 0.1452
> Epoch 3 Batch 200 Loss 2.1759 Accuracy 0.1462
> Epoch 3 Batch 250 Loss 2.1710 Accuracy 0.1471
> Epoch 3 Batch 300 Loss 2.1625 Accuracy 0.1473
> Epoch 3 Batch 350 Loss 2.1520 Accuracy 0.1476
> Epoch 3 Batch 400 Loss 2.1411 Accuracy 0.1481
> Epoch 3 Batch 450 Loss 2.1306 Accuracy 0.1484
> Epoch 3 Batch 500 Loss 2.1276 Accuracy 0.1490
> Epoch 3 Batch 550 Loss 2.1231 Accuracy 0.1497
> Epoch 3 Batch 600 Loss 2.1143 Accuracy 0.1500
> Epoch 3 Batch 650 Loss 2.1063 Accuracy 0.1508
> Epoch 3 Batch 700 Loss 2.1034 Accuracy 0.1519
> Epoch 3 Loss 2.1036 Accuracy 0.1519
> Time taken for 1 epoch: 328.1187334060669 secs
>
> Epoch 4 Batch 0 Loss 2.0632 Accuracy 0.1622
> Epoch 4 Batch 50 Loss 1.9662 Accuracy 0.1642
> Epoch 4 Batch 100 Loss 1.9674 Accuracy 0.1656
> Epoch 4 Batch 150 Loss 1.9682 Accuracy 0.1667
> Epoch 4 Batch 200 Loss 1.9538 Accuracy 0.1679
> Epoch 4 Batch 250 Loss 1.9385 Accuracy 0.1683
> Epoch 4 Batch 300 Loss 1.9296 Accuracy 0.1694
> Epoch 4 Batch 350 Loss 1.9248 Accuracy 0.1705
> Epoch 4 Batch 400 Loss 1.9178 Accuracy 0.1716
> Epoch 4 Batch 450 Loss 1.9068 Accuracy 0.1724
> Epoch 4 Batch 500 Loss 1.8983 Accuracy 0.1735
> Epoch 4 Batch 550 Loss 1.8905 Accuracy 0.1745
> Epoch 4 Batch 600 Loss 1.8851 Accuracy 0.1757
> Epoch 4 Batch 650 Loss 1.8793 Accuracy 0.1768
> Epoch 4 Batch 700 Loss 1.8742 Accuracy 0.1779
> Epoch 4 Loss 1.8746 Accuracy 0.1780
> Time taken for 1 epoch: 326.3032810688019 secs
>
> Epoch 5 Batch 0 Loss 1.9596 Accuracy 0.1979
> Epoch 5 Batch 50 Loss 1.7048 Accuracy 0.1961
> Epoch 5 Batch 100 Loss 1.6949 Accuracy 0.1969
> Epoch 5 Batch 150 Loss 1.6942 Accuracy 0.1986
> Epoch 5 Batch 200 Loss 1.6876 Accuracy 0.1992
> Epoch 5 Batch 250 Loss 1.6827 Accuracy 0.1994
> Epoch 5 Batch 300 Loss 1.6776 Accuracy 0.2006
> Epoch 5 Batch 350 Loss 1.6740 Accuracy 0.2013
> Epoch 5 Batch 400 Loss 1.6706 Accuracy 0.2019
> Epoch 5 Batch 450 Loss 1.6656 Accuracy 0.2028
> Epoch 5 Batch 500 Loss 1.6599 Accuracy 0.2035
> Epoch 5 Batch 550 Loss 1.6558 Accuracy 0.2040
> Epoch 5 Batch 600 Loss 1.6519 Accuracy 0.2047
> Epoch 5 Batch 650 Loss 1.6510 Accuracy 0.2053
> Epoch 5 Batch 700 Loss 1.6453 Accuracy 0.2058
> Saving checkpoint for epoch 5 at ./checkpoints/train/ckpt-1
> Epoch 5 Loss 1.6453 Accuracy 0.2058
> Time taken for 1 epoch: 307.13636589050293 secs
>
> Epoch 6 Batch 0 Loss 1.5280 Accuracy 0.2127
> Epoch 6 Batch 50 Loss 1.5062 Accuracy 0.2214
> Epoch 6 Batch 100 Loss 1.5121 Accuracy 0.2225
> Epoch 6 Batch 150 Loss 1.5051 Accuracy 0.2216
> Epoch 6 Batch 200 Loss 1.5014 Accuracy 0.2219
> Epoch 6 Batch 250 Loss 1.4984 Accuracy 0.2222
> Epoch 6 Batch 300 Loss 1.4966 Accuracy 0.2232
> Epoch 6 Batch 350 Loss 1.4929 Accuracy 0.2231
> Epoch 6 Batch 400 Loss 1.4900 Accuracy 0.2234
> Epoch 6 Batch 450 Loss 1.4836 Accuracy 0.2237
> Epoch 6 Batch 500 Loss 1.4792 Accuracy 0.2241
> Epoch 6 Batch 550 Loss 1.4727 Accuracy 0.2245
> Epoch 6 Batch 600 Loss 1.4695 Accuracy 0.2251
> Epoch 6 Batch 650 Loss 1.4659 Accuracy 0.2256
> Epoch 6 Batch 700 Loss 1.4625 Accuracy 0.2262
> Epoch 6 Loss 1.4619 Accuracy 0.2262
> Time taken for 1 epoch: 303.32839941978455 secs
>
> Epoch 7 Batch 0 Loss 1.1667 Accuracy 0.2262
> Epoch 7 Batch 50 Loss 1.3010 Accuracy 0.2407
> Epoch 7 Batch 100 Loss 1.3009 Accuracy 0.2400
> Epoch 7 Batch 150 Loss 1.2983 Accuracy 0.2414
> Epoch 7 Batch 200 Loss 1.2959 Accuracy 0.2428
> Epoch 7 Batch 250 Loss 1.2948 Accuracy 0.2436
> Epoch 7 Batch 300 Loss 1.2928 Accuracy 0.2439
> Epoch 7 Batch 350 Loss 1.2901 Accuracy 0.2442
> Epoch 7 Batch 400 Loss 1.2831 Accuracy 0.2448
> Epoch 7 Batch 450 Loss 1.2844 Accuracy 0.2458
> Epoch 7 Batch 500 Loss 1.2832 Accuracy 0.2463
> Epoch 7 Batch 550 Loss 1.2827 Accuracy 0.2469
> Epoch 7 Batch 600 Loss 1.2786 Accuracy 0.2470
> Epoch 7 Batch 650 Loss 1.2738 Accuracy 0.2473
> Epoch 7 Batch 700 Loss 1.2737 Accuracy 0.2480
> Epoch 7 Loss 1.2737 Accuracy 0.2480
> Time taken for 1 epoch: 314.8111472129822 secs
>
> Epoch 8 Batch 0 Loss 1.1562 Accuracy 0.2611
> Epoch 8 Batch 50 Loss 1.1305 Accuracy 0.2637
> Epoch 8 Batch 100 Loss 1.1262 Accuracy 0.2644
> Epoch 8 Batch 150 Loss 1.1193 Accuracy 0.2639
> Epoch 8 Batch 200 Loss 1.1210 Accuracy 0.2645
> Epoch 8 Batch 250 Loss 1.1177 Accuracy 0.2651
> Epoch 8 Batch 300 Loss 1.1182 Accuracy 0.2648
> Epoch 8 Batch 350 Loss 1.1200 Accuracy 0.2653
> Epoch 8 Batch 400 Loss 1.1212 Accuracy 0.2655
> Epoch 8 Batch 450 Loss 1.1207 Accuracy 0.2653
> Epoch 8 Batch 500 Loss 1.1222 Accuracy 0.2660
> Epoch 8 Batch 550 Loss 1.1219 Accuracy 0.2664
> Epoch 8 Batch 600 Loss 1.1229 Accuracy 0.2663
> Epoch 8 Batch 650 Loss 1.1211 Accuracy 0.2664
> Epoch 8 Batch 700 Loss 1.1206 Accuracy 0.2668
> Epoch 8 Loss 1.1207 Accuracy 0.2668
> Time taken for 1 epoch: 301.5652780532837 secs
>
> Epoch 9 Batch 0 Loss 0.8384 Accuracy 0.2751
> Epoch 9 Batch 50 Loss 0.9923 Accuracy 0.2793
> Epoch 9 Batch 100 Loss 0.9958 Accuracy 0.2796
> Epoch 9 Batch 150 Loss 0.9953 Accuracy 0.2787
> Epoch 9 Batch 200 Loss 0.9937 Accuracy 0.2790
> Epoch 9 Batch 250 Loss 0.9988 Accuracy 0.2800
> Epoch 9 Batch 300 Loss 0.9999 Accuracy 0.2801
> Epoch 9 Batch 350 Loss 1.0021 Accuracy 0.2800
> Epoch 9 Batch 400 Loss 1.0001 Accuracy 0.2800
> Epoch 9 Batch 450 Loss 1.0013 Accuracy 0.2800
> Epoch 9 Batch 500 Loss 1.0027 Accuracy 0.2805
> Epoch 9 Batch 550 Loss 1.0034 Accuracy 0.2804
> Epoch 9 Batch 600 Loss 1.0071 Accuracy 0.2810
> Epoch 9 Batch 650 Loss 1.0076 Accuracy 0.2810
> Epoch 9 Batch 700 Loss 1.0075 Accuracy 0.2806
> Epoch 9 Loss 1.0076 Accuracy 0.2806
> Time taken for 1 epoch: 304.53144931793213 secs
>
> Epoch 10 Batch 0 Loss 0.9130 Accuracy 0.3057
> Epoch 10 Batch 50 Loss 0.8950 Accuracy 0.2966
> Epoch 10 Batch 100 Loss 0.9066 Accuracy 0.2967
> Epoch 10 Batch 150 Loss 0.9128 Accuracy 0.2958
> Epoch 10 Batch 200 Loss 0.9099 Accuracy 0.2943
> Epoch 10 Batch 250 Loss 0.9131 Accuracy 0.2935
> Epoch 10 Batch 300 Loss 0.9155 Accuracy 0.2930
> Epoch 10 Batch 350 Loss 0.9144 Accuracy 0.2922
> Epoch 10 Batch 400 Loss 0.9148 Accuracy 0.2922
> Epoch 10 Batch 450 Loss 0.9170 Accuracy 0.2916
> Epoch 10 Batch 500 Loss 0.9164 Accuracy 0.2910
> Epoch 10 Batch 550 Loss 0.9175 Accuracy 0.2908
> Epoch 10 Batch 600 Loss 0.9193 Accuracy 0.2908
> Epoch 10 Batch 650 Loss 0.9229 Accuracy 0.2907
> Epoch 10 Batch 700 Loss 0.9245 Accuracy 0.2910
> Saving checkpoint for epoch 10 at ./checkpoints/train/ckpt-2
> Epoch 10 Loss 0.9247 Accuracy 0.2910
> Time taken for 1 epoch: 308.50231170654297 secs
>
> Epoch 11 Batch 0 Loss 0.8796 Accuracy 0.3030
> Epoch 11 Batch 50 Loss 0.8186 Accuracy 0.3025
> Epoch 11 Batch 100 Loss 0.8268 Accuracy 0.3020
> Epoch 11 Batch 150 Loss 0.8422 Accuracy 0.3026
> Epoch 11 Batch 200 Loss 0.8453 Accuracy 0.3023
> Epoch 11 Batch 250 Loss 0.8472 Accuracy 0.3020
> Epoch 11 Batch 300 Loss 0.8478 Accuracy 0.3019
> Epoch 11 Batch 350 Loss 0.8488 Accuracy 0.3018
> Epoch 11 Batch 400 Loss 0.8509 Accuracy 0.3017
> Epoch 11 Batch 450 Loss 0.8505 Accuracy 0.3012
> Epoch 11 Batch 500 Loss 0.8505 Accuracy 0.3009
> Epoch 11 Batch 550 Loss 0.8514 Accuracy 0.3005
> Epoch 11 Batch 600 Loss 0.8541 Accuracy 0.3001
> Epoch 11 Batch 650 Loss 0.8568 Accuracy 0.2998
> Epoch 11 Batch 700 Loss 0.8581 Accuracy 0.2995
> Epoch 11 Loss 0.8586 Accuracy 0.2996
> Time taken for 1 epoch: 326.4959843158722 secs
>
> Epoch 12 Batch 0 Loss 0.8353 Accuracy 0.3318
> Epoch 12 Batch 50 Loss 0.7892 Accuracy 0.3161
> Epoch 12 Batch 100 Loss 0.7778 Accuracy 0.3134
> Epoch 12 Batch 150 Loss 0.7817 Accuracy 0.3132
> Epoch 12 Batch 200 Loss 0.7845 Accuracy 0.3132
> Epoch 12 Batch 250 Loss 0.7881 Accuracy 0.3124
> Epoch 12 Batch 300 Loss 0.7903 Accuracy 0.3122
> Epoch 12 Batch 350 Loss 0.7894 Accuracy 0.3107
> Epoch 12 Batch 400 Loss 0.7889 Accuracy 0.3097
> Epoch 12 Batch 450 Loss 0.7917 Accuracy 0.3089
> Epoch 12 Batch 500 Loss 0.7947 Accuracy 0.3089
> Epoch 12 Batch 550 Loss 0.7965 Accuracy 0.3087
> Epoch 12 Batch 600 Loss 0.7990 Accuracy 0.3082
> Epoch 12 Batch 650 Loss 0.8002 Accuracy 0.3077
> Epoch 12 Batch 700 Loss 0.8026 Accuracy 0.3076
> Epoch 12 Loss 0.8028 Accuracy 0.3076
> Time taken for 1 epoch: 306.4404299259186 secs
>
> Epoch 13 Batch 0 Loss 0.7718 Accuracy 0.3059
> Epoch 13 Batch 50 Loss 0.7275 Accuracy 0.3206
> Epoch 13 Batch 100 Loss 0.7308 Accuracy 0.3206
> Epoch 13 Batch 150 Loss 0.7317 Accuracy 0.3186
> Epoch 13 Batch 200 Loss 0.7342 Accuracy 0.3174
> Epoch 13 Batch 250 Loss 0.7349 Accuracy 0.3171
> Epoch 13 Batch 300 Loss 0.7374 Accuracy 0.3167
> Epoch 13 Batch 350 Loss 0.7397 Accuracy 0.3166
> Epoch 13 Batch 400 Loss 0.7410 Accuracy 0.3163
> Epoch 13 Batch 450 Loss 0.7415 Accuracy 0.3154
> Epoch 13 Batch 500 Loss 0.7434 Accuracy 0.3150
> Epoch 13 Batch 550 Loss 0.7466 Accuracy 0.3148
> Epoch 13 Batch 600 Loss 0.7490 Accuracy 0.3142
> Epoch 13 Batch 650 Loss 0.7522 Accuracy 0.3142
> Epoch 13 Batch 700 Loss 0.7552 Accuracy 0.3142
> Epoch 13 Loss 0.7554 Accuracy 0.3142
> Time taken for 1 epoch: 299.16382122039795 secs
>
> Epoch 14 Batch 0 Loss 0.6654 Accuracy 0.3193
> Epoch 14 Batch 50 Loss 0.6744 Accuracy 0.3277
> Epoch 14 Batch 100 Loss 0.6809 Accuracy 0.3237
> Epoch 14 Batch 150 Loss 0.6830 Accuracy 0.3238
> Epoch 14 Batch 200 Loss 0.6875 Accuracy 0.3235
> Epoch 14 Batch 250 Loss 0.6942 Accuracy 0.3238
> Epoch 14 Batch 300 Loss 0.6976 Accuracy 0.3231
> Epoch 14 Batch 350 Loss 0.7000 Accuracy 0.3230
> Epoch 14 Batch 400 Loss 0.7019 Accuracy 0.3222
> Epoch 14 Batch 450 Loss 0.7035 Accuracy 0.3212
> Epoch 14 Batch 500 Loss 0.7077 Accuracy 0.3207
> Epoch 14 Batch 550 Loss 0.7078 Accuracy 0.3201
> Epoch 14 Batch 600 Loss 0.7095 Accuracy 0.3196
> Epoch 14 Batch 650 Loss 0.7127 Accuracy 0.3197
> Epoch 14 Batch 700 Loss 0.7148 Accuracy 0.3193
> Epoch 14 Loss 0.7153 Accuracy 0.3194
> Time taken for 1 epoch: 294.01167726516724 secs
>
> Epoch 15 Batch 0 Loss 0.6159 Accuracy 0.3546
> Epoch 15 Batch 50 Loss 0.6416 Accuracy 0.3339
> Epoch 15 Batch 100 Loss 0.6477 Accuracy 0.3323
> Epoch 15 Batch 150 Loss 0.6480 Accuracy 0.3300
> Epoch 15 Batch 200 Loss 0.6518 Accuracy 0.3286
> Epoch 15 Batch 250 Loss 0.6536 Accuracy 0.3283
> Epoch 15 Batch 300 Loss 0.6576 Accuracy 0.3276
> Epoch 15 Batch 350 Loss 0.6618 Accuracy 0.3274
> Epoch 15 Batch 400 Loss 0.6657 Accuracy 0.3272
> Epoch 15 Batch 450 Loss 0.6689 Accuracy 0.3269
> Epoch 15 Batch 500 Loss 0.6693 Accuracy 0.3263
> Epoch 15 Batch 550 Loss 0.6711 Accuracy 0.3255
> Epoch 15 Batch 600 Loss 0.6740 Accuracy 0.3249
> Epoch 15 Batch 650 Loss 0.6775 Accuracy 0.3250
> Epoch 15 Batch 700 Loss 0.6796 Accuracy 0.3247
> Saving checkpoint for epoch 15 at ./checkpoints/train/ckpt-3
> Epoch 15 Loss 0.6800 Accuracy 0.3247
> Time taken for 1 epoch: 296.7416775226593 secs
>
> Epoch 16 Batch 0 Loss 0.6764 Accuracy 0.3298
> Epoch 16 Batch 50 Loss 0.6024 Accuracy 0.3335
> Epoch 16 Batch 100 Loss 0.6089 Accuracy 0.3345
> Epoch 16 Batch 150 Loss 0.6135 Accuracy 0.3315
> Epoch 16 Batch 200 Loss 0.6191 Accuracy 0.3323
> Epoch 16 Batch 250 Loss 0.6214 Accuracy 0.3324
> Epoch 16 Batch 300 Loss 0.6230 Accuracy 0.3315
> Epoch 16 Batch 350 Loss 0.6268 Accuracy 0.3313
> Epoch 16 Batch 400 Loss 0.6294 Accuracy 0.3309
> Epoch 16 Batch 450 Loss 0.6325 Accuracy 0.3306
> Epoch 16 Batch 500 Loss 0.6350 Accuracy 0.3300
> Epoch 16 Batch 550 Loss 0.6385 Accuracy 0.3298
> Epoch 16 Batch 600 Loss 0.6405 Accuracy 0.3293
> Epoch 16 Batch 650 Loss 0.6434 Accuracy 0.3291
> Epoch 16 Batch 700 Loss 0.6472 Accuracy 0.3289
> Epoch 16 Loss 0.6476 Accuracy 0.3290
> Time taken for 1 epoch: 302.5653040409088 secs
>
> Epoch 17 Batch 0 Loss 0.7453 Accuracy 0.3696
> Epoch 17 Batch 50 Loss 0.5800 Accuracy 0.3427
> Epoch 17 Batch 100 Loss 0.5841 Accuracy 0.3422
> Epoch 17 Batch 150 Loss 0.5912 Accuracy 0.3409
> Epoch 17 Batch 200 Loss 0.5911 Accuracy 0.3384
> Epoch 17 Batch 250 Loss 0.5962 Accuracy 0.3389
> Epoch 17 Batch 300 Loss 0.5997 Accuracy 0.3389
> Epoch 17 Batch 350 Loss 0.6017 Accuracy 0.3383
> Epoch 17 Batch 400 Loss 0.6042 Accuracy 0.3376
> Epoch 17 Batch 450 Loss 0.6077 Accuracy 0.3375
> Epoch 17 Batch 500 Loss 0.6106 Accuracy 0.3369
> Epoch 17 Batch 550 Loss 0.6127 Accuracy 0.3361
> Epoch 17 Batch 600 Loss 0.6148 Accuracy 0.3352
> Epoch 17 Batch 650 Loss 0.6171 Accuracy 0.3346
> Epoch 17 Batch 700 Loss 0.6195 Accuracy 0.3339
> Epoch 17 Loss 0.6196 Accuracy 0.3339
> Time taken for 1 epoch: 303.3943374156952 secs
>
> Epoch 18 Batch 0 Loss 0.4733 Accuracy 0.3313
> Epoch 18 Batch 50 Loss 0.5544 Accuracy 0.3395
> Epoch 18 Batch 100 Loss 0.5637 Accuracy 0.3435
> Epoch 18 Batch 150 Loss 0.5625 Accuracy 0.3421
> Epoch 18 Batch 200 Loss 0.5686 Accuracy 0.3421
> Epoch 18 Batch 250 Loss 0.5714 Accuracy 0.3413
> Epoch 18 Batch 300 Loss 0.5727 Accuracy 0.3407
> Epoch 18 Batch 350 Loss 0.5770 Accuracy 0.3406
> Epoch 18 Batch 400 Loss 0.5759 Accuracy 0.3394
> Epoch 18 Batch 450 Loss 0.5779 Accuracy 0.3390
> Epoch 18 Batch 500 Loss 0.5810 Accuracy 0.3392
> Epoch 18 Batch 550 Loss 0.5836 Accuracy 0.3388
> Epoch 18 Batch 600 Loss 0.5870 Accuracy 0.3379
> Epoch 18 Batch 650 Loss 0.5905 Accuracy 0.3378
> Epoch 18 Batch 700 Loss 0.5945 Accuracy 0.3376
> Epoch 18 Loss 0.5947 Accuracy 0.3376
> Time taken for 1 epoch: 298.2541983127594 secs
>
> Epoch 19 Batch 0 Loss 0.5082 Accuracy 0.3261
> Epoch 19 Batch 50 Loss 0.5285 Accuracy 0.3451
> Epoch 19 Batch 100 Loss 0.5336 Accuracy 0.3472
> Epoch 19 Batch 150 Loss 0.5322 Accuracy 0.3440
> Epoch 19 Batch 200 Loss 0.5355 Accuracy 0.3439
> Epoch 19 Batch 250 Loss 0.5413 Accuracy 0.3441
> Epoch 19 Batch 300 Loss 0.5461 Accuracy 0.3443
> Epoch 19 Batch 350 Loss 0.5519 Accuracy 0.3441
> Epoch 19 Batch 400 Loss 0.5548 Accuracy 0.3436
> Epoch 19 Batch 450 Loss 0.5561 Accuracy 0.3427
> Epoch 19 Batch 500 Loss 0.5595 Accuracy 0.3423
> Epoch 19 Batch 550 Loss 0.5616 Accuracy 0.3416
> Epoch 19 Batch 600 Loss 0.5658 Accuracy 0.3412
> Epoch 19 Batch 650 Loss 0.5684 Accuracy 0.3407
> Epoch 19 Batch 700 Loss 0.5707 Accuracy 0.3405
> Epoch 19 Loss 0.5709 Accuracy 0.3406
> Time taken for 1 epoch: 297.59109830856323 secs
>
> Epoch 20 Batch 0 Loss 0.6551 Accuracy 0.3720
> Epoch 20 Batch 50 Loss 0.5086 Accuracy 0.3527
> Epoch 20 Batch 100 Loss 0.5160 Accuracy 0.3495
> Epoch 20 Batch 150 Loss 0.5196 Accuracy 0.3495
> Epoch 20 Batch 200 Loss 0.5210 Accuracy 0.3490
> Epoch 20 Batch 250 Loss 0.5241 Accuracy 0.3487
> Epoch 20 Batch 300 Loss 0.5287 Accuracy 0.3486
> Epoch 20 Batch 350 Loss 0.5312 Accuracy 0.3477
> Epoch 20 Batch 400 Loss 0.5337 Accuracy 0.3475
> Epoch 20 Batch 450 Loss 0.5369 Accuracy 0.3469
> Epoch 20 Batch 500 Loss 0.5377 Accuracy 0.3458
> Epoch 20 Batch 550 Loss 0.5400 Accuracy 0.3453
> Epoch 20 Batch 600 Loss 0.5441 Accuracy 0.3450
> Epoch 20 Batch 650 Loss 0.5469 Accuracy 0.3445
> Epoch 20 Batch 700 Loss 0.5507 Accuracy 0.3440
> Saving checkpoint for epoch 20 at ./checkpoints/train/ckpt-4
> Epoch 20 Loss 0.5507 Accuracy 0.3440
> Time taken for 1 epoch: 303.6011939048767 secs

## 9.6 评估

评估过程包含以下步骤：

- 使用`Portuguese tokenizer`对输入语句进行编码
- 解码输入`start token == tokenizer_en.vocab_size`
- 计算`padding_mask`和`look_ahead_mask`
- `decoder`输出预测结果
- 选择最后一个词，并且计算它的`argmax`
- 将之前输出的词拼接起来，作为`deocder`的输入，用于预测后面的词
- 最后的到最终的预测结果

> 这个评估过程非常重要，实际上这也是模型训练好以后，我们使用模型进行翻译的过程。我们可以看到这个过程是一步一步进行的，专业术语叫做*Auto-Regression*。虽然transformer的训练很快，但是推理却很慢，主要原因就是它做的是*Auto-regression*，不能进行并行化推理，所以后续很多对transformer的改进工作都是在这上面做的改进，我会在后续的博客中详细介绍相关模型。

```python
def evaluate(inp_sentence):
    start_token = [tokenizer_pt.vocab_size]
    end_token = [tokenizer_pt.vocab_size + 1]
  
    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + tokenizer_pt.encode(inp_sentence) + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)
  
    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)
  
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, 
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)
    
        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
      
        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer_en.vocab_size+1:
            return tf.squeeze(output, axis=0), attention_weights
    
        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights
```

```python
def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))
  
    sentence = tokenizer_pt.encode(sentence)
  
    attention = tf.squeeze(attention[layer], axis=0)
  
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)
    
        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')

        fontdict = {'fontsize': 10}
    
        ax.set_xticks(range(len(sentence)+2))
        ax.set_yticks(range(len(result)))
    
        ax.set_ylim(len(result)-1.5, -0.5)
        
        ax.set_xticklabels(
            ['<start>']+[tokenizer_pt.decode([i]) for i in sentence]+['<end>'], 
            fontdict=fontdict, rotation=90)
    
        ax.set_yticklabels([tokenizer_en.decode([i]) for i in result 
                           if i < tokenizer_en.vocab_size], 
                           fontdict=fontdict)
    
        ax.set_xlabel('Head {}'.format(head+1))
  
    plt.tight_layout()
    plt.show()
```

```python
def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)
  
    predicted_sentence = tokenizer_en.decode([i for i in result 
                                              if i < tokenizer_en.vocab_size])  

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))
  
    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)
```

> translate("este é um problema que temos que resolver.")
> print ("Real translation: this is a problem we have to solve .")

![png](https://www.tensorflow.org/beta/tutorials/text/transformer_files/output_t-kFyiOLH0xg_1.png)

# 10. 参考资料

[Transformer model for language understanding](https://www.tensorflow.org/beta/tutorials/text/transformer)

