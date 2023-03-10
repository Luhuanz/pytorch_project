#!/usr/bin/python
# -*- coding: UTF-8 -*-
#微信公众号 AI壹号堂 欢迎关注
#Author bruce
import numpy as np
embedding = np.load("embedding_SougouNews.npz")
print(embedding.embeddings[0])