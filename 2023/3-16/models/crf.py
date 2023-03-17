import math
from torch.autograd import Variable
import torch
import torch.nn as nn
from typing import List, Optional


class CRF(nn.Module):
#CRF 是一种常用于序列标注任务（例如词性标注或命名实体识别）的概率模型。
# 这个模块中的 CRF 类可以计算给定标记序列和发射分数张量的对数似然，并使用维特比算法寻找给定发射分数张量的最佳标记序列。

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0: #标记的数量，必须为正整数，否则会引发 ValueError 异常。
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first #一个布尔值，表示是否将批次维度放在第一维。默认值为 False。
        self.start_transitions = nn.Parameter(torch.empty(num_tags)) #表示起始转移分数张量
        self.end_transitions = nn.Parameter(torch.empty(num_tags))#表示结束转移分数张量
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))#表示状态转移分数张量。

        self.reset_parameters() #用于初始化模型的转移参数。

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
#该方法通过 __class__.__name__ 获取当前对象的类名，然后返回类似于 <类名>(num_tags=<num_tags>) 的字符串，
# 其中 <num_tags> 表示模型的标记数目。这个字符串可以用于调试和输出模型信息，方便用户查看模型的属性。
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(self, emissions: torch.Tensor,
                tags: torch.LongTensor,
                mask: Optional[torch.ByteTensor] = None,
                reduction: str = 'mean') -> torch.Tensor:
#emissions 是形状为 (seq_length, batch_size, num_tags) 的张量，表示模型对每个标记的打分（或者说概率）
#tags 是形状为 (seq_length, batch_size) 的张量，表示真实的标记序列。
#mask 是形状为 (seq_length, batch_size) 的张量，用于指示哪些位置需要参与计算（值为 1），哪些位置不需要（值为 0）。
#reduction 参数用于控制如何计算损失值

# 流程
# 1._validate() 方法检查输入的张量和掩码是否符合要求。
#2.使用 forward_alg() 方法计算给定标记序列的条件概率对数值（也就是前向算法的结果）。
#3._score_sentence() 方法计算给定标记序列的得分（也就是发射得分加上转移得分）。
#4.根据 reduction 参数的值来决定是否进行损失的缩减，返回计算得到的损失值。
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
     #用于检查输入的张量和掩码是否符合要求。这个方法的参数包括 emissions、tags 和 mask
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=tags.device) #全1
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, tags=tags, mask=mask)
# 转置维度
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
# _compute_score方法，计算序列标注模型的分子部分得分。这个方法会将发射分数和标签序列作为输入，并计算标签序列的得分。
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
#在这里，CRF计算了给定标签序列的对数似然，使用前向算法计算序列标注的概率的对数似然，并减去对数归一化常数的对数值，得到了标签序列的对数似然值。
# 这个值可以用来评估给定标签序列的好坏。其中，numerator是给定标签序列的得分，denominator是所有可能的标签序列的得分之和的对数值。

    # 实现了 CRF 的前向计算
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator
##4.根据 reduction 参数的值来决定是否进行损失的缩减，返回计算得到的损失值。
        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None,
               nbest: Optional[int] = None,
               pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            nbest (`int`): Number of most probable paths for each sequence
            pad_tag (`int`): Tag at padded positions. Often input varies in length and
                the length will be padded to the maximum length in the batch. Tags at
                the padded positions will be assigned with a padding tag, i.e. `pad_tag`
        Returns:
            A PyTorch tensor of the best tag sequence for each batch of shape
            (nbest, batch_size, seq_length)
        """
        if nbest is None:
            nbest = 1
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.uint8,
                              device=emissions.device)
        if mask.dtype != torch.uint8:
            mask = mask.byte()
        self._validate(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(self, emissions: torch.Tensor,
                  tags: Optional[torch.LongTensor] = None,
                  mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(self, emissions: torch.Tensor,
                       tags: torch.LongTensor,
                       mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(self, emissions: torch.Tensor,
                            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor,
                        pad_tag: Optional[int] = None) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size), pad_tag,
                             dtype=torch.long, device=device)

        # - score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # - history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score
        # shape: (batch_size, num_tags)
        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
                             end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size),
                                    dtype=torch.long, device=device)
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size)

        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(self, emissions: torch.FloatTensor,
                              mask: torch.ByteTensor,
                              nbest: int,
                              pad_tag: Optional[int] = None) -> List[List[List[int]]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (nbest, batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros((seq_length, batch_size, self.num_tags, nbest),
                                  dtype=torch.long, device=device)
        oor_idx = torch.zeros((batch_size, self.num_tags, nbest),
                              dtype=torch.long, device=device)
        oor_tag = torch.full((seq_length, batch_size, nbest), pad_tag,
                             dtype=torch.long, device=device)

        # + score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # + history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            if i == 1:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1)
                # shape: (batch_size, num_tags, num_tags)
                next_score = broadcast_score + self.transitions + broadcast_emission
            else:
                broadcast_score = score.unsqueeze(-1)
                broadcast_emission = emissions[i].unsqueeze(1).unsqueeze(2)
                # shape: (batch_size, num_tags, nbest, num_tags)
                next_score = broadcast_score + self.transitions.unsqueeze(1) + broadcast_emission

            # Find the top `nbest` maximum score over all possible current tag
            # shape: (batch_size, nbest, num_tags)
            next_score, indices = next_score.view(batch_size, -1, self.num_tags).topk(nbest, dim=1)

            if i == 1:
                score = score.unsqueeze(-1).expand(-1, -1, nbest)
                indices = indices * nbest

            # convert to shape: (batch_size, num_tags, nbest)
            next_score = next_score.transpose(2, 1)
            indices = indices.transpose(2, 1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags, nbest)
            score = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1).unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score shape: (batch_size, num_tags, nbest)
        end_score = score + self.end_transitions.unsqueeze(-1)
        _, end_tag = end_score.view(batch_size, -1).topk(nbest, dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(1, seq_ends.view(-1, 1, 1, 1).expand(-1, 1, self.num_tags, nbest),
                             end_tag.view(-1, 1, 1, nbest).expand(-1, 1, self.num_tags, nbest))
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros((seq_length, batch_size, nbest),
                                    dtype=torch.long, device=device)
        best_tags = torch.arange(nbest, dtype=torch.long, device=device) \
                         .view(1, -1).expand(batch_size, -1)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx].view(batch_size, -1), 1, best_tags)
            best_tags_arr[idx] = best_tags.data.view(batch_size, -1) // nbest

        return torch.where(mask.unsqueeze(-1), best_tags_arr, oor_tag).permute(2, 1, 0)