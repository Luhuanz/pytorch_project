# Transformer_usePytorch

模型调用流程（每个文件都可以测试，该测试即为demo）：
**1. 使用Source.py读取数据集；
2. 使用TokenTensorizer.py将token转换为embedding；
3. 使用BatchLoader.py将数据集载入；**
4. 使用Model.py中的transformer模型；
5. 使用Trainer.py中的训练器训练和测试模型。

---------------------

BERT和BERT_LM已经可以使用，因为此前的transformer和BERT中transformer不一致，因此准备了transformerBlock

-------------------
transformer-xl已经可以使用，模块还未整合。。。