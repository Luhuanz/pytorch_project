# PinSAGE
pinsage for wine recommendation
제 11회 투빅스 컨퍼런스 '투믈리에'조에서 준비한 와인 추천 시스템에 적용한 PinSAGE 패키지입니다
DGL 라이브러리를 바탕으로 구현했으며, PinSAGE 예제에서 프로젝트에 맞게 수정하였습니다.

PinSAGE paper: https://arxiv.org/pdf/1806.01973.pdf <br>
DGL: https://docs.dgl.ai/# <br>
DGL PinSAGE example: https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage <br>

## Requirements

- dgl
- dask
- pandas
- torch
- torchtext
- sklearn

## Dataset

### Vivino 
11,900,000 Wines & 42,000,000 Users
User feature: userID, user_follower_count, user_rating_count
Item feature: wine_id, body, acidity, alcohol, rating_average, grapes_id




**!!! Update 2021/11/16 !!!**
데이터 공유 요청이 있어서 일부분 사용하실 수 있도록 제공해드립니다.
* 리뷰 데이터 10만개 
* 유저 메타 데이터
* 와인 메타 데이터

전체 데이터가 아닌만큼 직접 학습하실 때 성능이 원하는만큼 나오지 않을 수 있습니다.
**process_wine.py**가 수집된 데이터를 DGL에 맞게 전처리하는 코드입니다 제공된 데이터를 사용하신다면 참고해주세요


## Run model

### Nearest-neighbor recommendation

이 모델은 모든 유저마다 K nearest neighbors로 와인을 추천합니다.
특정 유저가 소비했던 와인의 임베딩 벡터의 중심을 구하고, 중심 벡터로부터 가장 가까운 K개의 와인을 추천하는 방식입니다.

```
python model.py -d data.pkl -s model -k 500 --eval-epochs 100 --save-epochs 100 --num-epochs 500 --device 0 --hidden-dims 128 --batch-size 64 --batches-per-epoch 512
```

- d: 데이터 파일
- s: 저장될 모델 이름
- k: top K 개수
- eval epochs: 성능 출력 epoch 간격(0 = 출력 X)
- save epochs: 저장 epoch 간격(0 = 저장 X)
- num epochs: epoch 횟수
- hidden dims: 임베딩 차원
- batch size: 배치 크기
- batches per epoch: iteration 횟수

이 외에 PinSAGE에서 적용하는 파라미터가 있으니 model.py 코드를 참조바랍니다.

## Inference
하단의 코드는 추론 방법을 설명하는 코드로, model.py의 train 함수 부분을 발췌하여 설명하겠습니다
본 프로젝트의 성능 평가 방식은 기존 DGL PinSAGE에서 한 가지 아이템만 추천하는 방식과는 차이가 있습니다.

### Embeddings of all items
모델은 노드의 임베딩을 학습하는 데에 목적을 두고 있기 때문에, 모든 아이템의 임베딩을 얻은 후 벡터 간 연산을 통해서 유사도 측정 또는 군집화를 따로 진행해야 합니다.

```
model.py line 159

h_item = evaluation.get_all_emb(gnn, g.ndata['id'][item_ntype], data_dict['testset'], item_ntype, neighbor_sampler, args.batch_size, device)
```
DGL graph객체에서 노드 정보를 받아 모든 임베딩을 얻어옵니다. shape은 (유저 수, 임베딩 크기)가 됩니다.

```
model.py line 182~
h_center = torch.mean(h_nodes, axis=0)  # 중앙 임베딩  
dist = h_center @ h_item.t()  # 센터 임베딩 * 모든 임베딩 -> 행렬곱
topk = dist.topk(args.k)[1].cpu().numpy()  # dist 크기 순서로 k개 추출
```
추론에 쓰일 특정 유저의 노드 임베딩을 평균내어 중앙 임베딩 벡터를 얻고, 모든 임베딩과 행렬곱 연산으로 Distance를 얻습니다.
Distance가 작은 순으로 K개 만큼의 임베딩을 추출하여 최종 추천 항목으로 제시합니다.

뽑힌 아이템들이 검증용 데이터에 속해있는지에 대한 여부를 Recall과 Hitrate로 평가합니다.

## Performance

Model | Hitrate | Recall
------------ | ------------- | -------------
SVD | 0.854	| 0.476
PinSAGE | 0.942	| 0.693
