Version Record


KD_segformer_0520
"weights/weights_KD_segformer_0520/weights_KD_segformer_0520_{int(teacher_ratio*100)}"
->
使用 train_0501 資料集 (300張train+validation dataset、60張test dataset)
Apply Stratified K-Fold_5
Without Median Filter
Focal Loss in KD_Criterion(evaluate is still use Cross Entropy)

把開始實驗的日期放在檔名裡(包含ipynb, model weight, TrainRecords)
資料集版本將從此紀錄中查看
--------------------------------------------------------
KD_segformer_0604
"weights/weights_KD_segformer_0604/weights_KD_segformer_0604_{int(teacher_ratio*100)}"
->
使用 train_0501 資料集 (300張train+validation dataset、60張test dataset)
方式比[KD_segformer_0520]增加了 Position Embedding(Cosine Similarity, Absoluted Position Embedding) 和 Focal Loss

1. Position Embedding
Segformer 的宣告也改成 MySegFormer_0604，在 ipynb 中增加了選擇 backbone 的參數
Position Embedding 直接改在 conda 環境裡面 Hugging Face 的 Segformer(modeling_segformer.py 的 SegformerDecodeHead())
-> hidden_states shape = (batch_size, channel, height/4, width/4) = (1, 256, 256, 256)
針對 256 的 channel 的影像資訊直接強制加上位置資訊
接著繼續走MLP，在 classifier 之後得到
-> logits shape (batch_size, num_labels, height/4, width/4) = (1, 2, 256, 256)

2. Focal Loss
把 Focal Loss 的 (1 - pt)**gamma 也用來同步 Distillation Loss的影響力
讓 Original Loss 和 Distillation Loss 的權重接近
其中，modulating_number = torch.mean((1 - pt)**gamma)

-> total_loss = modulating_number*((1-teacher_ratio)*original_loss + teacher_ratio*distillation_loss)

--------------------------------------------------------
KD_segformer_0610
"weights/weights_KD_segformer_0610/weights_KD_segformer_0610_{int(teacher_ratio*100)}"
->
使用 train_0501 資料集 (300張train+validation dataset、60張test dataset)
方式和[KD_segformer_0604]的差異:
調整了在 Segformer 裡加的 Position Embedding 比例
因為原本是在256個channel加上位置資訊 ->
# Add Position Embedding
hidden_states_PE = hidden_states + (self.pos_embedding)，但其中
hidden_states 經過了 ReLu, mean = 0.35, std = 0.64
pos_embedding 是 cosine similarity, mean = 0.016(1), std = 0.706

[KD_segformer_0604]的結果顯示效果更差，嘗試降低位置資訊的比例: 嘗試調整成 影像:位置 = 2:1
hidden_states_PE = hidden_states + (self.pos_embedding)*0.5

另外，改寫 Loss Function
total_loss = (1-teacher_ratio)*original_loss + teacher_ratio*distillation_loss*modulating_number
--------------------------------------------------------
KD_segformer_0616
"weights/weights_KD_segformer_0616/weights_KD_segformer_0616_{int(teacher_ratio*100)}"
->
使用 [train_0501] 資料集 (300張train+validation dataset、60張test dataset)
在鱸魚黃XX養殖場使用 Focal Loss 作為 Original Loss，並將設 modulating_number = torch.mean((1 - pt)**gamma)
讓 Criterion 的完整 Loss 是 total_loss = (1-teacher_ratio)*original_loss + teacher_ratio*distillation_loss*modulating_number
讓 Distillation Loss 也可以使用 modulating_number 來關注少數的水花語義

先不使用 Position Embedding
打算先找出 Focal Loss 效果最好者再拿來加上 Position Embedding，以節省時間。
改 Hugging Face 的 Library,把 SegformerDecodeHead 加上有和沒有 Position Embedding 的版本
--------------------------------------------------------
KD_segformer_0621
"weights/weights_KD_segformer_0616/weights_KD_segformer_0616_{int(teacher_ratio*100)}"
->
使用 [train_0501] 資料集 (300張train+validation dataset、60張test dataset)
已完成[KD_segformer_0616]使用 Focal loss 改寫 Criterion 的訓練
接著要嘗試加入 Position Embedding 的效果
在 Hugging Face 的 modeling_segformer.py裡，class SegformerForSemanticSegmentation 的 decode_head 使用 try_SegformerDecodeHead_PE
將位置資訊和特徵資訊合併
--------------------------------------------------------
KD_segformer_0627
"weights/weights_KD_segformer_0627/weights_KD_segformer_0627_{int(teacher_ratio*100)}"
->
使用 [train_0501] 資料集 (300張train+validation dataset、60張test dataset)
Position Embedding 喧賓奪主 XD
把 feature map 和 position embedding 的比例調整成
hidden_states_PE = hidden_states + 0.01*(self.pos_embedding)
在 Hugging Face 的 modeling_segformer.py的第863行
--------------------------------------------------------
