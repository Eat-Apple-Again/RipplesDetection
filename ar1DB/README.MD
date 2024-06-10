Version Record

KD_segformer_0610
"weights/weights_KD_segformer_0610/weights_KD_segformer_0610_{int(teacher_ratio*100)}"
->
使用 [train_0610] 資料集 (300張train+validation dataset、60張test dataset)
在午仔魚李XX養殖場嘗試 Position Embedding(Cosine Similarity, Absoluted Position Embedding) 和 Focal Loss

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
