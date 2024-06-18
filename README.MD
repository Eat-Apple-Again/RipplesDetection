Version Record

KD_segformer_0616
"weights/weights_KD_segformer_0616/weights_KD_segformer_0616_{int(teacher_ratio*100)}"
->
使用 [train_0518] 資料集 (300張train+validation dataset、60張test dataset)
在午仔魚張XX養殖場使用 Focal Loss 作為 Original Loss，並將設 modulating_number = torch.mean((1 - pt)**gamma)
讓 Criterion 的完整 Loss 是 total_loss = (1-teacher_ratio)*original_loss + teacher_ratio*distillation_loss*modulating_number
讓 Distillation Loss 也可以使用 modulating_number 來關注少數的水花語義

先不使用 Position Embedding
打算先找出 Focal Loss 效果最好者再拿來加上 Position Embedding，以節省時間。
改 Hugging Face 的 Library,把 SegformerDecodeHead 加上有和沒有 Position Embedding 的版本
--------------------------------------------------------

