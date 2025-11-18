



# TorchInfo Unique Layers

## Non‑PyTorch Modules

### LlamaFlashAttention2


**class_name**: LlamaFlashAttention2

**class_name_qualified**: transformers.models.llama.modeling_llama.LlamaFlashAttention2

**filepaths**:
- /workspace/code/llm-perf-opt/.pixi/envs/rtx5090/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py


**children**:
- torch.nn.modules.linear.Linear
- transformers.models.llama.modeling_llama.LlamaRotaryEmbedding

### LlamaRotaryEmbedding


**class_name**: LlamaRotaryEmbedding

**class_name_qualified**: transformers.models.llama.modeling_llama.LlamaRotaryEmbedding

**filepaths**:
- /workspace/code/llm-perf-opt/.pixi/envs/rtx5090/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py


**children**: (none)
### Attention


**class_name**: Attention

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.Attention

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/deepencoder.py


**children**:
- torch.nn.modules.linear.Linear

### Block


**class_name**: Block

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.Block

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/deepencoder.py


**children**:
- torch.nn.modules.activation.GELU
- torch.nn.modules.linear.Linear
- torch.nn.modules.normalization.LayerNorm
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.Attention
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.MLPBlock

### CLIPVisionEmbeddings


**class_name**: CLIPVisionEmbeddings

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.CLIPVisionEmbeddings

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/deepencoder.py


**children**:
- torch.nn.modules.sparse.Embedding

### ImageEncoderViT


**class_name**: ImageEncoderViT

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.ImageEncoderViT

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/deepencoder.py


**children**:
- torch.nn.modules.activation.GELU
- torch.nn.modules.container.ModuleList
- torch.nn.modules.container.Sequential
- torch.nn.modules.conv.Conv2d
- torch.nn.modules.linear.Linear
- torch.nn.modules.normalization.LayerNorm
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.Attention
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.Block
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.LayerNorm2d
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.MLPBlock
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.PatchEmbed

### LayerNorm2d


**class_name**: LayerNorm2d

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.LayerNorm2d

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/deepencoder.py


**children**: (none)
### MLPBlock


**class_name**: MLPBlock

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.MLPBlock

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/deepencoder.py


**children**:
- torch.nn.modules.activation.GELU
- torch.nn.modules.linear.Linear

### MlpProjector


**class_name**: MlpProjector

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.MlpProjector

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/deepencoder.py


**children**:
- torch.nn.modules.linear.Linear

### NoTPAttention


**class_name**: NoTPAttention

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPAttention

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/deepencoder.py


**children**: (none)
### NoTPFeedForward


**class_name**: NoTPFeedForward

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPFeedForward

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/deepencoder.py


**children**: (none)
### NoTPTransformer


**class_name**: NoTPTransformer

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPTransformer

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/deepencoder.py


**children**:
- torch.nn.modules.container.ModuleList
- torch.nn.modules.linear.Linear
- torch.nn.modules.normalization.LayerNorm
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPAttention
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPFeedForward
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPTransformerBlock

### NoTPTransformerBlock


**class_name**: NoTPTransformerBlock

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPTransformerBlock

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/deepencoder.py


**children**:
- torch.nn.modules.linear.Linear
- torch.nn.modules.normalization.LayerNorm
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPAttention
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPFeedForward

### PatchEmbed


**class_name**: PatchEmbed

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.PatchEmbed

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/deepencoder.py


**children**:
- torch.nn.modules.conv.Conv2d

### VitModel


**class_name**: VitModel

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.VitModel

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/deepencoder.py


**children**:
- torch.nn.modules.container.ModuleList
- torch.nn.modules.linear.Linear
- torch.nn.modules.normalization.LayerNorm
- torch.nn.modules.sparse.Embedding
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.CLIPVisionEmbeddings
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPAttention
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPFeedForward
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPTransformer
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPTransformerBlock

### DeepseekOCRModel


**class_name**: DeepseekOCRModel

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekocr.DeepseekOCRModel

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/modeling_deepseekocr.py


**children**:
- torch.nn.modules.activation.GELU
- torch.nn.modules.activation.SiLU
- torch.nn.modules.container.ModuleList
- torch.nn.modules.container.Sequential
- torch.nn.modules.conv.Conv2d
- torch.nn.modules.linear.Linear
- torch.nn.modules.normalization.LayerNorm
- torch.nn.modules.sparse.Embedding
- transformers.models.llama.modeling_llama.LlamaFlashAttention2
- transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.Attention
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.Block
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.CLIPVisionEmbeddings
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.ImageEncoderViT
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.LayerNorm2d
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.MLPBlock
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.MlpProjector
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPAttention
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPFeedForward
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPTransformer
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.NoTPTransformerBlock
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.PatchEmbed
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.VitModel
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2DecoderLayer
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2MLP
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2MoE
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2RMSNorm
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.MoEGate

### DeepseekV2DecoderLayer


**class_name**: DeepseekV2DecoderLayer

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2DecoderLayer

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/modeling_deepseekv2.py


**children**:
- torch.nn.modules.activation.SiLU
- torch.nn.modules.container.ModuleList
- torch.nn.modules.linear.Linear
- transformers.models.llama.modeling_llama.LlamaFlashAttention2
- transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2MLP
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2MoE
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2RMSNorm
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.MoEGate

### DeepseekV2MLP


**class_name**: DeepseekV2MLP

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2MLP

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/modeling_deepseekv2.py


**children**:
- torch.nn.modules.activation.SiLU
- torch.nn.modules.linear.Linear

### DeepseekV2MoE


**class_name**: DeepseekV2MoE

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2MoE

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/modeling_deepseekv2.py


**children**:
- torch.nn.modules.activation.SiLU
- torch.nn.modules.container.ModuleList
- torch.nn.modules.linear.Linear
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2MLP
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.MoEGate

### DeepseekV2RMSNorm


**class_name**: DeepseekV2RMSNorm

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2RMSNorm

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/modeling_deepseekv2.py


**children**: (none)
### MoEGate


**class_name**: MoEGate

**class_name_qualified**: transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.MoEGate

**filepaths**:
- /home/igame/.cache/huggingface/modules/transformers_modules/2c968b433af61a059311cbf8997765023806a24d/modeling_deepseekv2.py


**children**: (none)
## PyTorch Builtins

### GELU


**class_name**: GELU

**class_name_qualified**: torch.nn.modules.activation.GELU

**filepaths**:
- /workspace/code/llm-perf-opt/.pixi/envs/rtx5090/lib/python3.12/site-packages/torch/nn/modules/activation.py


**children**: (none)
### SiLU


**class_name**: SiLU

**class_name_qualified**: torch.nn.modules.activation.SiLU

**filepaths**:
- /workspace/code/llm-perf-opt/.pixi/envs/rtx5090/lib/python3.12/site-packages/torch/nn/modules/activation.py


**children**: (none)
### ModuleList


**class_name**: ModuleList

**class_name_qualified**: torch.nn.modules.container.ModuleList

**filepaths**:
- /workspace/code/llm-perf-opt/.pixi/envs/rtx5090/lib/python3.12/site-packages/torch/nn/modules/container.py


**children**:
- torch.nn.modules.activation.SiLU
- torch.nn.modules.container.ModuleList
- torch.nn.modules.linear.Linear
- transformers.models.llama.modeling_llama.LlamaFlashAttention2
- transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2DecoderLayer
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2MLP
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2MoE
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.DeepseekV2RMSNorm
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.modeling_deepseekv2.MoEGate

### Sequential


**class_name**: Sequential

**class_name_qualified**: torch.nn.modules.container.Sequential

**filepaths**:
- /workspace/code/llm-perf-opt/.pixi/envs/rtx5090/lib/python3.12/site-packages/torch/nn/modules/container.py


**children**:
- torch.nn.modules.conv.Conv2d
- transformers_modules.2c968b433af61a059311cbf8997765023806a24d.deepencoder.LayerNorm2d

### Conv2d


**class_name**: Conv2d

**class_name_qualified**: torch.nn.modules.conv.Conv2d

**filepaths**:
- /workspace/code/llm-perf-opt/.pixi/envs/rtx5090/lib/python3.12/site-packages/torch/nn/modules/conv.py


**children**: (none)
### Linear


**class_name**: Linear

**class_name_qualified**: torch.nn.modules.linear.Linear

**filepaths**:
- /workspace/code/llm-perf-opt/.pixi/envs/rtx5090/lib/python3.12/site-packages/torch/nn/modules/linear.py


**children**: (none)
### LayerNorm


**class_name**: LayerNorm

**class_name_qualified**: torch.nn.modules.normalization.LayerNorm

**filepaths**:
- /workspace/code/llm-perf-opt/.pixi/envs/rtx5090/lib/python3.12/site-packages/torch/nn/modules/normalization.py


**children**: (none)
### Embedding


**class_name**: Embedding

**class_name_qualified**: torch.nn.modules.sparse.Embedding

**filepaths**:
- /workspace/code/llm-perf-opt/.pixi/envs/rtx5090/lib/python3.12/site-packages/torch/nn/modules/sparse.py


**children**: (none)