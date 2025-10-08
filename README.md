## AI based Turn Detection (Whisper + FRU-Adapter)

팀원 : [허명범](https://github.com/MyungBeomHer)

### 프로젝트 주제 
발화 경계 검출 모델 개발

### 프로젝트 언어 및 환경
프로젝트 언어 : Pytorch

### Dataset
- [pipecat-ai/smart-turn-data](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3-train)

you don't have to download this link, code in this repository can download this dataset.

### SETUP
Set up the environments:
```bash
conda create -n turn_detection
conda activate turn_detection
pip install -r requirements.txt
```

You may need to install PortAudio development libraries if not already installed as those are required for PyAudio:
Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-dev
```
macOS (using Homebrew)
```bash
brew install portaudio
```

### 1st train
```bash
CUDA_VISIBLE_DEVICES=2,3 \
MASTER_ADDR=localhost MASTER_PORT=29512 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 train_FRU_parallel.py
```

### next train (you already downloaded the dataset and safetensors of model. So you can not download them one more time.)
```bash
export HF_HOME=/data/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub
export WANDB_API_KEY=dummy_key
export WANDB_MODE=offline

CUDA_VISIBLE_DEVICES=2,3 \
MASTER_ADDR=localhost MASTER_PORT=29512 \
torchrun --standalone --nnodes=1 --nproc_per_node=2 train_FRU_parallel.py
```

### Model
<p align="center">
  <img src="/figure/model.png" width=100%> <br>
</p>

```
class SmartTurnV3Model(WhisperPreTrainedModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)

        config.max_source_positions = 400
        self.encoder = WhisperEncoder(config)

        self.fru_adapter = nn.ModuleList([
            FRU_Adapter(embded_dim=config.d_model) for _ in range(config.encoder_layers)
        ])

        # Use the encoder's hidden size
        hidden_size = config.d_model

        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Initialize classifier weights
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

        # Initialize attention pooling weights
        for module in self.pool_attention:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, input_features, labels=None):
        """
        Forward pass using Whisper encoder only

        Args:
            input_features: Log-mel spectrogram features [batch_size, n_mels, n_frames] - now (batch_size, 80, 800)
            labels: Binary labels for endpointing (1 = complete, 0 = incomplete)
        """
        # Use only the encoder part of Whisper
        
        expected_seq_len = (
            self.encoder.config.max_source_positions
            * self.encoder.conv1.stride[0]
            * self.encoder.conv2.stride[0]
        )
        if input_features.shape[-1] != expected_seq_len:
            raise ValueError(
                f"Whisper expects mel length {expected_seq_len}, but got {input_features.shape[-1]}."
            )

        # conv1/conv2 + GELU
        x = torch.nn.functional.gelu(self.encoder.conv1(input_features))
        x = torch.nn.functional.gelu(self.encoder.conv2(x))  # (B, D, T')

        # (B, T, D)로 변환
        x = x.permute(0, 2, 1)  # (B, T, D)
        T = x.size(1)
        pos_ids = torch.arange(T, device=x.device)
        pos_emb = self.encoder.embed_positions(pos_ids)      # (T, D)
        x = x + pos_emb                                      # (B, T, D)

        # dropout
        x = torch.nn.functional.dropout(x, p=self.encoder.dropout, training=self.training)

        # === 1) Transformer encoder 반복 (병렬 FRU) ===
        # layerdrop 로직도 HF와 동일하게 반영
        for i, encoder_layer in enumerate(self.encoder.layers):
            # (옵션) LayerDrop
            if self.training and (torch.rand([]) < self.encoder.layerdrop):
                x = x + self.fru_adapter[i](x)
                continue

            prev = x  # FRU의 입력은 레이어 입력 x_i

            # WhisperEncoderLayer.forward(...)는 버전에 따라
            # - tensor만 반환하거나
            # - (hidden_states, attn) 형태로 반환할 수 있음 → 안전하게 처리
            out = encoder_layer(
                hidden_states=x,
                attention_mask=None,
                layer_head_mask=None,
                output_attentions=False
            )

            x = out[0] if isinstance(out, (tuple, list)) else out

            # 병렬 residual: Layer 출력 + FRU(prev)
            x = x + self.fru_adapter[i](prev)

        # 마지막 레이어 정규화
        hidden_states = self.encoder.layer_norm(x)  # (B, T, D)
        
        # encoder_outputs = self.encoder(input_features=input_features)
        # hidden_states = encoder_outputs.last_hidden_state
        attention_weights = self.pool_attention(hidden_states)
        attention_weights = softmax(attention_weights, dim=1)
        pooled = torch.sum(hidden_states * attention_weights, dim=1)

        logits = self.classifier(pooled)

        if torch.isnan(logits).any():
            raise ValueError("NaN values detected in logits")

        if labels is not None:
            # Calculate positive sample weight based on batch statistics
            pos_weight = ((labels == 0).sum() / (labels == 1).sum()).clamp(min=0.1, max=10.0)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            labels = labels.float()
            loss = loss_fct(logits.view(-1), labels.view(-1))

            probs = torch.sigmoid(logits.detach())
            return {"loss": loss, "logits": probs}

        probs = torch.sigmoid(logits)
        return {"logits": probs}
```
[train_FRU_parallel.py](train_FRU_parallel.py)

- Benchmark (pipecat-ai Dataset)

|Model|Tuning Method|Trainable Params|Accuracy(Korean)|
|:------:|:------:|:---:|:---:|
|smart turn detection|Full Finetuning|7.8M|96.85|
|smart turn detection+LORA|Partial Tuning|1.8M|97.08||
|**smart turn detection+FRU-Adapter**|Partial Tuning|1.5M|**98.09**||

### Reference Repo
- [Whisper](https://github.com/openai/whisper)
- [Whisper Finetuned by Zeroth-Korean Dataset](https://huggingface.co/seastar105/whisper-medium-ko-zeroth)
- [Zeroth-Korean Dataset](https://huggingface.co/datasets/kresnik/zeroth_korean?utm_source=chatgpt.com)
- [Facebook denoiser](https://github.com/facebookresearch/denoiser)
- [MetricGAN+](https://github.com/wooseok-shin/MetricGAN-plus-pytorch)
- [Demucs](https://github.com/facebookresearch/demucs)
- [LORA](https://github.com/huggingface/peft)
