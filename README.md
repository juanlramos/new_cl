# Classificação de Espécies de Plantas com Aprendizado Contrastivo

Projeto de classificação de espécies de plantas pertencentes à mesma família utilizando Contrastive Learning.

A implementação utiliza MoCo (Momentum Contrast) como método de aprendizado contrastivo, com ResNet-50 como backbone. Após o treinamento da representação, as características extraídas pelo modelo são utilizadas em uma segunda etapa de avaliação utilizando um SVM (Support Vector Machine).

---

## 1. Tecnologias utilizadas

* Python
* PyTorch
* Torchvision
* CUDA
* ResNet-50
* MoCo (Momentum Contrast)
* Scikit-learn
* SVM
* TensorBoard
* NumPy
* Pandas
* Matplotlib

---

## 2. Requisitos

Recomenda-se utilizar:

* Python 3.10 ou compatível
* GPU NVIDIA compatível com CUDA para acelerar o treinamento

O projeto também pode ser executado em CPU, porém o treinamento será significativamente mais lento.

---

# 3. Clonando o projeto

Clone o repositório utilizando:

```bash
git clone https://github.com/juanlramos/new_cl.git
```

Entre na pasta do projeto:

```bash
cd new_cl
```

---

# 4. Criando o ambiente virtual

É recomendado utilizar um ambiente virtual para evitar conflitos entre as dependências do projeto e as bibliotecas instaladas globalmente no computador.

No Windows:

```powershell
python -m venv .venv
```

Ative o ambiente virtual:

```powershell
.venv\Scripts\Activate.ps1
```

Quando o ambiente estiver ativo, o terminal deverá apresentar algo semelhante a:

```text
(.venv) PS C:\...\projeto>
```

---

# 5. Instalação do PyTorch e Torchvision com CUDA

O projeto utiliza PyTorch e Torchvision para treinamento do modelo.

Para o ambiente utilizado neste projeto, foram utilizadas:

```text
PyTorch: 2.7.1+cu126
Torchvision: 0.22.1+cu126
```

A instalação utilizada foi:

```powershell
pip3 install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```

> **Observação:** a versão `cu126` corresponde ao build CUDA utilizado pelo PyTorch neste ambiente. A presença de uma versão mais recente do CUDA indicada pelo driver NVIDIA não significa que seja necessário instalar exatamente essa versão para o PyTorch.

Após a instalação, verifique se o PyTorch reconhece a GPU:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU disponível:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Um resultado esperado é semelhante a:

```text
PyTorch: 2.7.1+cu126
CUDA: 12.6
GPU disponível: True
GPU: NVIDIA GeForce RTX 2050
```

Se aparecer:

```text
GPU disponível: False
```

o projeto ainda poderá executar em CPU, porém o treinamento será muito mais lento.

---

# 6. Instalação das demais dependências

Com o ambiente virtual ativado, execute:

```powershell
pip install -r requirements.txt
```

O arquivo `requirements.txt` contém as demais bibliotecas utilizadas pelo projeto:

```text
tensorboard_logger
pytorch-lightning
matplotlib
ipywidgets
seaborn
tqdm
pandas
tensorboard
scikit-learn
```

---

# 7. Estrutura do projeto

A estrutura esperada do projeto é semelhante a:

```text
projeto/
│
├── data/
│   └── pr_dataset/
│       ├── treino/
│       ├── teste/
│       ├── original/
│       ├── info_dataset
│       ├── info_levels
│       └── info_samples
│
├── saved_models/
│
├── main.py
├── config.py
├── moco_builder.py
├── evaluator.py
├── network.py
├── requirements.txt
└── README.md
```

---

# 8. Dataset

O dataset deve ser colocado dentro da pasta:

```text
data/pr_dataset/
```

A estrutura esperada é:

```text
data/
└── pr_dataset/
    ├── treino/
    ├── teste/
    ├── original/
    ├── info_dataset
    ├── info_levels
    └── info_samples
```

---

# 9. Treinamento do MoCo

O treinamento é realizado através do `main.py`.

Com o ambiente virtual ativado e o dataset preparado, execute:

```powershell
python main.py --train --batch_sizes 32
```

Nesse comando:

```text
--train
```

indica que o pipeline deve executar a etapa de treinamento.

```text
--batch_sizes 32
```

define o tamanho do batch utilizado no treinamento.

---

# 10. Configuração do treinamento

As principais configurações estão no arquivo:

```text
config.py
```

Entre elas estão:

```text
batch_size = 64
resize_size = 112
epochs = 2000
learning_rate = 0.05
weight_decay = 1e-4
momentum = 0.9
```

Para executar o experimento com 32 imagens por batch, o valor pode ser informado diretamente pela linha de comando:

```powershell
python main.py --train --batch_sizes 32
```

O número de épocas é definido na configuração de treinamento.

Para um experimento de teste, pode-se utilizar um número menor de épocas, como:

```text
epochs = 10
```

Para o experimento definitivo, deve ser utilizado o número de épocas definido no protocolo experimental do trabalho.

---

# 11. Método MoCo

O projeto utiliza MoCo para realizar o aprendizado contrastivo.

A implementação possui dois encoders:

```text
encoder_q
encoder_k
```

O `encoder_q` é atualizado normalmente pelo processo de otimização.

O `encoder_k` é atualizado utilizando uma média móvel (momentum) do encoder de consulta.

O modelo também utiliza uma fila para armazenar representações utilizadas como exemplos negativos durante o aprendizado contrastivo.

A configuração utilizada pelo modelo inclui:

```text
dim = 128
K = 65536
m = 0.999
T = 0.07
```

Onde:

* `dim`: dimensão da representação aprendida;
* `K`: quantidade de representações negativas mantidas na fila;
* `m`: coeficiente de momentum utilizado para atualizar o encoder de chave;
* `T`: temperatura utilizada no cálculo dos logits contrastivos.

---

# 12. Checkpoints

Durante o treinamento, os modelos são salvos em:

```text
saved_models/
```

A pasta do experimento possui um nome que identifica suas configurações, por exemplo:

```text
saved_models/
└── pr_dataset_VAL_CROP[0.1]_IMG[112]_B[32]_E[10]_resnet50_MoCo/
```

Dentro dela são armazenados os checkpoints:

```text
checkpoints/
└── ckpt_epoch_10.pth
```

O checkpoint contém os parâmetros aprendidos pelo modelo durante o treinamento.

---

# 13. Avaliação

Após o treinamento, é realizada a segunda etapa do experimento: a avaliação da representação aprendida.

A avaliação utiliza um SVM.

Execute:

```powershell
python main.py --eval --batch_sizes 32
```

O programa localizará automaticamente os checkpoints existentes dentro de:

```text
saved_models/<experimento>/checkpoints/
```

e utilizará o modelo treinado para extrair as características das imagens.

---

# 14. Métricas

A avaliação calcula:

### F1-score

O F1-score combina precisão e revocação (recall) em uma única métrica.

O projeto utiliza:

```python
average='weighted'
```

para considerar a quantidade de exemplos de cada classe.

### Top-3 Accuracy

Verifica se a classe correta está entre as três classes com maior probabilidade previstas pelo SVM.

### Top-5 Accuracy

Verifica se a classe correta está entre as cinco classes com maior probabilidade.

---

# 15. Resultados

Os resultados da avaliação são salvos em:

```text
saved_models/<experimento>/results/results.csv
```

O arquivo possui informações como:

```text
epoch,f1_score,top3,top5
```

Exemplo:

```text
10,0.057779057127056506,0.27835051546391754,0.3848797250859107
```

Esses valores permitem acompanhar o desempenho da representação aprendida pelo MoCo.