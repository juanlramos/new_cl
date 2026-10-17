# Classificação de Espécies de Plantas com MoCo

Este projeto implementa um método de aprendizado auto-supervisionado baseado em **Momentum Contrast (MoCo)** para extração de características de imagens de plantas. As características extraídas são utilizadas posteriormente em um classificador SVM para avaliar o desempenho na classificação das espécies.

O projeto permite utilizar as versões **MoCo v1 e MoCo v2**, com a opção de inicializar a ResNet-50 com pesos pré-treinados no ImageNet.

## 1. Requisitos

- Python 3.10 ou compatível com as dependências do projeto.
- PyTorch e Torchvision.
- GPU NVIDIA compatível com a versão CUDA instalada é recomendada para acelerar o treinamento.
- Dataset de imagens organizado conforme a estrutura descrita abaixo.

Instale as dependências utilizando:

```bash
pip install -r requirements.txt
```

**Observação:** a instalação do PyTorch pode variar conforme o sistema operacional e a versão do CUDA. Consulte a documentação oficial do [PyTorch](https://pytorch.org/get-started/locally/) caso seja necessário instalar uma versão específica.

## 2. Dataset

O dataset utilizado neste projeto contém imagens de espécies de plantas pertencentes à mesma família.

O conjunto de dados não é versionado no GitHub. Portanto, é necessário obtê-lo separadamente aqui: [Dataset](https://drive.google.com/drive/folders/1JEDnJ1lWobpZnsCRRyNjIg9cYRQHalwv). O dataset deve ser colocado no diretório `data/pr_dataset/`.

A estrutura esperada é:

```text
data/
└── pr_dataset/
    ├── original/
    ├── treino/
    ├── teste/
    ├── info_dataset
    ├── info_levels
    └── info_samples
```

- `original/`: imagens originais do dataset.
- `treino/`: imagens destinadas ao treinamento.
- `teste/`: imagens destinadas à avaliação.
- `info_dataset`, `info_levels` e `info_samples`: arquivos auxiliares com informações sobre o conjunto de dados.

**Importante:** mantenha a divisão original entre treinamento e teste fornecida para o projeto. Não gere uma nova divisão sem necessidade, pois isso pode comprometer a comparação com os experimentos anteriores.

## 3. Configuração

As principais configurações estão no arquivo `config.py`.

Entre os parâmetros disponíveis estão:

| Parâmetro | Descrição | Valor padrão |
|---|---|---|
| `dataset_name` | Nome do dataset | `pr_dataset` |
| `resize_size` | Tamanho para redimensionamento das imagens | `112` |
| `batch_size` | Tamanho do lote | `64` no arquivo de configuração |
| `epochs` | Número de épocas de treinamento | `2000` |
| `learning_rate` | Taxa de aprendizado | `0.05` |
| `weight_decay` | Regularização do otimizador | `0.0001` |
| `momentum` | Momentum do SGD | `0.9` |
| `moco_version` | Versão do MoCo | `2` |
| `pretrained` | Inicialização com pesos do ImageNet | `False` |

O argumento `--batch_sizes` da linha de comando tem valor padrão `32`. Quando esse argumento é utilizado, o valor informado substitui o tamanho do lote definido no arquivo de configuração.

O tamanho das imagens pode ser definido pela linha de comando utilizando o argumento `--resize_size`. Os valores aceitos são `112` e `224`, sendo `112` o padrão caso o argumento não seja informado.

Exemplos:

```bash
# Imagens com tamanho 112x112
python main.py --train --batch_sizes 32 --moco_version 1 --resize_size 112

# Imagens com tamanho 224x224
python main.py --train --batch_sizes 32 --moco_version 1 --resize_size 224
```

O tamanho informado é aplicado ao processamento das imagens e registrado no nome do diretório do experimento, permitindo distinguir experimentos realizados com tamanhos diferentes.

## 4. Treinamento

O treinamento é executado pelo arquivo `main.py`.

### 4.1. Treinar com MoCo v1

Para treinar utilizando a primeira versão do MoCo:

```bash
python main.py --train --batch_sizes 32 --moco_version 1
```

### 4.2. Treinar com MoCo v2

Para utilizar a segunda versão do MoCo:

```bash
python main.py --train --batch_sizes 32 --moco_version 2
```

A principal diferença de implementação entre as versões utilizadas neste projeto é a aplicação de uma MLP na cabeça de projeção do encoder, habilitada no MoCo v2.

### 4.3. Utilizar pesos pré-treinados no ImageNet

Para inicializar a ResNet-50 com pesos pré-treinados no ImageNet, utilize o argumento `--pre_trained`.

Exemplo com MoCo v1:

```bash
python main.py --train --batch_sizes 32 --moco_version 1 --pre_trained
```

Exemplo com MoCo v2:

```bash
python main.py --train --batch_sizes 32 --moco_version 2 --pre_trained
```

O argumento `--pre_trained` habilita a inicialização do backbone ResNet-50 com pesos do ImageNet. Isso **não significa carregar um checkpoint de um treinamento MoCo anterior**: os pesos pré-treinados são utilizados como ponto de partida para o treinamento do projeto.

### 4.4. Utilizar diferentes tamanhos de lote

É possível executar experimentos com diferentes tamanhos de lote:

```bash
python main.py --train --batch_sizes 8 16 32 64 --moco_version 2
```

Nesse exemplo, o programa executa um treinamento para cada tamanho de lote informado.

O treinamento também pode ser combinado com a opção de pesos pré-treinados:

```bash
python main.py --train --batch_sizes 8 16 32 64 --moco_version 2 --pre_trained
```

**Atenção:** o programa não permite iniciar um treinamento caso o diretório do experimento já exista. Isso evita sobrescrever acidentalmente os resultados de uma execução anterior. Para repetir um experimento, verifique o diretório correspondente antes de executar o comando.

## 5. Organização dos resultados

Os experimentos são organizados em diretórios dentro de `saved_models/`.

O nome de cada diretório registra informações importantes do experimento, como dataset, tamanho das imagens, tamanho do lote, número de épocas, arquitetura, versão do MoCo e uso de pesos pré-treinados.

Exemplo:

```text
saved_models/
└── pr_dataset_VAL_CROP[0.1]_IMG[112]_B[32]_E[2000]_resnet50_MoCo_V2_Pretrained/
    ├── checkpoints/
    ├── logs/
    └── results/
```

Os diretórios têm as seguintes finalidades:

- `checkpoints/`: armazena os checkpoints do treinamento.
- `logs/`: armazena os registros utilizados para acompanhar o treinamento.
- `results/`: armazena os resultados da avaliação.

Os nomes exatos dos diretórios dependem das configurações utilizadas. Por exemplo, a versão do MoCo e a opção de pesos pré-treinados alteram o nome do experimento.

## 6. Checkpoints

Os checkpoints são salvos periodicamente durante o treinamento, conforme a configuração `save_freq` em `config.py`, e também ao final da execução.

Os arquivos seguem o padrão:

```text
checkpoints/
├── ckpt_epoch_100.pth
├── ckpt_epoch_200.pth
├── ckpt_epoch_300.pth
└── ...
```

Cada checkpoint contém informações como:

- Época do treinamento.
- Pesos do modelo.
- Estado do otimizador.
- Configuração utilizada no experimento.

Esses arquivos permitem utilizar os pesos obtidos durante o treinamento na etapa de avaliação.

## 7. Avaliação

A avaliação utiliza as características extraídas pelo encoder treinado para ajustar e avaliar um classificador SVM.

O procedimento inclui a extração das características, a normalização com `StandardScaler` e a busca de hiperparâmetros do SVM por meio de `GridSearchCV`, utilizando validação cruzada com três divisões.

### 7.1. Avaliar um experimento MoCo v1

Para avaliar um experimento treinado com MoCo v1 e sem pesos pré-treinados:

```bash
python main.py --eval --batch_sizes 32 --moco_version 1
```

Para um experimento MoCo v1 que utilizou pesos pré-treinados:

```bash
python main.py --eval --batch_sizes 32 --moco_version 1 --pre_trained
```

### 7.2. Avaliar um experimento MoCo v2

Para avaliar um experimento treinado com MoCo v2 e sem pesos pré-treinados:

```bash
python main.py --eval --batch_sizes 32 --moco_version 2
```

Para um experimento MoCo v2 que utilizou pesos pré-treinados:

```bash
python main.py --eval --batch_sizes 32 --moco_version 2 --pre_trained
```

**Importante:** os argumentos usados na avaliação devem corresponder aos utilizados no treinamento, incluindo a versão do MoCo, o uso de pesos pré-treinados, o tamanho do lote e o tamanho das imagens (`--resize_size`). Essas configurações participam da identificação do diretório do experimento.

A avaliação precisa encontrar os checkpoints já gerados. Portanto, execute o treinamento antes de avaliar um novo experimento.

## 8. Métricas

Os resultados da avaliação são armazenados no arquivo:

```text
saved_models/<diretorio_do_experimento>/results/results.csv
```

O arquivo contém as seguintes colunas:

| Coluna | Descrição |
|---|---|
| `epoch` | Época correspondente ao checkpoint avaliado |
| `f1_score` | F1-score ponderado utilizado para avaliar a classificação |
| `top3_acc` | Acurácia Top-3 |
| `top5_acc` | Acurácia Top-5 |

A avaliação é realizada para os checkpoints encontrados no diretório do experimento. Os resultados são registrados no CSV, permitindo comparar o desempenho ao longo do treinamento.

## 9. Principais argumentos

| Argumento | Descrição |
|---|---|
| `--train` | Executa o treinamento |
| `--eval` | Executa a avaliação dos checkpoints |
| `--batch_sizes` | Define um ou mais tamanhos de lote |
| `--moco_version {1,2}` | Seleciona MoCo v1 ou MoCo v2 |
| `--pre_trained` | Habilita os pesos pré-treinados do ImageNet |
| `--resize_size {112,224}` | Define o tamanho das imagens de entrada; padrão: `112` |

Para consultar a ajuda disponibilizada pelo programa:

```bash
python main.py --help
```

## 10. Reprodutibilidade dos experimentos

Para comparar os resultados de maneira consistente:

1. Utilize a mesma divisão do dataset entre treinamento e teste.
2. Mantenha as configurações do experimento registradas.
3. Utilize a mesma versão do MoCo na avaliação e no treinamento.
4. Informe `--pre_trained` na avaliação quando essa opção tiver sido utilizada no treinamento.
5. Utilize o mesmo tamanho de lote para localizar o diretório correto do experimento.
6. Utilize o mesmo valor de `--resize_size` no treinamento e na avaliação.
7. Preserve os checkpoints e os arquivos CSV gerados.

A reprodutibilidade também depende de fatores como versões das bibliotecas, disponibilidade de GPU, sementes aleatórias e comportamento não determinístico de algumas operações. Portanto, registrar o ambiente e as configurações utilizadas é importante para interpretar diferenças entre execuções.

## 11. Estrutura geral do projeto

```text
new_cl/
├── data/
│   ├── pr_dataset/
│   └── split_dataset.py
├── saved_models/
├── config.py
├── main.py
├── trainer.py
├── evaluator.py
├── moco_builder.py
└── requirements.txt
```

A estrutura acima apresenta os principais arquivos e diretórios do projeto. Outros módulos e arquivos auxiliares podem estar presentes no repositório.

## 12. Observações finais

Este projeto permite investigar a utilização do aprendizado auto-supervisionado com Momentum Contrast na classificação de espécies de plantas.

As principais possibilidades de experimentação incluem:

- Comparação entre MoCo v1 e MoCo v2.
- Comparação entre treinamentos com e sem pesos pré-treinados no ImageNet.
- Avaliação de diferentes tamanhos de lote.
- Comparação entre imagens de entrada de 112 × 112 e 224 × 224.
- Comparação das métricas de classificação ao longo das épocas.

Os resultados devem ser interpretados considerando as configurações de cada experimento e a divisão do dataset utilizada.