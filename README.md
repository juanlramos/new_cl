# Classificação de Espécies de Plantas com MoCo

Este projeto implementa um método de aprendizado auto-supervisionado baseado em **Momentum Contrast (MoCo)** para extração de características de imagens de plantas. As características extraídas são utilizadas por um classificador SVM para avaliar o desempenho na classificação das espécies.

O projeto permite utilizar as versões **MoCo v1 e MoCo v2**, selecionar entre as arquiteturas **ResNet-50 e ResNet-152** e inicializar o backbone com pesos pré-treinados no ImageNet.

Além da avaliação convencional, o projeto permite investigar como a quantidade de dados rotulados disponíveis para o treinamento do SVM influencia o desempenho da classificação, utilizando diferentes porcentagens do conjunto de treinamento.

## 1. Requisitos

- Python 3.10 ou compatível com as dependências do projeto.
- PyTorch e Torchvision.
- GPU NVIDIA compatível com a versão CUDA instalada é recomendada para acelerar o treinamento.
- Dataset de imagens organizado conforme a estrutura descrita abaixo.

Instale as dependências utilizando:

```bash
pip install -r requirements.txt
```

**Observação:** a instalação do PyTorch pode variar conforme o sistema operacional e a versão do CUDA. Consulte a [documentação oficial do PyTorch](https://pytorch.org/get-started/locally/) caso seja necessário instalar uma versão específica.

## 2. Dataset

O dataset utilizado neste projeto contém imagens de espécies de plantas pertencentes à mesma família.

O conjunto de dados não é versionado no GitHub. Portanto, é necessário obtê-lo separadamente pelo seguinte link: [Dataset](https://drive.google.com/drive/folders/1JEDnJ1lWobpZnsCRRyNjIg9cYRQHalwv).

O dataset deve ser colocado no diretório `data/pr_dataset/`.

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

A avaliação com diferentes porcentagens de dados rotulados não modifica as pastas do dataset. A seleção das amostras utilizadas pelo SVM é realizada durante a execução do programa.

## 3. Configuração

As principais configurações estão no arquivo `config.py`.

Entre os parâmetros disponíveis estão:

| Parâmetro | Descrição | Valor padrão |
|---|---|---|
| `dataset_name` | Nome do dataset | `pr_dataset` |
| `resize_size` | Tamanho para redimensionamento das imagens | `112` |
| `batch_size` | Tamanho do lote no arquivo de configuração | `64` |
| `epochs` | Número de épocas de treinamento | `2000` |
| `learning_rate` | Taxa de aprendizado | `0.05` |
| `weight_decay` | Regularização do otimizador | `0.0001` |
| `momentum` | Momentum do SGD | `0.9` |
| `moco_version` | Versão do MoCo | `2` |
| `pretrained` | Inicialização com pesos do ImageNet | `False` |
| `name` | Arquitetura utilizada | `resnet50` |

O argumento `--batch_sizes` da linha de comando tem valor padrão `32`. Quando utilizado, o valor informado substitui o tamanho do lote definido no arquivo de configuração.

O tamanho das imagens pode ser definido pela linha de comando utilizando `--resize_size`. Os valores aceitos são `112` e `224`, sendo `112` o padrão.

Por padrão, o projeto utiliza a ResNet-50. Para utilizar a ResNet-152, informe o argumento `--resnet_152`.

Exemplos:

```bash
# Imagens com tamanho 112x112
python main.py --train --batch_sizes 32 --moco_version 1 --resize_size 112

# Imagens com tamanho 224x224
python main.py --train --batch_sizes 32 --moco_version 1 --resize_size 224
```

O tamanho informado é aplicado ao processamento das imagens e registrado no nome do diretório do experimento.

## 4. Treinamento

O treinamento é executado pelo arquivo `main.py`.

### 4.1. Treinar com MoCo v1

```bash
python main.py --train --batch_sizes 32 --moco_version 1
```

### 4.2. Treinar com MoCo v2

```bash
python main.py --train --batch_sizes 32 --moco_version 2
```

A principal diferença de implementação entre as versões utilizadas neste projeto é a aplicação de uma MLP na cabeça de projeção do encoder, habilitada no MoCo v2.

### 4.3. Utilizar pesos pré-treinados no ImageNet

Para inicializar o backbone com pesos pré-treinados no ImageNet, utilize o argumento `--pre_trained`.

Exemplo com MoCo v1:

```bash
python main.py --train --batch_sizes 32 --moco_version 1 --pre_trained
```

Exemplo com MoCo v2:

```bash
python main.py --train --batch_sizes 32 --moco_version 2 --pre_trained
```

O argumento `--pre_trained` habilita a inicialização do backbone com pesos do ImageNet. Isso **não significa carregar um checkpoint de um treinamento MoCo anterior**: os pesos pré-treinados são utilizados como ponto de partida para o treinamento do projeto.

### 4.4. Utilizar diferentes tamanhos de lote

É possível executar experimentos com diferentes tamanhos de lote:

```bash
python main.py --train --batch_sizes 8 16 32 64 --moco_version 2
```

O treinamento também pode ser combinado com a opção de pesos pré-treinados:

```bash
python main.py --train --batch_sizes 8 16 32 64 --moco_version 2 --pre_trained
```

**Atenção:** o programa não permite iniciar um treinamento caso o diretório do experimento já exista. Isso evita sobrescrever acidentalmente os resultados de uma execução anterior.

### 4.5. Utilizar a ResNet-152

Para realizar o treinamento com a ResNet-152, utilize o argumento `--resnet_152`.

Exemplo com MoCo v1, imagens de 224 × 224 e pesos pré-treinados no ImageNet:

```bash
python main.py --train --batch_sizes 32 --moco_version 1 --pre_trained --resize_size 224 --resnet_152
```

Também é possível executar experimentos com diferentes tamanhos de lote:

```bash
python main.py --train --batch_sizes 8 16 32 64 --moco_version 1 --pre_trained --resize_size 224 --resnet_152
```

Quando combinado com `--pre_trained`, a arquitetura é inicializada com os pesos pré-treinados correspondentes disponíveis no ImageNet.

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

Os nomes exatos dos diretórios dependem das configurações utilizadas. Por isso, os argumentos utilizados na avaliação devem corresponder aos do experimento que se deseja avaliar.

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

Esses arquivos permitem utilizar os pesos obtidos durante o treinamento na etapa de avaliação, sem precisar treinar o modelo novamente.

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

Para avaliar um experimento com ResNet-152, MoCo v1 e pesos pré-treinados no ImageNet:

```bash
python main.py --eval --batch_sizes 32 --moco_version 1 --pre_trained --resize_size 224 --resnet_152
```

O argumento `--resnet_152` deve ser informado para que o programa localize o diretório do experimento correspondente à ResNet-152.

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

### 7.3. Avaliar com diferentes porcentagens de dados rotulados

O projeto permite avaliar o classificador SVM utilizando diferentes porcentagens das imagens do conjunto de treinamento como dados rotulados.

Essa funcionalidade possibilita investigar como a quantidade de dados rotulados influencia o desempenho da classificação, mantendo o modelo MoCo já treinado e o conjunto de teste completo.

As porcentagens aceitas vão de 1% a 100%. É possível informar vários valores em uma única execução. Por exemplo:

```bash
python main.py --eval --batch_sizes 8 --moco_version 1 --pre_trained --resize_size 224 --percent_labels 10 15 20 25 30 40 50 100
```

Nesse exemplo, o programa realiza avaliações com:

- 10% dos dados de treinamento rotulados.
- 15% dos dados de treinamento rotulados.
- 20% dos dados de treinamento rotulados.
- 25% dos dados de treinamento rotulados.
- 30% dos dados de treinamento rotulados.
- 40% dos dados de treinamento rotulados.
- 50% dos dados de treinamento rotulados.
- 100% dos dados de treinamento rotulados.

Para cada porcentagem, o programa seleciona as amostras do conjunto de treinamento, ajusta um novo SVM e calcula as métricas utilizando o conjunto de teste completo.

A seleção das amostras é realizada em memória, sem alterar ou recriar as pastas `original/`, `treino/` e `teste/`. Na implementação atual, a seleção utiliza amostragem estratificada com semente fixa, buscando preservar a representação das classes e tornar o procedimento reproduzível.

**Observação:** a opção `--percent_labels` controla a quantidade de dados utilizados para treinar o SVM. Ela não altera a quantidade de imagens usada no treinamento auto-supervisionado do MoCo e não executa um novo treinamento MoCo.

### 7.4. Seleção automática do melhor checkpoint

Na avaliação com diferentes porcentagens, o programa consulta o arquivo `results.csv` do experimento para identificar a época com o maior `f1_score`.

Em seguida, carrega o checkpoint correspondente, seguindo o padrão:

```text
checkpoints/ckpt_epoch_<epoca>.pth
```

Assim, não é necessário informar manualmente a época escolhida.

Essa funcionalidade pressupõe que `results.csv` contenha os resultados das épocas e que o checkpoint correspondente à época selecionada esteja disponível no diretório `checkpoints/`.

**Atenção metodológica:** quando o maior F1-Score é escolhido com base em resultados calculados no conjunto de teste, esse conjunto também influencia a seleção do modelo. Para uma avaliação experimental sem esse viés, a escolha do checkpoint deve ser realizada com base em um conjunto de validação separado, reservando o teste para a avaliação final.

## 8. Métricas e arquivos de resultados

Os resultados são armazenados no diretório `results/` do experimento.

### 8.1. Avaliação convencional

O arquivo utilizado na avaliação convencional é:

```text
results/results.csv
```

Ele contém as seguintes colunas:

| Coluna | Descrição |
|---|---|
| `epoch` | Época correspondente ao checkpoint avaliado |
| `f1_score` | F1-Score ponderado utilizado para avaliar a classificação |
| `top3_acc` | Acurácia Top-3 |
| `top5_acc` | Acurácia Top-5 |

### 8.2. Avaliação por porcentagem de dados rotulados

Na avaliação com diferentes porcentagens, o programa salva um arquivo CSV separado para cada valor informado.

Exemplo:

```text
results/
├── loss.csv
├── results.csv
├── results_labels_10.csv
├── results_labels_15.csv
├── results_labels_20.csv
├── results_labels_25.csv
├── results_labels_30.csv
├── results_labels_40.csv
├── results_labels_50.csv
└── results_labels_100.csv
```

Cada arquivo `results_labels_<porcentagem>.csv` registra o resultado da avaliação correspondente à porcentagem utilizada.

As colunas são:

| Coluna | Descrição |
|---|---|
| `epoch` | Época do checkpoint selecionado |
| `percent_labels` | Porcentagem de dados rotulados utilizada no treinamento do SVM |
| `f1_score` | F1-Score ponderado |
| `top3_acc` | Acurácia Top-3 |
| `top5_acc` | Acurácia Top-5 |

O arquivo `results.csv` original não é substituído por esses resultados. Caso uma avaliação com a mesma porcentagem seja executada novamente, o arquivo correspondente será atualizado.

Esses arquivos permitem comparar o desempenho do classificador conforme a quantidade de dados rotulados disponível.

## 9. Principais argumentos

| Argumento | Descrição |
|---|---|
| `--train` | Executa o treinamento |
| `--eval` | Executa a avaliação dos checkpoints |
| `--batch_sizes` | Define um ou mais tamanhos de lote |
| `--moco_version {1,2}` | Seleciona MoCo v1 ou MoCo v2 |
| `--pre_trained` | Habilita os pesos pré-treinados do ImageNet |
| `--resize_size {112,224}` | Define o tamanho das imagens de entrada; padrão: `112` |
| `--resnet_152` | Utiliza ResNet-152 em vez de ResNet-50 |
| `--percent_labels` | Define uma ou mais porcentagens de dados rotulados para a avaliação; utilizada com `--eval` |

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
8. Mantenha as porcentagens e a semente aleatória constantes ao comparar as avaliações com diferentes quantidades de dados rotulados.

A reprodutibilidade também depende de fatores como versões das bibliotecas, disponibilidade de GPU, sementes aleatórias e comportamento não determinístico de algumas operações. Registrar o ambiente e as configurações utilizadas é importante para interpretar diferenças entre execuções.

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

A estrutura apresenta os principais arquivos e diretórios do projeto. Outros módulos e arquivos auxiliares podem estar presentes no repositório.

## 12. Observações finais

Este projeto permite investigar a utilização do aprendizado auto-supervisionado com Momentum Contrast na classificação de espécies de plantas.

As principais possibilidades de experimentação incluem:

- Comparação entre MoCo v1 e MoCo v2.
- Comparação entre treinamentos com e sem pesos pré-treinados no ImageNet.
- Avaliação de diferentes tamanhos de lote.
- Comparação entre imagens de entrada de 112 × 112 e 224 × 224.
- Comparação das métricas de classificação ao longo das épocas.
- Análise do desempenho do SVM com diferentes porcentagens de dados rotulados, sem a necessidade de treinar novamente o MoCo.

Os resultados devem ser interpretados considerando as configurações de cada experimento, o procedimento de seleção das amostras e a divisão do dataset utilizada.
