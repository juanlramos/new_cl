
import os
import logging
import re
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import f1_score, top_k_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from config import Config
import moco_builder
import torchvision.models as models


logger = logging.getLogger(__name__)


class LinearEvaluator:
    def __init__(
        self,
        config: Config,
        train_loader,
        test_loader,
        percent_labels: list[int] | int = 100
    ):
        # Aceita tanto uma porcentagem quanto uma lista de porcentagens
        if isinstance(percent_labels, int):
            percent_labels = [percent_labels]

        if not percent_labels:
            raise ValueError("Informe pelo menos uma porcentagem.")

        if any(not 1 <= p <= 100 for p in percent_labels):
            raise ValueError(
                "Todas as porcentagens devem estar entre 1 e 100."
            )

        if len(percent_labels) != len(set(percent_labels)):
            raise ValueError("Existem porcentagens repetidas.")

        self.cfg = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.percent_labels = percent_labels
        self.device = self.cfg.train.device

        self.results_path = os.path.join(
            self.cfg.train.checkpoint_dir,
            "results"
        )
        os.makedirs(self.results_path, exist_ok=True)

        # Resultados da avaliação anterior, usados para escolher a época
        self.csv_file = os.path.join(
            self.results_path,
            "results.csv"
        )

    def find_best_checkpoint(self):
        """Encontra o checkpoint correspondente ao maior F1-Score."""

        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(
                f"Arquivo de resultados não encontrado: {self.csv_file}"
            )

        df = pd.read_csv(self.csv_file)

        if not {"epoch", "f1_score"}.issubset(df.columns):
            raise ValueError(
                "O results.csv precisa conter as colunas "
                "'epoch' e 'f1_score'."
            )

        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df["f1_score"] = pd.to_numeric(
            df["f1_score"], errors="coerce"
        )

        df = df.dropna(subset=["epoch", "f1_score"])

        if df.empty:
            raise ValueError(
                "Não existem resultados válidos no results.csv."
            )

        best_row = df.loc[df["f1_score"].idxmax()]
        best_epoch = int(best_row["epoch"])
        best_f1 = float(best_row["f1_score"])

        checkpoints_dir = (
            Path(self.cfg.train.checkpoint_dir) / "checkpoints"
        )
        checkpoint_path = (
            checkpoints_dir / f"ckpt_epoch_{best_epoch}.pth"
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint da melhor época não encontrado: "
                f"{checkpoint_path}"
            )

        logger.info(f"Melhor época: {best_epoch}")
        logger.info(f"F1-Score registrado: {best_f1:.6f}")
        logger.info(f"Checkpoint selecionado: {checkpoint_path}")

        return str(checkpoint_path), best_epoch

    def load_backbone(self, checkpoint_path: str):

        logger.info(f"Carregando checkpoint: {checkpoint_path}")

        if self.cfg.model.name == "resnet152":
            base_encoder = models.resnet152
        else:
            base_encoder = models.resnet50

        model = moco_builder.MoCo(
            base_encoder,
            dim=self.cfg.model.feat_dim,
            K=65536,
            m=0.999,
            T=self.cfg.model.temp,
            mlp=(self.cfg.model.moco_version == 2),
            pretrained=self.cfg.model.pretrained
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        state_dict = checkpoint["model"]

        new_state_dict = {
            re.sub(r"^module\.", "", key): value
            for key, value in state_dict.items()
        }

        model.load_state_dict(new_state_dict)
        model = model.to(self.device)
        model.eval()

        return model

    def extract_features(self, model, loader):
        """Extrai características e rótulos de um conjunto de imagens."""

        features_list = []
        labels_list = []

        with torch.no_grad():
            for images, labels in tqdm(
                loader,
                desc="Extraindo características"
            ):
                if isinstance(images, (list, tuple)):
                    images = images[0]

                images = images.to(self.device)
                features = model.encoder_q(images)

                features_list.append(features.cpu().numpy())
                labels_list.append(labels.numpy())

        if not features_list:
            raise ValueError(
                "Nenhuma imagem foi encontrada para extrair características."
            )

        X = np.concatenate(features_list, axis=0)
        y = np.concatenate(labels_list, axis=0)

        return X, y

    def select_labeled_data(self, X_train, y_train, percent):
        """Seleciona a porcentagem de dados usada para treinar o SVM."""

        if percent == 100:
            logger.info("Utilizando 100% dos dados de treino.")
            return X_train, y_train

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=percent / 100,
            random_state=42
        )

        try:
            selected_indices, _ = next(
                splitter.split(X_train, y_train)
            )
        except ValueError as error:
            raise ValueError(
                f"Não foi possível selecionar {percent}% dos dados "
                "mantendo as espécies representadas."
            ) from error

        X_selected = X_train[selected_indices]
        y_selected = y_train[selected_indices]

        class_counts = np.unique(
            y_selected, return_counts=True
        )[1]

        if len(class_counts) != len(np.unique(y_train)):
            raise ValueError(
                "A seleção deixou alguma espécie sem exemplos."
            )

        if np.min(class_counts) < 3:
            raise ValueError(
                "Alguma espécie ficou com menos de 3 exemplos. "
                "O GridSearchCV com cv=3 não poderá ser executado."
            )

        logger.info(
            f"Dados selecionados: {len(y_selected)}/{len(y_train)} "
            f"({percent}%)."
        )

        return X_selected, y_selected

    def train_svm(self, X_train, y_train):
        """Treina o SVM e procura os melhores hiperparâmetros."""

        logger.info("Iniciando GridSearch do SVM...")

        param_grid = {
            "svc__C": self.cfg.eval.svm_c,
            "svc__kernel": self.cfg.eval.svm_kernel
        }

        pipeline = make_pipeline(
            StandardScaler(),
            SVC(probability=True, random_state=42)
        )

        clf = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            n_jobs=self.cfg.eval.n_jobs,
            scoring="f1_weighted",
            verbose=1
        )

        clf.fit(X_train, y_train)

        logger.info(f"Melhores parâmetros do SVM: {clf.best_params_}")

        return clf.best_estimator_

    def compute_metrics(self, clf, X_test, y_test, epoch_num, percent):

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted"
        )

        n_classes = len(clf.classes_)

        acc3 = (
            top_k_accuracy_score(
                y_test, y_prob, k=3, labels=clf.classes_
            )
            if n_classes > 3
            else 1.0
        )

        acc5 = (
            top_k_accuracy_score(
                y_test, y_prob, k=5, labels=clf.classes_
            )
            if n_classes > 5
            else 1.0
        )

        return {
            "epoch": epoch_num,
            "percent_labels": percent,
            "f1_score": f1,
            "top3_acc": acc3,
            "top5_acc": acc5
        }

    def save_results(self, metrics: dict):
        """Salva um arquivo CSV exclusivo para a porcentagem avaliada."""

        percent = metrics["percent_labels"]

        output_csv_file = os.path.join(
            self.results_path,
            f"results_labels_{percent}.csv"
        )

        pd.DataFrame([metrics]).to_csv(
            output_csv_file,
            index=False
        )

        logger.info(f"Resultados salvos em: {output_csv_file}")
        logger.info(f"Métricas: {metrics}")

    def run(self):
        """Extrai características uma vez e avalia todas as porcentagens."""

        # 1. Descobre e carrega a melhor época apenas uma vez
        checkpoint_path, best_epoch = self.find_best_checkpoint()
        model = self.load_backbone(checkpoint_path)

        # 2. Extrai características uma única vez
        logger.info("Extraindo características de TREINO...")
        X_train, y_train = self.extract_features(
            model, self.train_loader
        )

        logger.info("Extraindo características de TESTE...")
        X_test, y_test = self.extract_features(
            model, self.test_loader
        )

        # Libera a referência ao modelo; as características já foram extraídas
        del model

        # 3. Executa um experimento para cada porcentagem
        for percent in self.percent_labels:
            logger.info(
                f">>> AVALIANDO COM {percent}% DOS DADOS ROTULADOS <<<"
            )

            try:
                X_selected, y_selected = self.select_labeled_data(
                    X_train,
                    y_train,
                    percent
                )

                clf = self.train_svm(X_selected, y_selected)

                metrics = self.compute_metrics(
                    clf,
                    X_test,
                    y_test,
                    best_epoch,
                    percent
                )

                self.save_results(metrics)

            except Exception:
                logger.exception(
                    f"Falha na avaliação com {percent}% dos dados."
                )
                raise
