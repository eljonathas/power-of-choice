import time
from typing import Dict, Tuple, List

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Cliente federado
from flwr.client import ClientApp, NumPyClient

# Estratégia FedAvg
from flwr.server.strategy import FedAvg

# Componentes do servidor
from flwr.server import ServerApp, ServerConfig, ServerAppComponents

# Execução da simulação
from flwr.simulation import run_simulation

# Silenciar logs do TensorFlow
tf.get_logger().setLevel("ERROR")


# 1. Definição do modelo
def create_cnn() -> tf.keras.Model:
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer((32, 32, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# 2. Particionamento non-IID (2 classes por cliente)
def load_non_iid_cifar10(
    num_clients: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    y_train = y_train.flatten()

    class_indices = {i: np.where(y_train == i)[0] for i in range(10)}
    shards_per_client = 2
    clients_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for cid in range(num_clients):
        classes = [(cid * shards_per_client + i) % 10 for i in range(shards_per_client)]
        idx = np.concatenate([class_indices[c] for c in classes])
        np.random.shuffle(idx)
        clients_data[cid] = (x_train[idx], y_train[idx])

    return clients_data


# 3. Cliente Flower usando API NumPyClient
class CifarClient(NumPyClient):
    def __init__(self, cid: int, train_data: Tuple[np.ndarray, np.ndarray]):
        self.cid = cid
        self.model = create_cnn()
        self.x_train, self.y_train = train_data
        # Mantém um conjunto de validação local
        num_val = int(0.2 * len(self.x_train))
        self.x_val = self.x_train[:num_val]
        self.y_val = self.y_train[:num_val]
        self.x_train = self.x_train[num_val:]
        self.y_train = self.y_train[num_val:]

    def get_parameters(self, config) -> List[np.ndarray]:
        # Retorna pesos do modelo como listas de ndarrays
        return self.model.get_weights()

    def fit(
        self, parameters: List[np.ndarray], config
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        # Atualiza pesos e treina
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=1,
            batch_size=32,
            verbose=0,
        )
        # Avalia localmente
        loss, accuracy = self.model.evaluate(
            self.x_val, self.y_val, verbose=0
        )
        # Retorna novos parâmetros, número de exemplos e métricas
        return self.model.get_weights(), len(self.x_train), {
            "loss": float(loss),
            "accuracy": float(accuracy),
        }

    def evaluate(
        self, parameters: List[np.ndarray], config
    ) -> Tuple[float, int, Dict[str, float]]:
        # Avaliação no conjunto de validação
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(
            self.x_val, self.y_val, verbose=0
        )
        return float(loss), len(self.x_val), {"accuracy": float(accuracy)}


# 4. Função de avaliação centralizada no servidor
def get_evaluate_fn():
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    y_test = y_test.flatten()
    x_val, y_val = x_test[:5000], y_test[:5000]

    def evaluate(
        server_round: int, parameters: List[np.ndarray], config
    ) -> Tuple[float, Dict[str, float]]:
        model = create_cnn()
        model.set_weights(parameters)
        loss, acc = model.evaluate(x_val, y_val, verbose=0)
        print(f"Round {server_round}: loss={loss:.4f}, acc={acc:.4f}")
        return float(loss), {"accuracy": float(acc)}

    return evaluate


# 5. Estratégia Power-of-Choice (override configure_fit)
class PowerOfChoice(FedAvg):
    def __init__(
        self,
        fraction_fit: float,
        k: int,
        min_available_clients: int,
        evaluate_fn,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
        )
        self.k = k

    def configure_fit(self, server_round, parameters, client_manager):
        # Seleciona k clientes com menor perda na última avaliação global
        losses: Dict[int, float] = {}
        for cid in client_manager.all().keys():
            loss, _ = self.evaluate_fn(server_round, parameters, {})
            losses[cid] = loss
        selected_ids = sorted(losses, key=losses.get)[: self.k]

        # Obter FitIns do FedAvg e filtrar
        sample = super().configure_fit(server_round, parameters, client_manager)
        filtered = [(c, ins) for c, ins in sample if c.cid in selected_ids]
        return filtered


# 6. Execução da simulação
if __name__ == "__main__":
    NUM_CLIENTS = 10
    NUM_ROUNDS = 10
    K = 5

    # Dados
    clients_data = load_non_iid_cifar10(NUM_CLIENTS)

    # Map para context.node_id -> cid sequencial
    node_id_map: Dict[str, int] = {}

    def client_fn(context):
        node = context.node_id
        if node not in node_id_map:
            node_id_map[node] = len(node_id_map)
        cid = node_id_map[node]
        return CifarClient(cid, clients_data[cid]).to_client()

    client_app = ClientApp(client_fn=client_fn)

    # Servidor FedAvg
    def server_fn_baseline(context) -> ServerAppComponents:
        strategy = FedAvg(
            fraction_fit=0.5,
            min_available_clients=NUM_CLIENTS,
            evaluate_fn=get_evaluate_fn(),
        )
        config = ServerConfig(num_rounds=NUM_ROUNDS)
        return ServerAppComponents(config=config, strategy=strategy)

    server_app_baseline = ServerApp(server_fn=server_fn_baseline)

    # Servidor Power-of-Choice
    def server_fn_poc(context) -> ServerAppComponents:
        strategy = PowerOfChoice(
            fraction_fit=1.0,
            k=K,
            min_available_clients=NUM_CLIENTS,
            evaluate_fn=get_evaluate_fn(),
        )
        config = ServerConfig(num_rounds=NUM_ROUNDS)
        return ServerAppComponents(config=config, strategy=strategy)

    server_app_poc = ServerApp(server_fn=server_fn_poc)

    backend_config = {"client_resources": {"num_cpus": 1}}

    # Simulação FedAvg
    start = time.time()
    history_fedavg = run_simulation(
        server_app=server_app_baseline,
        client_app=client_app,
        num_supernodes=1,  # reduz concorrência para evitar OOM
        backend_config=backend_config,
    )
    fedavg_time = time.time() - start

    # Simulação Power-of-Choice
    start = time.time()
    history_poc = run_simulation(
        server_app=server_app_poc,
        client_app=client_app,
        num_supernodes=1,  # reduz concorrência para evitar OOM
        backend_config=backend_config,
    )
    poc_time = time.time() - start

    # Extrair métricas
    rounds = [r for r, _ in history_fedavg.metrics_centralized["evaluate"]]
    acc_fedavg = [m.metrics["accuracy"] for _, m in history_fedavg.metrics_centralized["evaluate"]]
    acc_poc = [m.metrics["accuracy"] for _, m in history_poc.metrics_centralized["evaluate"]]

    loss_fedavg = [m.loss for _, m in history_fedavg.metrics_centralized["evaluate"]]
    loss_poc = [m.loss for _, m in history_poc.metrics_centralized["evaluate"]]

    # Plotagem
    plt.figure()
    plt.plot(rounds, acc_fedavg, "b-o", label="FedAvg")
    plt.plot(rounds, acc_poc, "r-o", label="Power-of-Choice")
    plt.title("Acurácia Global por Rodada")
    plt.xlabel("Rodada")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(rounds, loss_fedavg, "b-o", label="FedAvg")
    plt.plot(rounds, loss_poc, "r-o", label="Power-of-Choice")
    plt.title("Perda Global por Rodada")
    plt.xlabel("Rodada")
    plt.ylabel("Perda")
    plt.legend()
    plt.show()

    plt.figure()
    plt.bar(["FedAvg", "Power-of-Choice"], [fedavg_time, poc_time])
    plt.title("Tempo Total de Simulação")
    plt.ylabel("Segundos")
    plt.show()

    plt.figure()
    plt.plot(rounds, loss_fedavg, "b-o", label="FedAvg")
    plt.plot(rounds, loss_poc, "r-o", label="Power-of-Choice")
    plt.title("Perda Global por Rodada")
    plt.xlabel("Rodada")
    plt.ylabel("Perda")
    plt.legend()
    plt.show()

    plt.figure()
    plt.bar(["FedAvg", "Power-of-Choice"], [fedavg_time, poc_time])
    plt.title("Tempo Total de Simulação")
    plt.ylabel("Segundos")
    plt.show()
