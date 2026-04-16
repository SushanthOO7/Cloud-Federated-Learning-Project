"""
LeNet-5 Classifier for Federated Learning on MNIST.
  - LeNet5 model class (PyTorch)
  - Serialization helpers: state_dict <-> .npz bytes
  - Model creation and loading utilities
"""

import io
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os
import boto3
import logging
import time
import glob
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)

NUM_CLASSES = 10

class LeNet5(nn.Module):
    """LeNet-5 for MNIST classification.

    Input:  (batch, 1, 28, 28)
    Output: (batch, 10)
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_model(num_classes=NUM_CLASSES):
    """Create a fresh LeNet-5 model with random weights."""
    return LeNet5(num_classes=num_classes)


def load_model(state_dict, num_classes=NUM_CLASSES):
    """Create a LeNet-5 model and load the given state_dict.

    Args:
        state_dict: OrderedDict of PyTorch tensors (from deserialize_state_dict).
        num_classes: Number of output classes (default 10).

    Returns:
        LeNet5 model with loaded weights, ready for training or inference.
    """
    model = LeNet5(num_classes=num_classes)
    model.load_state_dict(state_dict)
    return model


def serialize_state_dict(state_dict):
    """Convert a PyTorch state_dict to .npz bytes for S3 upload.

    Args:
        state_dict: OrderedDict from model.state_dict()
                    (keys are layer names, values are torch.Tensor)

    Returns:
        bytes — the .npz archive contents, ready for s3.put_object(Body=...)

    Example:
        sd = model.state_dict()
        data = serialize_state_dict(sd)
        s3.put_object(Bucket=bucket, Key="models/global_model_round_0.npz", Body=data)
    """
    buf = io.BytesIO()
    np.savez(buf, **{k: v.cpu().numpy() for k, v in state_dict.items()})
    return buf.getvalue()


def deserialize_state_dict(data):
    """Convert .npz bytes from S3 to a PyTorch state_dict.

    Args:
        data: bytes — raw .npz file content from s3.get_object()["Body"].read()

    Returns:
        OrderedDict of torch.Tensor — ready for model.load_state_dict() or load_model()

    Example:
        resp = s3.get_object(Bucket=bucket, Key="models/global_model_round_0.npz")
        sd = deserialize_state_dict(resp["Body"].read())
        model = load_model(sd)
    """
    npz = np.load(io.BytesIO(data))
    return OrderedDict({k: torch.from_numpy(npz[k]) for k in npz.files})


# ============================================================================
# worker service execution part
# ============================================================================

def train_local(model, dataloader, lr, epochs):
    """Train the model locally and return metrics.

    Args:
        model: LeNet5 model to train
        dataloader: PyTorch DataLoader with training data
        lr: learning rate
        epochs: number of local training epochs

    Returns:
        dict with keys:
            "train_loss": float — average training loss
            "train_accuracy": float — average training accuracy
            "num_samples": int — number of training samples
    """
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()

    totalLoss = 0.0
    totalCorrect = 0
    totalSamples = 0

    for epoch in range(epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            batchSize = labels.size(0)
            totalLoss += loss.item() * batchSize
            totalCorrect += (outputs.argmax(dim = 1) == labels).sum().item()
            totalSamples += batchSize

    dataset_size = len(dataloader.dataset)
    return {
        "trainings_loss": totalLoss / totalSamples,
        "trainings_accuracy": totalCorrect / totalSamples,
        "number_of_samples": dataset_size,
    }

# For loading the MNIST partition
def loading_data(partitionId, asuId):
    dataDirectory = f"/home/ubuntu/fl-client/data_cache/client-{partitionId}"

    s3client = boto3.client("s3")
    globalBucket = f"{asuId}-global-bucket"

    data = None
    for i in range(120):
        try:
            response = s3client.get_object(Bucket = globalBucket, Key = "labels.csv")
            data = response["Body"].read().decode()
            break
        except Exception:
            time.sleep(1)

    alllabels = {}
    for line in data.strip().split("\n")[1:]:
        parts = line.strip().split(",")
        alllabels[parts[0]] = int(parts[2])

    # Loading th eimages
    images = []
    targets = []
    image_paths = []

    for i in range(120):
        image_paths = sorted(glob.glob(os.path.join(dataDirectory, "*.png")))
        if image_paths:
            break
        time.sleep(1)

    for path in image_paths:
        filename = os.path.basename(path)

        if filename not in alllabels:
            continue

        img = Image.open(path).convert("L").resize((28, 28))

        MNIST_mean = 0.1307
        MNIST_standard_deviation = 0.3081
        array = np.array(img, dtype=np.float32) / 255.0
        array = (array - MNIST_mean) / MNIST_standard_deviation

        images.append(array.reshape(1, 28, 28))
        targets.append(alllabels[filename])

    if not images:
        raise RuntimeError(
            f"No training images found for partition {partitionId}\n"
            f"Checked: {dataDirectory}"
        )

    imagesTensor = torch.tensor(np.array(images), dtype = torch.float32)
    labelsTensor = torch.tensor(targets, dtype = torch.long)

    batchSize = 96
    dataset = TensorDataset(imagesTensor, labelsTensor)
    dataloader = DataLoader(dataset, batch_size = batchSize, shuffle = True)
    logging.info(
        f"{len(dataset)} samples are loaded from this partition {partitionId} from {dataDirectory}"
    )

    return dataloader, len(dataset)

# Extracting SQS message
def get_message(body):
    payload = json.loads(body)
    if isinstance(payload, dict) and "Message" in payload:
        try:
            return json.loads(payload["Message"])
        except Exception:
            return payload["Message"]
    return payload

# Forming the queue url for each worker
def get_round_queue_url(asuId, partitionId):
    account_id = "987081394736"
    region = "us-west-2"
    queue_name = f"{asuId}-round-queue-{partitionId}"
    return f"https://sqs.{region}.amazonaws.com/{account_id}/{queue_name}"

# Waiting until the worker gets the next rounds flag to start
def wait_for_round_start(wantedRound, sqsclient, queue_url):

    logging.info(f"Waiting for signal of round {wantedRound} !!!")

    while True:
        try:
            response = sqsclient.receive_message(
                QueueUrl = queue_url,
                MaxNumberOfMessages = 1,
                WaitTimeSeconds = 20,
                VisibilityTimeout = 30,
            )

        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))

            if code == "AccessDenied":
                raise RuntimeError(
                    "CHeck for permissions !!!"
                ) from exc
            raise

        messages = response.get("Messages", [])
        if not messages:
            continue
        message = messages[0]

        try:
            payload = get_message(message["Body"])
            message_round = int(payload.get("round"))

        except Exception as exc:
            logging.warning(f"Discarding unwanted message: {exc}")
            sqsclient.delete_message(QueueUrl = queue_url, ReceiptHandle = message["ReceiptHandle"])
            continue

        if message_round != wantedRound:
            logging.info(f"Ignoring this message for round {message_round} but expected is {wantedRound}.")
            sqsclient.delete_message(QueueUrl = queue_url, ReceiptHandle = message["ReceiptHandle"])
            continue

        sqsclient.delete_message(QueueUrl = queue_url, ReceiptHandle = message["ReceiptHandle"])

        logging.info(f"Successfully round signal received for round {wantedRound}.")

        return payload

def worker_main():
    """FL worker main loop.

    This function runs on each EC2 instance. You need to:

    1. Read PARTITION_ID and ASU_ID from environment variables
    2. Set up boto3 S3 client
    3. Load your MNIST partition from local disk
       (data is at /home/ubuntu/fl-worker/data_cache/client-{PARTITION_ID}/)
    4. For each round (0 to 5-1):
       a. Poll S3 for global model: models/global_model_round_{R}.npz
       b. Download and deserialize the global model
       c. Train locally on your partition
       d. Upload trained model .npz to local-bucket (TRIGGERS Lambda)
          Key: updates/local_model_round_{R}_worker_{C}.npz
    5. Exit after all rounds complete
    """
    partitionId = int(os.environ.get("PARTITION_ID", "0"))
    asuId = os.environ.get("ASU_ID", "1237312494")

    s3client = boto3.client("s3", region_name = "us-west-2")
    sqsclient = boto3.client("sqs", region_name = "us-west-2")
    globalBucket = f"{asuId}-global-bucket"
    localBucket = f"{asuId}-local-bucket"
    roundStartQueueUrl = get_round_queue_url(asuId, partitionId)

    logging.info(f"The Worker {partitionId} is starting !!!")

    # Taking the local data
    dataloader = None

    for i in range(30):
        try:
            dataloader, number_of_samples = loading_data(partitionId, asuId)
            break
        except Exception as e:
            logging.warning(f"Data loading failed due to {e}, retrying !!!")
            time.sleep(2)

    if dataloader is None:
        raise RuntimeError(f"Worker {partitionId} failed to load data after retries")

    # Main FL loop
    for round in range(5):
        if round > 0:
            wait_for_round_start(round, sqsclient, roundStartQueueUrl)

        globalKey = f"models/global_model_round_{round}.npz"

        logging.info(f"Round {round}: fetching global model {globalKey} !!!")

        # AFter aggragator saves the global model
        model_data = None
        for i in range(30):
            try:
                response = s3client.get_object(Bucket = globalBucket, Key = globalKey)
                model_data = response["Body"].read()
                break

            except Exception as e:
                logging.warning(f"Round {round} : global model not ready as: {e}, retrying !!!")
                time.sleep(1)

        if model_data is None:
            raise RuntimeError(f"Round {round} : failed to download {globalKey}")

        # Loading the global model
        stateDict = deserialize_state_dict(model_data)
        model = load_model(stateDict)
        logging.info(f"Round {round}: global model loaded successfully !!!")

        # Traing the model
        logging.info("Training the model !!!")
        metrics = train_local(model, dataloader, 0.001, 5)
        logging.info(
            f"Round {round} - Loss = {metrics['trainings_loss']:.4f},\n"
            f"Round {round} - Accuracy = {metrics['trainings_accuracy']:.4f},\n"
            f"Round {round} - Samples = {metrics['number_of_samples']}\n"
        )

        # uploading the local mdal to my local bucket
        localKey = f"updates/local_model_round_{round}_worker_{partitionId}.npz"
        s3client.put_object(
            Bucket = localBucket,
            Key = localKey,
            Body = serialize_state_dict(model.state_dict()),
        )
        logging.info(f"Round {round} - SUccessfully uploaded the local model to {localKey}")

    logging.info(f"Worker {partitionId} completed all the 5 rounds.")

if __name__ == "__main__":
    worker_main()