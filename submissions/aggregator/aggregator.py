"""
Federated Averaging + Evaluation for Lambda Aggregator.

This file provides:
  - federated_average()  — weighted FedAvg on numpy state dicts
  - lenet5_forward()     — numpy-only LeNet-5 forward pass
  - evaluate_model()     — compute accuracy and loss on test set
  - load_test_data()     — load test images from S3 tar.gz
  - save_npz() / load_npz() — .npz serialization helpers

"""

import io
import os
import json
import tarfile
import logging

import boto3
import numpy as np
from PIL import Image
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aggregator")
logger.setLevel(logging.INFO)

# MNIST normalization constants (same as torchvision default)
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# S3 key prefixes
MODELS_PREFIX = "models/"
UPDATES_PREFIX = "updates/"
METRICS_PREFIX = "metrics/"

# S3 client (reused across warm Lambda invocations)
s3_client = boto3.client("s3", region_name="us-west-2")

# Test data cache (persists across warm invocations)
_cached_test_data = None

sns_client = boto3.client("sns", region_name = "us-west-2")
sns_topic = "arn:aws:sns:us-west-2:987081394736:1237312494-round-topic"

# ============================================================================
# FedAvg Aggregation (numpy)
# ============================================================================

def federated_average(client_updates):
    """Weighted Federated Averaging.

    Computes:
      global_weights[k] = SUM( (n_i / n_total) * client_weights_i[k] )

    Args:
        client_updates: list of (state_dict, num_samples) tuples.
            state_dict: dict of numpy arrays (keys = layer names)
            num_samples: int — how many training samples that client used

    Returns:
        dict of numpy arrays — the aggregated global model state_dict.

    Example:
        # After downloading all client .npz files:
        client_updates = [
            (load_npz(client_0_bytes), 600),
            (load_npz(client_1_bytes), 600),
            ...
        ]
        global_sd = federated_average(client_updates)
        save_npz(global_sd)  # upload to S3
    """
    if not client_updates:
        raise ValueError("No client updates to aggregate")

    total = sum(n for _, n in client_updates)
    if total == 0:
        raise ValueError("Total samples across all clients is 0")

    first = client_updates[0][0]
    result = {k: np.zeros_like(first[k], dtype=np.float64) for k in first}

    for sd, n in client_updates:
        w = n / total
        for k in result:
            result[k] += w * sd[k].astype(np.float64)

    return {k: v.astype(first[k].dtype) for k, v in result.items()}


# ============================================================================
# .npz Serialization Helpers
# ============================================================================

def save_npz(state_dict):
    """Serialize a numpy state_dict to .npz bytes.

    Args:
        state_dict: dict of numpy arrays (e.g., from federated_average())

    Returns:
        bytes — .npz content, ready for s3.put_object(Body=...)
    """
    buf = io.BytesIO()
    np.savez(buf, **state_dict)
    return buf.getvalue()


def load_npz(data):
    """Deserialize .npz bytes to a dict of numpy arrays.

    Args:
        data: bytes — raw .npz content from s3.get_object()["Body"].read()

    Returns:
        dict of numpy arrays (keys = layer names)
    """
    npz = np.load(io.BytesIO(data))
    return {k: npz[k] for k in npz.files}


# ============================================================================
# Numpy-only LeNet-5 Forward Pass (for evaluation in Lambda)
# ============================================================================

def _conv2d(x, w, b, pad=0):
    """2D convolution. x: (N,C,H,W), w: (F,C,kH,kW), b: (F,)."""
    if pad > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    N, C, H, W = x.shape
    F, _, kH, kW = w.shape
    oH, oW = H - kH + 1, W - kW + 1
    out = np.zeros((N, F, oH, oW))
    for f in range(F):
        for i in range(oH):
            for j in range(oW):
                out[:, f, i, j] = np.sum(
                    x[:, :, i:i+kH, j:j+kW] * w[f], axis=(1, 2, 3)
                ) + b[f]
    return out


def _relu(x):
    return np.maximum(0, x)


def _max_pool2d(x, size=2):
    N, C, H, W = x.shape
    oH, oW = H // size, W // size
    out = np.zeros((N, C, oH, oW))
    for i in range(oH):
        for j in range(oW):
            out[:, :, i, j] = x[:, :,
                                i*size:(i+1)*size,
                                j*size:(j+1)*size].max(axis=(2, 3))
    return out


def _linear(x, w, b):
    return x @ w.T + b


def lenet5_forward(sd, images):
    """Forward pass through LeNet-5 using numpy arrays only.

    Args:
        sd: dict of numpy arrays (model state_dict with keys:
            conv1.weight, conv1.bias, conv2.weight, conv2.bias,
            fc1.weight, fc1.bias, fc2.weight, fc2.bias,
            fc3.weight, fc3.bias)
        images: numpy array of shape (N, 1, 28, 28) — preprocessed MNIST images

    Returns:
        numpy array of shape (N, 10) — logits (unnormalized class scores)
    """
    x = images
    x = _max_pool2d(_relu(_conv2d(x, sd['conv1.weight'], sd['conv1.bias'], pad=2)), 2)
    x = _max_pool2d(_relu(_conv2d(x, sd['conv2.weight'], sd['conv2.bias'])), 2)
    x = x.reshape(x.shape[0], -1)
    x = _relu(_linear(x, sd['fc1.weight'], sd['fc1.bias']))
    x = _relu(_linear(x, sd['fc2.weight'], sd['fc2.bias']))
    x = _linear(x, sd['fc3.weight'], sd['fc3.bias'])
    return x


# ============================================================================
# Evaluation Helpers
# ============================================================================

def cross_entropy_loss(logits, labels):
    """Compute cross-entropy loss (numpy).

    Args:
        logits: (N, C) float array — raw model output
        labels: (N,) int array — ground truth class indices

    Returns:
        float — mean cross-entropy loss
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


def transform_image(img):
    """Convert a PIL image to normalized numpy array.

    Args:
        img: PIL Image

    Returns:
        numpy array of shape (1, 28, 28) — normalized grayscale image
    """
    img = img.convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    return ((arr - MNIST_MEAN) / MNIST_STD).reshape(1, 28, 28)


def load_test_data(global_bucket):
    """Download test set from S3 and return as numpy arrays.

    Caches in memory across warm Lambda invocations.
    Test data (labels.csv, archives/test.tar.gz) is in global-bucket.

    Args:
        global_bucket: global-bucket name (e.g., "{ASU_ID}-global-bucket")

    Returns:
        (images, labels) tuple:
            images: numpy array (N, 1, 28, 28) float32
            labels: numpy array (N,) int64
    """
    global _cached_test_data
    if _cached_test_data is not None:
        return _cached_test_data

    logger.info("Loading test set from S3 (one-time cache) ...")

    # Labels
    resp = s3_client.get_object(Bucket=global_bucket, Key="labels.csv")
    content = resp["Body"].read().decode()
    labels_map = {}
    for line in content.strip().split("\n")[1:]:
        parts = line.strip().split(",")
        labels_map[parts[0]] = int(parts[2])

    # Test images
    resp = s3_client.get_object(Bucket=global_bucket, Key="archives/test.tar.gz")
    tar_bytes = resp["Body"].read()

    images = []
    targets = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".png"):
                continue
            filename = os.path.basename(member.name)
            if filename not in labels_map:
                continue
            f = tar.extractfile(member)
            img = Image.open(io.BytesIO(f.read()))
            images.append(transform_image(img))
            targets.append(labels_map[filename])

    images_np = np.concatenate(images, axis=0).reshape(len(images), 1, 28, 28)
    labels_np = np.array(targets, dtype=np.int64)
    _cached_test_data = (images_np, labels_np)
    logger.info(f"Test set cached: {len(images)} images")
    return _cached_test_data


def evaluate_model(sd, test_images, test_labels):
    """Evaluate a model state_dict on the test set.

    Args:
        sd: dict of numpy arrays (model state_dict)
        test_images: numpy array (N, 1, 28, 28)
        test_labels: numpy array (N,) int64

    Returns:
        dict with keys:
            "accuracy": float (0.0-1.0)
            "loss": float (cross-entropy)
            "total": int (number of test samples)
            "correct": int (number correct)

    Example:
        images, labels = load_test_data(global_bucket)
        result = evaluate_model(global_sd, images, labels)
        # result["accuracy"] → 0.9729
        # result["loss"] → 0.0862
    """
    logits = lenet5_forward(sd, test_images)
    preds = logits.argmax(axis=1)
    acc = float((preds == test_labels).mean())
    loss = cross_entropy_loss(logits, test_labels)
    return {
        "accuracy": acc,
        "loss": loss,
        "total": len(test_labels),
        "correct": int((preds == test_labels).sum()),
    }

# Message to signal workers to start the nxt round
def signal_round_start(nextRound, globalBucket, modelKey):

    message = {
        "event_type": "round_start",
        "round": nextRound,
        "global_bucket": globalBucket,
        "model_key": modelKey,
    }

    sns_client.publish(
        TopicArn = sns_topic,
        Subject = f"fl-round-{nextRound}",
        Message = json.dumps(message),
    )
    logger.info(f"SIgnal sent for round {nextRound}: {message}")


# ============================================================================
# Lambda handler
# ============================================================================

def handler(event, context):
    """Lambda handler — triggered by S3 event on updates/*.npz.

    This function is invoked each time a worker uploads a .npz model
    file to the local-bucket. You need to:

    1. Parse the S3 event to get bucket name and object key
    2. Extract round_id from the key
    3. List all .npz files for this round to check if all clients reported
    4. If not all clients → return early
    5. If all clients reported → aggregate:
       a. Download each client's .npz model weights
       b. Call federated_average() to get the aggregated model
          (use equal weighting: 1 per client)
       c. Upload aggregated model to global-bucket
       d. Evaluate on test set
       e. Write metrics/round_{R}.json to global-bucket

    Args:
        event: S3 event dict (see S3 EVENT FORMAT in docstring above)
        context: Lambda context object (not used)

    Returns:
        dict with statusCode and body
    """
    clientCount = 10

    # Checking S3 event
    record = event["Records"][0]
    localBucket = record["s3"]["bucket"]["name"]
    key = record["s3"]["object"]["key"]

    logger.info(f"S3 Event triggered by - {localBucket} : {key}")

    # Taking the round number from key
    filename = key.split("/")[-1]
    parts = filename.replace(".npz", "").split("_")
    round = int(parts[3])

    # Taking the bucket names
    asuId = localBucket.replace("-local-bucket", "")
    globalBucket = f"{asuId}-global-bucket"
    logger.info(f"Round - {round} !!!")

    nextRound = round + 1
    nextKey = f"models/global_model_round_{nextRound}.npz"

    try:
        s3_client.head_object(Bucket = globalBucket, Key=  nextKey)
        logger.info(f"Round - {round}: {nextKey} already exists !!!")
        return {
            "statusCode": 200,
            "body": json.dumps({"round": round, "status": "already_aggregated"}),
        }
    except ClientError as exc:
        error_code = str(exc.response.get("Error", {}).get("Code", ""))
        if error_code not in {"404", "NoSuchKey", "NotFound"}:
            raise

    # Getting all the .npz files
    prefix = f"updates/local_model_round_{round}_worker_"
    response = s3_client.list_objects_v2(Bucket = localBucket, Prefix = prefix)

    if "Contents" not in response:
        logger.info("No changes found !!!")
        return {"statusCode": 200, "body": "No updates"}

    updateKeys = [
        o["Key"]
        for o in response["Contents"]
        if o["Key"].endswith(".npz")
    ]
    logger.info(f"Round - {round}: {len(updateKeys)}/{clientCount} updated successfully !!!")

    if len(updateKeys) < clientCount:
        return {
            "statusCode": 200,
            "body": f"Waiting - {len(updateKeys)}/{clientCount} for updates",
        }

    logger.info(f"Round - {round}: all {clientCount} clients reported, now aggregating ...")

    # Downloading the model weights
    clientUpdates = []
    for key in updateKeys:
        response = s3_client.get_object(Bucket = localBucket, Key = key)
        sd = load_npz(response["Body"].read())
        clientUpdates.append((sd, 1))
    logger.info(f"{len(clientUpdates)} models are downloaded !!!")

    global_sd = federated_average(clientUpdates)

    # Uploading the aggreagated model to my global bucket for the nxt round
    s3_client.put_object(
        Bucket = globalBucket,
        Key = nextKey,
        Body = save_npz(global_sd),
    )
    logger.info(f"Uploaded global model - {nextKey} !!!")

    # Notify workers that the next round can start.
    if nextRound < 5:
        signal_round_start(nextRound, globalBucket, nextKey)
    else:
        logger.info("Final round is completed !!!")

    # Evaluating the model
    test_images, test_labels = load_test_data(globalBucket)
    result = evaluate_model(global_sd, test_images, test_labels)
    logger.info(
        f"Round - {round} : Accuracy = {result['accuracy']:.4f},\n"
        f"Round - {round} : Loss = {result['loss']:.4f}"
    )

    metrics = {
        "round": round,
        "accuracy": float(np.round(result["accuracy"], 4)),
        "loss": float(np.round(result["loss"], 4)),
    }
    metricsKey = f"metrics/round_{round}.json"
    s3_client.put_object(
        Bucket = globalBucket,
        Key = metricsKey,
        Body = json.dumps(metrics),
        ContentType = "application/json",
    )
    logger.info(f"the metrics: {metricsKey} → {metrics}")

    return {
        "statusCode": 200,
        "body": json.dumps(metrics)
    }