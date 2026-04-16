# Cloud-Federated-Learning-Project

Federated Learning (FL) application
where multiple distributed clients collaboratively train a deep learning model

## Round Orchestration

The worker/aggregator flow now supports an event-driven round transition:

- The first round starts immediately when the workers boot.
- The aggregator Lambda publishes the completion of each round to an SNS topic.
- Each worker receives its next-round signal through a dedicated SQS queue.
- Rounds 1 through 4 wait for the queue signal before starting.

Required environment variables:

- `ROUND_START_TOPIC_ARN` for the aggregator Lambda.
- `ROUND_START_QUEUE_URL` or `ROUND_START_QUEUE_NAME` for each worker.
