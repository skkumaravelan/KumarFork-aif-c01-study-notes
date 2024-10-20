# AIF-C01 Study Notes

## Useful Links
- [Official Exam Guide](https://d1.awsstatic.com/training-and-certification/docs-ai-practitioner/AWS-Certified-AI-Practitioner_Exam-Guide.pdf)
- [Official Exam Prep - Amazon Skill Builder](https://explore.skillbuilder.aws/learn/course/internal/view/elearning/19554/exam-prep-standard-course-aws-certified-ai-practitioner-aif-c01)


## Tables

### Types of Diffusion Models

| Diffusion Model Type | Description                                          | When to Use                                  | Examples                 |
|----------------------|------------------------------------------------------|----------------------------------------------|--------------------------|
| **Forward Diffusion**| Adds noise to data progressively.                    | Understanding data degradation processes.    | Experimental AI studies  |
| **Reverse Diffusion**| Reconstructs original data from noise.               | Creating detailed images from distorted inputs. | Image restoration tools |
| **Stable Diffusion** | Works in reduced latent space, not directly in pixels. | Generating detailed imagery with efficiency. | Midjourney, DALL-E       |

### Generative AI Architectures

| Generative AI Type          | Description                                                   | When to Use                                              | Examples                       |
|-----------------------------|---------------------------------------------------------------|----------------------------------------------------------|--------------------------------|
| **Generative Adversarial Network (GANs)** | Two-part model with generator and discriminator networks.     | Creating realistic images, videos, and voice synthesis.  | StyleGAN for photorealistic portraits |
| **Variational Autoencoders (VAEs)**       | Uses probability distributions to encode and decode data.      | Generating new data points with similar statistical properties. | Drug discovery, anomaly detection |
| **Transformers**                         | Leverages attention mechanisms to weigh the influence of different parts of the input data. | Natural language processing, image generation at scale.  | GPT-3 for text, DALL-E for images   |


### Amazon SageMaker Inference Methods

| Inference Type     | Deployed To         | Characteristics                                                         |
|--------------------|---------------------|-------------------------------------------------------------------------|
| **Batch**          | EC2                 | Cost-effective for infrequent, large jobs                               |
| **Asynchronous**   | EC2                 | Suitable for non-time-sensitive applications and large payload          |
| **Serverless**     | Lambda              | Intermittent traffic, periods of no traffic, auto-scaling out of the box|
| **Real-Time**      | EC2                 | Live predictions, sustained traffic, low latency, consistent performance|


### Types of Training Data for Machine Learning/AI

| Data Type       | AWS Data Source Example                | Actual Example                           |
|-----------------|----------------------------------------|------------------------------------------|
| **Structured**  | SQL data stored in RDS then moved to S3| Customer information in relational tables|
| **Semi-Structured** | Data in DynamoDB or DocumentDB then moved to S3 | JSON logs of user activity           |
| **Unstructured**| Objects and files stored directly in S3| Images, videos, and PDF documents        |
| **Time-Series** | Time-stamped data stored in S3         | IoT device data, stock market data       |


### Confusion Matrix Evaluation Metrics

| Metric Name              | Formula                                                        | Use Case                                                                                      |
|--------------------------|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **Accuracy**             | (True Positives + True Negatives) / Total                       | General correctness; e.g., evaluating overall performance of a transaction system.            |
| **Precision**            | True Positives / (True Positives + False Positives)             | Minimizing false positives; e.g., in spam detection to avoid marking legitimate emails as spam.|
| **Recall (TPR)**         | True Positives / (True Positives + False Negatives)             | Minimizing false negatives; e.g., in disease screening to avoid missing actual cases.         |
| **F1 Score**             | (2 * Precision * Recall) / (Precision + Recall)                 | Balancing precision and recall; useful in document classification where both metrics matter.  |
| **False Positive Rate (FPR)** | False Positives / (True Negatives + False Positives)        | Minimizing incorrect positive predictions; e.g., in security alarms to reduce false alerts.   |
| **Specificity (TNR)**    | True Negatives / (True Negatives + False Positives)             | Maximizing true negatives; e.g., in medical tests to correctly identify non-diseased patients.|
