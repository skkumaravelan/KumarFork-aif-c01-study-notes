# AIF-C01 Study Notes

## Useful Links
- [Official Exam Guide](https://d1.awsstatic.com/training-and-certification/docs-ai-practitioner/AWS-Certified-AI-Practitioner_Exam-Guide.pdf)
- [Official Exam Prep - Amazon Skill Builder](https://explore.skillbuilder.aws/learn/course/internal/view/elearning/19554/exam-prep-standard-course-aws-certified-ai-practitioner-aif-c01)

## Services 

### AWS Managed AI Services

AWS offers a range of managed AI services designed to be easily integrated into applications, providing powerful AI capabilities without the need for deep machine learning expertise. Here's an overview of key services:

#### Computer Vision
- **AWS Rekognition**
  - Facial comparison and analysis, object and text detection, content moderation. Good for _screening content_ such as identifying violent or inappropriate material.

#### Text and Document Analysis
- **AWS Textract** (OCR)
  - Converts scanned images to text, enabling digital content management.
- **Amazon Comprehend** (NLP)
  - Extracts key phrases, entities, and sentiment from text. Useful for analyzing _sentiment of feedback_ and detecting _PII data_.

#### Language AI
- **Amazon Lex**
  - Builds conversational interfaces for both voice and text, ideal for creating _chatbots_.
- **Amazon Transcribe**
  - Speech to text service, perfect for creating _captions for audio_.
- **Amazon Polly**
  - Converts text into lifelike speech, enhancing user engagement through voice.

#### Customer Experience
- **Amazon Kendra**
  - Provides intelligent document search capabilities.
- **Amazon Personalize**
  - Offers personalized product recommendations, akin to "you may also like" features.
- **Amazon Translate**
  - Facilitates language translation, supporting global user interaction.

#### Business Metrics
- **Amazon Forecast**
  - Predicts future points in time-series data, such as _inventory levels_.
- **Amazon Fraud Detection**
  - Detects potential fraud in various scenarios including online transactions and new account creations.

#### Enterprise Data Insights

- **Amazon Q Business**
  - Generative AI-powered assistant for answering questions, generating content, and automating tasks from enterprise data sources like S3, SharePoint, and Salesforce.

---

### Amazon SageMaker

Amazon SageMaker is an integrated machine learning service that enables developers and data scientists to build, train, and deploy machine learning models at scale. Users can create custom models from scratch or use and fine-tune existing ones through SageMaker JumpStart. This platform offers more control than high-level AI services like AWS Rekognition, allowing for detailed customization and optimization to meet specific project requirements.

#### Features
- **SageMaker Feature Store**: Central repository for storing, retrieving, and sharing machine learning features.
- **SageMaker Model Registry**: Manages different versions of models and their metadata.
- **SageMaker Pipelines**: Provides a workflow orchestration service for building, training, and deploying models with repeatable workflows.
- **SageMaker Model Monitor**: Utilizes built-in rules to _detect data drift_ or create custom rules, monitoring results can be sent to _CloudWatch_, and automates corrective measures.
- **SageMaker Ground Truth**: Leverages actual humans for labeling data, ensuring high-quality training datasets.
- **SageMaker Canvas**: A visual, no-code tool that allows users to build models based on their data with less technical expertise.
- **SageMaker JumpStart**: Access to _PreTrained Models_ and a hub for easily deploying machine learning solutions.
- **SageMaker Clarify**: Tools to help _detect biases_ and explain predictions to increase transparency.
- **SageMaker Role Manager**: Manages permissions for SageMaker resources and services.
- **SageMaker Model Cards**: Create transparent documentation for trained models.
- **SageMaker ML Lineage Tracking**: Tracks the lineage of ML models to _establish model governance_, _reproduce workflows_, and maintain work history.
- **SageMaker Model Dashboard**: Provides a unified interface to manage and monitor all model-related activities.
- **SageMaker Data Wrangler**: Simplifies the process of data preparation for machine learning, enabling quick and easy data cleaning, transformation, and visualization.

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
