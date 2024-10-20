# AIF-C01 Study Notes

## Useful Links
- [Official Exam Guide](https://d1.awsstatic.com/training-and-certification/docs-ai-practitioner/AWS-Certified-AI-Practitioner_Exam-Guide.pdf)
- [Official Exam Prep - Amazon Skill Builder](https://explore.skillbuilder.aws/learn/course/internal/view/elearning/19554/exam-prep-standard-course-aws-certified-ai-practitioner-aif-c01)

## The Machine Learning (ML) Pipeline

The ML Pipeline is a systematic process used to build, train, and deploy machine learning models. It ensures that each stage, from identifying business goals to monitoring deployed models, is properly managed and optimized for performance. The typical steps in the pipeline are as follows:

**Steps:**
1. Identify Business Goal
2. Frame ML Problem
3. Collect Data
4. Pre-Process Data
5. Engineer Features
6. Train, Tune, Evaluate
7. Deploy
8. Monitor

---

### 1. Identify Business Goal
- **Description**: Define success criteria and align stakeholders to ensure the ML project meets business objectives.
- **Key Activities**:
  - Establish clear success metrics.
  - Align with stakeholders across the organization.

### 2. Frame the ML Problem
- **Description**: Define the ML problem, inputs, outputs, and metrics while considering feasibility and cost/benefit analysis.
- **Key Activities**:
  - Identify inputs, outputs, and performance metrics.
  - Start with simple model options (e.g., linear regression).
  - Perform cost-benefit analysis to evaluate feasibility.
  
- **Model Options**:
  - **AI/ML Hosted Service** (e.g., AWS Comprehend, Forecast, Personalize): No training required.
  - **Pre-trained Models** (e.g., Amazon Bedrock, SageMaker JumpStart): Fine-tune with your data.
  - **Fully Custom Model**: Build and train from scratch.

### 3. Collect Data
- **Description**: Collect and prepare the necessary data for training the model.
- **Key Activities**:
  - Identify data sources (e.g., databases, data lakes).
  - Ingest and label data.
  
- **Tools**:
  - **AWS Glue**: For ETL (Extract, Transform, Load) processes.
  - **SageMaker Ground Truth**: Human labeling of ambiguous data.

### 4. Pre-Process Data
- **Description**: Clean and prepare the data, ensuring it is suitable for training.
- **Key Activities**:
  - Perform exploratory data analysis (EDA).
  - Clean the data, removing duplicates, filling missing values, and anonymizing PII.
  - Split data into training (80%), validation (10%), and test (10%) sets.

- **Tools**:
  - **AWS Glue**: ETL service with built-in transformations.
  - **SageMaker Canvas**: Data import, preparation, and visualization.
  - **AWS Glue DataBrew**: Visual data preparation with quality rules.

### 5. Engineer Features
- **Description**: Select and engineer features that will enhance model performance.
- **Key Activities**:
  - Feature selection and creation based on domain knowledge.
  
- **Tools**:
  - **SageMaker Feature Store**: Store and manage features as a single source of truth.

### 6. Train, Tune, and Evaluate the Model
- **Description**: Train the model, tune hyperparameters, and evaluate performance.
- **Key Activities**:
  - Train the model iteratively and fine-tune parameters.
  - Tune hyperparameters (e.g., epoch, batch size, learning rate) and run experiments.
  - Evaluate the model using metrics and compare performance.

- **Parameters**:
  - **Inference Parameters** (supported by Amazon Bedrock):
    - **Randomness and Diversity**: Temperature, Top K, Top P.
    - **Length**: Response length, penalties, stop sequences.
  - **Model Training Parameters** (Hyperparameters):
    - **Epoch**: The number of iterations through the entire dataset.
    - **Batch Size**: Number of samples before updating model parameters.
    - **Learning Rate**: Controls how fast the model learns.

- **Tools**:
  - **SageMaker Training Jobs**: Manage training processes, specify training data, hyperparameters, and compute resources.
  - **SageMaker Experiments**: Track model runs and hyperparameter tuning.
  - **Automatic Model Tuning (AMT)**: Automatically tune hyperparameters using the specified metric.

### 7. Deploy the Model
- **Description**: Deploy the trained model to make predictions.
- **Key Activities**:
  - Choose between batch or real-time inference.
  - Select a hosting option (e.g., AWS Batch, ECS, EKS, Lambda, EC2).
  
- **Tools**:
  - **AWS API Gateway**: Expose model as an API endpoint for integration with applications.

### 8. Monitor the Model
- **Description**: Continuously monitor the model's performance and detect any data or concept drift.
- **Key Activities**:
  - Monitor model in real-time for data and concept drift.
  - Set alerts and automate corrective actions if performance degrades.

- **Types of Drift**:
  - **Data Drift**: When the input data changes from what the model was trained on.
  - **Concept Drift**: When the relationship between inputs and outputs shifts.

- **Tools**:
  - **SageMaker Model Monitor**: Schedule and monitor data drift, send results to CloudWatch, and automate corrective measures.

---

## MLOps and Automation

- **Description**: MLOps applies DevOps principles to machine learning, ensuring rapid experimentation, version control, continuous integration, and active performance monitoring.
  
- **Key Activities**:
  - Infrastructure as Code (IaC) for automating pipeline setup.
  - Rapid experimentation and versioning of models and pipelines.
  - Active performance monitoring for continuous model optimization.

- **Tools**:
  - **SageMaker Pipelines**: Automate end-to-end ML workflows.
  - **AWS CodeCommit**: Version control for models and code.
  - **SageMaker Model Building Pipelines**: Manage data processing, model training, and deployment.
  - **AWS Step Functions**: Orchestrate complex workflows.
  - **Amazon Managed Workflows for Apache Airflow (MWAA)**: For managing and orchestrating large-scale workflows.

---

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

####  Training Process

The typical SageMaker training process includes several key elements that help configure and manage the training jobs:

- **Training Data Locations**: Data is typically stored in Amazon S3 and accessed via S3 URLs.
- **ML Compute Instances**: SageMaker leverages EC2 instances (ECR instances) for scalable compute power.
- **Training Images**: The training process is run using Docker container images specifically designed for machine learning.
- **Hyperparameters**: Parameters that guide the learning process (e.g., learning rate, batch size).
- **S3 Output Bucket**: The trained model artifacts are stored in an S3 bucket for later use.

#### Features
- **SageMaker Feature Store**  
  - Central repository for storing, retrieving, and sharing machine learning features.
- **SageMaker Model Registry**  
  - Manages different versions of models and their metadata.
- **SageMaker Pipelines**  
  - Provides a workflow orchestration service for building, training, and deploying models with repeatable workflows.
- **SageMaker Model Monitor**  
  - Utilizes built-in rules to detect data drift or create custom rules, monitoring results can be sent to CloudWatch, and automates corrective measures.
- **SageMaker Ground Truth**  
  - Leverages actual humans for labeling data, ensuring high-quality training datasets.
- **SageMaker Canvas**  
  - A visual, no-code tool that allows users to build models based on their data with less technical expertise.
- **SageMaker JumpStart**  
  - Access to PreTrained Models and a hub for easily deploying machine learning solutions.
- **SageMaker Clarify**  
  - Tools to help detect biases and explain predictions to increase transparency.
- **SageMaker Role Manager**  
  - Manages permissions for SageMaker resources and services.
- **SageMaker Model Cards**  
  - Create transparent documentation for trained models.
- **SageMaker ML Lineage Tracking**  
  - Tracks the lineage of ML models to establish model governance, reproduce workflows, and maintain work history.
- **SageMaker Model Dashboard**  
  - Provides a unified interface to manage and monitor all model-related activities.
- **SageMaker Data Wrangler**  
  - Simplifies the process of data preparation for machine learning, enabling quick and easy data cleaning, transformation, and visualization.
- **SageMaker Experiments (Now called MLflow with Amazon SageMaker)**
  - Tracks, organizes, views, analyzes, and compares iterative ML experimentation.

---

### Amazon Bedrock

Amazon Bedrock is a fully managed, serverless service that provides access to high-performing foundation models (FMs) from leading AI companies through a single API. It is designed to facilitate the creation of generative AI applications, prioritizing security, privacy, and responsible AI.

Here's the updated section for Amazon Bedrock features, formatted as per your request:

#### Features
- **Model Choice**  
  - Access a variety of foundation models from AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI, and Amazon, allowing for optimal model selection based on specific application needs.
  - Amazon Titan Models
    - Exclusive to Amazon Bedrock, Amazon Titan models are pretrained, high-performing foundation models created by AWS, designed for a wide range of use cases with responsible AI practices.
- **Customization**  
  - Customize foundation models privately with your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG), enhancing the model's relevance and accuracy.
- **Bedrock Knowledge Bases**
  - Uses Retrieval Augmented Generation (RAG) to fetch data from private sources, delivering more relevant and accurate responses with full support for the RAG workflow—handling data ingestion, retrieval, and prompt augmentation.
  - Enhances outputs by integrating retrieval processes that pull relevant external knowledge into the generative model.
- **Bedrock Agents**  
  - Create agents capable of planning and executing multistep tasks using company systems and data sources—streamlining complex tasks such as customer inquiries and order processing.
- **Serverless**  
  - Eliminates the need for infrastructure management, simplifying the deployment and scaling of AI capabilities.
- **Security and Privacy Guardrails**  
  - Features robust controls to restrict AI outputs, preventing the generation of inappropriate content and restricting sensitive advice like financial recommendations, ensuring ethical and compliant AI usage.
- **PartyRock**
  - A hands-on AI app-building playground powered by Amazon Bedrock, where users can quickly build generative AI apps and experiment with models without writing code.

---

### AWS Glue 

AWS Glue is a fully managed, cloud-optimized ETL (Extract, Transform, Load) service that helps prepare and load data for analytics and AI models.

#### Features

- **AWS Glue ETL Service**
  - Cloud-based ETL service for data preparation with built-in transformations like dropping duplicates and filling missing values.
  - Example Workflow: AWS Kinesis Data Streams -> AWS Glue ETL Job -> CSV to Parquet -> S3 Data Lake.
- **AWS Glue Data Catalog**
  - Centralized repository for managing and monitoring ETL jobs, storing metadata (schemas, not data) with built-in classifiers.
- **AWS Glue Databrew**
  - Visual tool for data preparation, defining data quality rules, and saving transformations as "recipes."
- **AWS Glue Data Quality**
  - Detects anomalies and recommends data quality rules for ensuring clean, high-quality data for AI models.

## Tables

### Traditional ML vs Deep Learning

| Category          | Traditional ML                                      | Deep Learning                                 |
|-------------------|-----------------------------------------------------|-----------------------------------------------|
| **Task Complexity** | Well-defined tasks                                  | Complex tasks                                 |
| **Data Type**      | Structured / Labeled Data                           | Unstructured Data (Images, Video, Text)       |
| **Methodology**    | Solves problems through statistics and mathematics  | Utilizes neural networks                      |
| **Feature Handling** | Manually select and extract features                | Features are learned automatically by the model |
| **Cost**           | Less costly                                         | Higher costs due to computational demands     |


### Types of Machine Learning

| Learning Type         | Description                                          | Challenges                            | AWS Tools              | Common Use Cases                   |
|-----------------------|------------------------------------------------------|---------------------------------------|------------------------|------------------------------------|
| **Supervised Learning** | Uses pre-labeled data to train models.               | Labelling the data can be challenging. | SageMaker Ground Truth | Image classification, spam detection |
| **Unsupervised Learning** | Works with unlabeled data to find patterns.         | Requires methods to interpret patterns.| None specific          | Clustering, anomaly detection, LLMs for initial training phases |
| **Reinforcement Learning** | Learns through trial and error to maximize rewards. | Requires environment for agent interaction. | AWS DeepRacer      | Gaming, robotics, real-time decisions |


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


### Generative AI performance metrics
| Metric Name                                         | Explanation                                                             |
|-----------------------------------------------------|-------------------------------------------------------------------------|
| **Recall Oriented Understudy for Gisting Evaluation (ROUGE)** | Measures overlap between generated and reference text, good for summaries. |
| **Bilingual Evaluation Understudy (BLEU)**           | Evaluates translation accuracy by comparing n-grams between outputs and references. |
| **General Language Understanding Evaluation (GLUE)** | Assesses model performance on multiple natural language understanding tasks. |
| **Holistic Evaluation of Language Models (HELM)**    | Provides broad, task-specific evaluation of language model capabilities.  |
| **Massive Multitask Language Understanding (MMLU)**  | Tests model knowledge across a variety of domains and topics.            |
| **Beyond the Imitation Game Benchmark (BIG-bench)**  | Evaluates models on creative and difficult AI tasks not covered by standard benchmarks. |
| **Perplexity**                                       | Measures how well a model predicts the likelihood of the next token or word. |

