# AWS Certified AI Practitioner (AIF-C01) Exam Study Notes

## Useful Links
- [Official Exam Guide](https://d1.awsstatic.com/training-and-certification/docs-ai-practitioner/AWS-Certified-AI-Practitioner_Exam-Guide.pdf)
- [Official Exam Prep - Amazon Skill Builder](https://explore.skillbuilder.aws/learn/course/internal/view/elearning/19554/exam-prep-standard-course-aws-certified-ai-practitioner-aif-c01) - recommend watching in 1.5x
- [Official Exam Practice Questions](https://explore.skillbuilder.aws/learn/course/internal/view/elearning/19790/exam-prep-official-practice-question-set-aws-certified-ai-practitioner-aif-c01-english) - About the level of real exam

---

# 1. Fundamentals of AI and ML

## Overview

### General AI
- **Definition**: Refers to a broader concept of artificial intelligence, aiming to build systems that can perform any intellectual task that a human can. It's often used to describe long-term goals of creating highly autonomous, flexible AI systems.
- **Key Concept**: AI with a broad ability to perform multiple tasks across domains, unlike specialized AI systems.
- **Examples**: **Expert Systems**, **Rules Engines** (e.g., MYCIN), which use predefined logic and rules to make decisions.

### Machine Learning (ML)
- **Definition**: A subset of AI that enables systems to learn from data and make predictions or decisions without explicit programming or rules.
- **Key Concept**: AI that operates through data patterns and training, rather than explicitly coded instructions.
- **Examples**: **Linear Regression**, **Decision Trees**, **Support Vector Machines**—supervised learning algorithms used for predictive tasks.

### Deep Learning
- **Definition**: A subset of machine learning that uses neural networks with multiple layers (hence "deep") to model complex patterns in data.
- **Key Concept**: Effective for tasks like image recognition, natural language processing, and autonomous systems.
- **Examples**: **Convolutional Neural Networks (CNNs)** for image processing, **Recurrent Neural Networks (RNNs)** for sequential data like time-series or language models.

#### Deep Learning Architectures
- **CNN (Convolutional Neural Networks)**: 
  - **Use Cases**: Image classification, object detection, computer vision tasks
- **RNN (Recurrent Neural Networks)**: 
  - **Use Cases**: Sequential data processing, time series analysis, natural language processing
- **GAN (Generative Adversarial Networks)**: 
  - **Use Cases**: Image generation, data augmentation, style transfer

### Supervised Learning Types
Learning from **labeled** data where input-output pairs are known. Models predict outputs for new inputs based on training.

| Type | Description | Use Case | Input Data Type |
|------|-------------|----------|-----------------|
| Classification | Predicts categories/classes | Email spam detection | Labeled |
| Binary Classification | Two-class classification | Spam vs not spam email | Labeled |
| Multi-class Classification | Multiple categories | Image recognition (cats, dogs) | Labeled |
| Multi-label Classification | Multiple labels per input | Movie genre tagging | Labeled |
| Regression | Predicts continuous values | House price prediction | Labeled |
| Linear Regression | Fits straight-line relationships | Sales vs advertisement spend | Labeled |
| Polynomial Regression | Models curved relationships | Growth curve modeling | Labeled |
| Multiple Regression | Uses multiple inputs to predict output | Stock price prediction | Labeled |

### Unsupervised Learning Types
Learns from **unlabeled** data to find hidden structures or patterns.

| Type | Description | Use Case | Input Data Type |
|------|-------------|----------|-----------------|
| Clustering | Groups similar data points | Customer segmentation | Unlabeled |
| Dimensionality Reduction | Reduces features while retaining info | Visualizing high-dimensional data | Unlabeled |
| Association Rule Learning | Finds item co-occurrences | Market basket analysis | Unlabeled |
| Anomaly Detection | Detects outliers/unusual data | Fraud detection | Unlabeled |

### Semi-Supervised Learning Types
Combines a small amount of **labeled** data with large amounts of **unlabeled** data for training.

| Type | Description | Use Case | Input Data Type |
|------|-------------|----------|-----------------|
| Self-training | Model labels unlabeled data iteratively | Image labeling | Mostly unlabeled + some labeled |
| Co-training | Different models trained on different features | Web page classification | Mostly unlabeled + some labeled |
| Graph-based | Uses data relations to improve learning | Social network analysis | Mostly unlabeled + some labeled |

### Self-Supervised Learning Types
Learns from **unlabeled** data by creating its own supervised signals.

| Type | Description | Use Case | Input Data Type |
|------|-------------|----------|-----------------|
| Contrastive Learning | Learns by contrasting similar vs dissimilar pairs | Image similarity search | Unlabeled |
| Masked Language Modeling | Predicts hidden parts (e.g., missing words) | Language understanding (BERT) | Unlabeled |
| Autoregressive | Predicts next item in a sequence | Text generation (GPT) | Unlabeled |

### Transfer Learning Types
Leverages pre-trained models on new but related tasks with smaller datasets.

| Type | Description | Use Case | Input Data Type |
|------|-------------|----------|-----------------|
| Feature Extraction | Uses frozen features from pre-trained models | Image classification with pre-trained CNN | Labeled (target task) |
| Fine-tuning | Adjusts pre-trained models to new domains | Domain-specific language models | Labeled (target task) |
| Domain Adaptation | Transfers learning between related domains | News to social media text | Labeled (source + target) |

### Bias in Machine Learning

#### Types of Bias
| Bias Type | Description | Examples | Mitigation Strategies |
|-----------|-------------|----------|----------------------|
| **Human Bias** | Prejudices from human decisions | Gender/racial stereotypes in hiring | Diverse teams, bias training |
| **Algorithmic Bias** | Unfair outcomes from model design | Biased feature selection | Algorithm audits, fairness constraints |
| **Data Skew** | Unrepresentative training data | Underrepresented demographics | Data balancing, synthetic data |

#### Benchmark Datasets for Bias Evaluation
- **COMPAS**: Criminal risk assessment fairness
- **Adult Census**: Income prediction fairness
- **FICO Credit**: Credit scoring bias evaluation
- **CelebA**: Facial attribute bias in computer vision
- **IMDB**: Sentiment analysis demographic bias

### Search Methods
- **Keyword Search**: Matches exact terms in the search query.
- **Semantic Search**: Uses embeddings to understand the meaning behind the search query, allowing for more accurate and meaningful results.

### Vector Databases
- **Definition**: A type of database designed for storing and querying vectors (embeddings), which is useful for tasks like semantic search.
- **Key Features**: 
  - **Similarity Search**: Find vectors most similar to a query vector
  - **Approximate Nearest Neighbor (ANN)**: Fast similarity search algorithms
  - **Indexing**: Efficient storage and retrieval of high-dimensional vectors
  - **Scalability**: Handle millions to billions of vectors

#### Similarity Search Methods
- **Cosine Similarity**: Measures angle between vectors, good for normalized data
- **Euclidean Distance**: Measures straight-line distance between points
- **Dot Product**: Measures vector alignment, faster computation
- **Manhattan Distance**: Sum of absolute differences, robust to outliers

#### Vector Database Options on AWS
- **Amazon OpenSearch Service**: Supports k-nearest neighbor (k-NN) search for vector databases. Useful for log analytics, real-time application monitoring, and search.
- **Amazon Aurora PostgreSQL-Compatible Edition & Amazon RDS for PostgreSQL**: Supports the __pgvector__ extension, enabling efficient storage of embeddings and similarity searches.
- **Amazon Neptune ML**: Uses Graph Neural Networks (GNNs) to make predictions based on graph data, supporting vectorized data in graph databases.
- **Amazon MemoryDB**: Supports high-speed vector storage and retrieval with millisecond query times and tens of thousands of queries per second (QPS).
- **Amazon DocumentDB**: Supports vector search with MongoDB compatibility, enabling storage, indexing, and search of millions of vectors with millisecond response times.

## The Machine Learning (ML) Pipeline (ML Lifecycle)

The ML Pipeline is a systematic process used to build, train, and deploy machine learning models. It ensures that each stage, from identifying business goals to monitoring deployed models, is properly managed and optimized for performance.

**Steps:**
1. Identify Business Goal
2. Frame ML Problem
3. Collect Data
4. Pre-Process Data
5. Engineer Features
6. Train, Tune, Evaluate
7. Deploy
8. Monitor

### 1. Identify Business Goal
- **Description**: Define success criteria and align stakeholders to ensure the ML project meets business objectives.
- **Key Activities**:
  - Establish clear success metrics.
  - Align with stakeholders across the organization.

### 2. Frame the ML Problem
- **Description**: Define the ML problem, inputs, outputs, and metrics while considering feasibility and cost/benefit analysis.
- **Key Activities**:
  - Identify inputs, outputs, requirements, and performance metrics.
  - Perform cost-benefit analysis to evaluate feasibility.

- **Model Options**:
  - **AI/ML Hosted Service** (e.g., AWS Comprehend, Forecast, Personalize): No training required.
  - **Pre-trained Models** (e.g., Amazon Bedrock, SageMaker JumpStart): Fine-tune with your data.
  - **Fully Custom Model**: Build and train from scratch.

### 3. Collect Data
- **Description**: Collect and prepare the necessary data for training the model.
- **Key Activities**:
  - Identify data sources (e.g., databases, data lakes, external APIs).
  - Ingest and label data.

- **Where Data is Stored**:  
  - Data collected for machine learning is typically stored in **Amazon S3**. This applies to both data sourced from internal systems and third-party datasets accessed via **AWS Data Exchange**.

- **Tools**:
  - **AWS Glue**: For **ETL (Extract, Transform, Load)** processes, moving and transforming data into a usable format before storage in **Amazon S3**.
  - **SageMaker Ground Truth**: For human labeling of ambiguous data, with labeled data stored in **Amazon S3**.
  - **AWS Data Exchange**: Allows secure access to third-party datasets. These datasets can be used as additional sources of training data, and the ingested data is stored in **Amazon S3**.
  - **Amazon S3**: Primary storage service where collected data is securely stored before being processed or used for training.

### 4. Pre-Process Data
- **Description**: Clean and prepare the data, ensuring it is suitable for training.
- **Key Activities**:
  - Perform **Exploratory Data Analysis (EDA)**.
  - Clean the data, removing duplicates, filling missing values, and anonymizing **PII**.
  - Split data is often split into ratios of **training (80%)**, **validation (10%)**, and **test (10%)** sets.

#### Data Splitting Strategies
| Split Type | Purpose | Typical Ratio | Best Practices |
|------------|---------|---------------|----------------|
| **Training Set** | Model learning | 60-80% | Largest portion, representative |
| **Validation Set** | Hyperparameter tuning | 10-20% | Used during training for model selection |
| **Test Set** | Final evaluation | 10-20% | Never used during training, unbiased evaluation |

Here are brief explanations of each concept in structured and unstructured data processing, as well as data augmentation and regularization, commonly used in machine learning and data science workflows:[5][6]

## Structured Data Processing

- **Normalization**: Adjusts numeric features to a standard range (typically ) using Min-Max scaling, making them comparable and helping algorithms converge efficiently.[1][6][5]
- **Standardization**: Centers data by transforming features to have a mean of 0 and standard deviation of 1 using the Z-score, which removes biases from features on different scales.[5]
- **Encoding**: Converts categorical variables into a numerical format—for example, using One-hot (binary columns for each category), Label (assigning numbers to categories), or Target encoding (using target statistics).[5]
- **Handling Missing Values**: Deals with absent data using strategies like imputation (filling with mean/median/most common value), deletion (removing incomplete records), or flagging (marking missing as an indicator feature).[6]

## Unstructured Data Processing

- **Tokenization**: Splits raw text into smaller units—words, subwords, or characters—allowing further text analysis or processing.[5]
- **Vectorization**: Translates text into numeric representations (e.g., TF-IDF, word embeddings), so algorithms can use the data.[5]
- **Image Processing**: Applies techniques like resizing, normalization, and augmentation (rotating, flipping) to prepare images for computer vision tasks.[5]
- **Audio Processing**: Extracts numerical features (such as Mel-frequency cepstral coefficients—MFCCs, or spectrograms) from speech or audio for analysis or modeling.[5]

## Data Augmentation and Regularization

### Data Augmentation

- **Images**: Creates new samples from originals by rotating, flipping, cropping, or adjusting color, increasing data variety.[6][5]
- **Text**: Generates synthetic data using methods like synonym replacement, back-translation, or paraphrasing to enrich the dataset.[6]
- **Audio**: Alters audio data by stretching time, shifting pitch, or adding noise, augmenting the dataset.[6]

### Regularization

- **L1 (Lasso)**: Encourages sparsity by shrinking some model weights to zero, which acts as automatic feature selection.[5]
- **L2 (Ridge)**: Penalizes large weights to prevent the model from fitting noise (overfitting).[5]
- **Dropout**: Randomly disables neurons during neural network training to reduce dependence on specific nodes and prevent overfitting.[5]
- **Early Stopping**: Halts training when model performance on a validation set stops improving, avoiding overfitting.[5]

These steps and techniques are essential for preparing, transforming, and optimizing data for improved machine learning performance.[6][5]

- **How to Clean Data**:
  - **AWS Glue Transformations**: Glue has built-in transformations for tasks like removing duplicates or filling missing values, and allows custom transformations using Python or Spark.
  - **Macie for PII**: AWS Macie detects and anonymizes **PII** data, working with **Amazon S3** to scan and mask sensitive information.
  - **AWS Glue DataBrew**: Enables data preparation and cleaning through a visual interface. You can apply **data quality rules**, such as filling missing values, and save these transformations as **recipes** for reuse.
  - **SageMaker Canvas**: Facilitates data import, transformation, and visualization without requiring deep technical knowledge. It uses built-in transformations that are added step-by-step to prepare data for training, and each step can be visually tracked in the flow.

- **Tools**:
  - **AWS Glue**: An ETL service with built-in transformations for data cleaning.
  - **AWS Glue DataBrew**: A visual data preparation tool where you can define and apply transformation rules (called **recipes**), with built-in data quality checks.
  - **SageMaker Canvas**: A tool for importing, preparing, and transforming data with a visual, no-code interface. Each transformation step is part of a clear workflow, making data preparation more accessible.

### 5. Engineer Features
- **Description**: Select and engineer features that will enhance model performance.
- **Key Activities**:
  - Feature Selection: Identifying the most relevant features from your dataset based on domain knowledge, reducing dimensionality and improving model efficiency.
  - Feature Creation: Transforming raw data into new features (e.g., scaling, encoding categorical variables, and deriving new metrics from existing ones).
  - Feature Transformation: Applying mathematical or statistical transformations (e.g., normalization, standardization) to improve model convergence and performance.

#### Feature Engineering Steps
| Step | Description | Techniques | Tools |
|------|-------------|------------|-------|
| **Feature Selection** | Choose most relevant features | Filter, wrapper, embedded methods | Correlation analysis, RFE |
| **Feature Extraction** | Create new features from existing data | Domain knowledge, polynomial features | SciKit-learn, AWS Glue |
| **Feature Transformation** | Modify feature distributions | Log, square root, Box-Cox transforms | Data Wrangler, DataBrew |
| **Feature Scaling** | Normalize feature ranges | Min-max, standard scaling, robust scaling | SageMaker preprocessing |

- Automated Feature Engineering: Consider using SageMaker Autopilot, which can automatically generate and rank features that could improve model performance.
- Dimensionality Reduction: Techniques can be applied to reduce the number of features while retaining the most important information.
- Feature Scaling: Tools like AWS Glue or SageMaker DataBrew can be used to apply scaling techniques (e.g., normalization or standardization), ensuring that features contribute equally to model training.
- Categorical Encoding: Convert categorical variables into numerical values using techniques like one-hot encoding or target encoding using SageMaker Data Wrangler or AWS Glue.

- **Tools**:
  - **SageMaker Feature Store**: Store and manage features as a single source of truth.

### 6. Train, Tune, and Evaluate the Model
- **Description**: Train the model, tune it, and evaluate performance.
- **Key Activities**:
  - Train the model iteratively and fine-tune parameters.
  - Tune hyperparameters (e.g., epoch, batch size, learning rate) and run experiments.
  - Evaluate the model using metrics and compare performance.

#### Model Performance Issues

##### Overfitting vs Underfitting
| Issue | Description | Symptoms | Solutions |
|-------|-------------|----------|-----------|
| **Overfitting** | Model memorizes training data | High training accuracy, poor test accuracy | Regularization, dropout, early stopping, more data |
| **Underfitting** | Model too simple for data complexity | Poor performance on both training and test | More complex model, feature engineering, reduce regularization |

##### Bias vs Variance Trade-off
- **High Bias (Underfitting)**: Model assumptions too simplistic
  - **Symptoms**: Poor performance on training and test data
  - **Solutions**: Increase model complexity, add features, reduce regularization
- **High Variance (Overfitting)**: Model too sensitive to training data
  - **Symptoms**: Large gap between training and validation performance
  - **Solutions**: Regularization, cross-validation, ensemble methods

##### Prevention Techniques
- **Cross-Validation**: k-fold validation to assess model stability
- **Regularization**: L1/L2 penalties to prevent overfitting
- **Pruning**: Remove unnecessary model components
- **Ensemble Methods**: Combine multiple models to reduce variance

##### Generalization and Inference
- **Generalization**: Model's ability to perform well on unseen data
- **Inference Types**:
  - **Batch Inference**: Process large amounts of data offline
  - **Real-time Inference**: Immediate predictions for individual requests
  - **Streaming Inference**: Continuous processing of data streams

### 7. Deploy the Model
- **Description**: Deploy the trained model to make predictions.

- SageMaker Deployment Options:
  - **Real-Time Inference**: For low-latency, sustained traffic predictions with auto-scaling capabilities.
  - **Batch Transform**: For processing large batches of data asynchronously.
  - **Asynchronous Inference**: For long-running inference requests with large payloads, handled without immediate responses.
  - **Serverless Inference**: For intermittent traffic, where the model scales automatically without infrastructure management.

- **Tools**:
  - **AWS API Gateway (Optional)**: Used to expose models as RESTful APIs, enabling seamless integration with other applications or microservices. It's optional and typically used when you want external applications to interact with your model endpoint.

#### AWS Trainium and AWS Inferentia
| Chip | Purpose | Use Case | Benefits |
|------|---------|----------|----------|
| **AWS Trainium** | Machine learning training | Large-scale model training | Cost-effective, high performance for training workloads |
| **AWS Inferentia** | Machine learning inference | Real-time and batch inference | Low latency, high throughput, cost-optimized inference |

### 8. Monitor the Model
- **Description**: Continuously monitor the model's performance and detect any data or concept drift.
- **Key Activities**:
  - Monitor model in real-time for data and concept drift.
  - Set alerts and automate corrective actions if performance degrades.

- **Types of Drift**:
  - Data Drift: The input data changes, but the relationship between inputs and outputs remains the same (e.g., a different demographic).
  - Concept Drift: The relationship between inputs and outputs changes, meaning the model's learned patterns no longer apply (e.g., new patterns of fraud that the model wasn't trained on).

- **Tools**:
  - **SageMaker Model Monitor**: Schedule and monitor data drift, send results to CloudWatch, and automate corrective measures.

## Evaluation Metrics

### Regression Metrics
- **MSE (Mean Squared Error):** Measures average squared difference between predicted and true values. Lower is better. Used for continuous value predictions like house prices.
- **RMSE (Root Mean Squared Error):** Square root of MSE; error in same units as output. Easier to interpret than MSE.
- **MAE (Mean Absolute Error):** Average absolute difference; less sensitive to outliers. Useful when outliers should not dominate evaluation.
- **MAPE (Mean Absolute Percentage Error):** Percentage-based error metric; good for comparing across scales but can be biased near zero values.

### Classification Metrics
- **Confusion Matrix:** Summarizes prediction results into True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
- **Accuracy:** Overall correctness of predictions. Best with balanced classes.
- **Precision:** Ratio of correct positive predictions to total positives predicted. Important when false positives are costly (e.g., spam detection).
- **Recall (Sensitivity):** Ratio of detected positives to all actual positives. Important when missing positives is costly (e.g., disease detection).
- **F1-Score:** Balance between precision and recall, useful when both errors matter.
- **Specificity:** Ratio of correct negative predictions; important when identifying negatives well matters.
- **AUC-ROC Curve:** Visualizes trade-off between true positive and false positive rates at different thresholds. Higher area under curve means better class separation.

### Performance Metrics
- **Average Response Time:** Time taken by model to provide prediction; important for real-time use cases.
- **Data Throughput:** Number of requests processed per second; relevant for high-volume environments.
- **P95/P99 Latency:** Measures worst-case response times to ensure service level objectives (SLOs).

### Advanced Evaluation Metrics
- **ROUGE Score:** Measures overlap between generated text and reference text; used in summarization.
- **BLEU Score:** Evaluates the quality of machine-translated text through n-gram matching.
- **BERTScore:** Uses contextual embeddings to evaluate semantic similarity in generated text quality.

## Comparison Tables

### Traditional ML vs Deep Learning
| Category | Traditional ML | Deep Learning |
|----------|----------------|---------------|
| **Task Complexity** | Well-defined tasks | Complex tasks |
| **Data Type** | Structured / Labeled Data | Unstructured Data (Images, Video, Text) |
| **Methodology** | Solves problems through statistics and mathematics | Utilizes neural networks |
| **Feature Handling** | Manually select and extract features | Features are learned automatically by the model |
| **Cost** | Less costly | Higher costs due to computational demands |

### Types of Machine Learning
| Learning Type | Description | Challenges | AWS Tools | Common Use Cases |
|---------------|-------------|------------|-----------|------------------|
| **Supervised Learning** | Uses pre-labeled data to train models. | Labelling the data can be challenging. | SageMaker Ground Truth | Image classification, spam detection |
| **Unsupervised Learning** | Works with unlabeled data to find patterns. | Requires methods to interpret patterns. | None specific | Clustering, anomaly detection, LLMs for initial training phases |
| **Reinforcement Learning** | Learns through trial and error to maximize rewards. | Requires environment for agent interaction. | AWS DeepRacer | Gaming, robotics, real-time decisions |
| **Semi-Supervised Learning** | Combines labeled and unlabeled data. | Balancing labeled/unlabeled data quality. | Custom implementations | When labeled data is scarce |
| **Self-Supervised Learning** | Creates labels from input data itself. | Designing effective pretext tasks. | Transformer models | Pre-training language models |
| **Transfer Learning** | Adapts pre-trained models to new tasks. | Domain mismatch between source and target. | SageMaker JumpStart | Fine-tuning for specific domains |

### Types of Training Data for Machine Learning/AI
| Data Type | AWS Data Source Example | Actual Example |
|-----------|------------------------|----------------|
| **Structured** | SQL data stored in RDS then moved to S3 | Customer information in relational tables |
| **Semi-Structured** | Data in DynamoDB or DocumentDB then moved to S3 | JSON logs of user activity |
| **Unstructured** | Objects and files stored directly in S3 | Images, videos, and PDF documents |
| **Time-Series** | Time-stamped data stored in S3 | IoT device data, stock market data |

---

# 2. Fundamentals of Generative AI

### Generative AI
- **Definition**: A subset of deep learning focused on creating new content (such as text, images, or music) from learned data. Generative AI models are generally based on **foundation models**, which are large, pre-trained models that can be adapted to a wide range of tasks.
- **Key Concept**: AI systems that generate novel outputs based on patterns learned during training. Foundation models provide the versatility needed for fine-tuning on specific tasks.
- **Examples**: **ChatGPT** for text generation, **DALL·E** for image generation, **DeepDream** for artistic visual generation.

#### Generative AI vs Traditional ML
| Aspect | Traditional ML | Generative AI |
|--------|----------------|---------------|
| **Purpose** | Prediction/Classification | Content Generation |
| **Output** | Labels, predictions, decisions | New content (text, images, code) |
| **Training Data** | Typically structured, labeled data | Large amounts of diverse, often unlabeled data |
| **Model Size** | Smaller, task-specific models | Large foundation models (billions of parameters) |
| **Flexibility** | Task-specific | Multi-purpose, adaptable |
| **Cost** | Lower computational requirements | High computational requirements |

### Large Language Models (LLMs)
- **Definition**: A type of **Generative AI** focused on understanding and generating human-like text. LLMs are trained on vast amounts of text data to learn language patterns, grammar, and context.
- **Key Concept**: LLMs generate coherent and high-quality text based on input prompts and are capable of performing tasks like summarization, translation, and question answering.
- **Examples**: **GPT-4**, **BERT**, **T5**, **Claude 2**—used for tasks like text generation, sentiment analysis, summarization, and natural language understanding.

#### Popular Large Language Models
| Model | Company | Type | Key Features |
|-------|---------|------|--------------|
| **GPT-4** | OpenAI | Generative | Multimodal, advanced reasoning |
| **Claude 2** | Anthropic | Conversational | Safety-focused, constitutional AI |
| **PaLM 2** | Google | Generative | Multilingual, reasoning capabilities |
| **LLaMA 2** | Meta | Open-source | Efficient, customizable |
| **Titan** | Amazon | Enterprise | AWS-integrated, responsible AI |

#### How GPT Works
| Component | Function | Process |
|-----------|----------|---------|
| **Transformer Architecture** | Core neural network structure | Self-attention mechanism processes sequences |
| **Pre-training** | Learn language patterns from vast text | Predict next token in billions of text sequences |
| **Tokenization** | Convert text to numerical tokens | Break text into subword units for processing |
| **Attention Mechanism** | Focus on relevant parts of input | Weigh importance of different words in context |
| **Autoregressive Generation** | Generate text one token at a time | Use previous tokens to predict next token |
| **Fine-tuning** | Adapt for specific tasks | Train on task-specific labeled data |

### Small Language Models (SLMs)
- **Definition**: Compact language models designed for edge computing and resource-constrained environments.
- **Key Features**: Lower latency, reduced computational requirements, optimized for specific tasks.
- **Use Cases**: Mobile applications, IoT devices, real-time inference, on-device processing.
- **Examples**: DistilBERT, MobileBERT, TinyBERT.

### Foundation Models
- **Definition**: Large, pre-trained models that serve as a base for multiple downstream tasks through fine-tuning or prompting.
- **Key Concept**: Versatile models trained on diverse data that can be adapted for specific use cases.

#### Popular Foundation Models
| Model | Company | Modality | Key Strengths |
|-------|---------|----------|---------------|
| **GPT-4** | OpenAI | Text/Multimodal | General intelligence, reasoning |
| **DALL-E 2** | OpenAI | Image Generation | Text-to-image synthesis |
| **Claude 2** | Anthropic | Text | Safety, long context |
| **Stable Diffusion** | Stability AI | Image | Open-source, customizable |
| **PaLM 2** | Google | Text | Multilingual, mathematical reasoning |

### Advanced Prompt Engineering Techniques

#### Prompt Types
| Prompt Type | Description | Use Case | Example |
|-------------|-------------|----------|---------|
| Zero-Shot | No examples provided | When no examples available | "Translate 'Hello' to French." |
| One-Shot | Single example provided | Simple task demonstration | "Translate 'Hello' to French. Example: 'Hi' -> 'Salut'." |
| Few-Shot | Multiple examples provided | Complex task guidance | "Translate English to French. Examples: 'Hi' -> 'Salut', 'Bye' -> 'Au revoir'." |
| Chain-of-Thought | Encourages step-by-step reasoning | Problem-solving tasks | "Solve 23 + 17. First add 20 + 10 = 30, then 3 + 7 = 10, total 40." |
| Prompt Template | Pre-defined format/structure | Standardized interactions | "Translate: [input sentence]" |
| Prompt Tuning | Optimizing prompts for performance | Model optimization | Fine-tuning prompt wording to improve answers |
| Adversarial Prompts | Designed to test model robustness | Security testing, bias detection | Input designed to confuse or trick the model |
| System Prompts | High-level instructions for model behavior | Setting model personality/role | "You are a helpful assistant." |
| Instruction Following | Clear task instructions | Direct command execution | "Write a summary of the following article." |

#### Advanced Prompting Techniques
| Technique | Description | Use Case | Example |
|-----------|-------------|----------|---------|
| **Outpainting** | Extend images beyond original boundaries | Image expansion, scene completion | Extending landscape photos |
| **Mask Prompting** | Use masks to control specific image regions | Selective editing, object replacement | Replace objects in photos |
| **Positive Prompts** | Explicit instructions for desired outcomes | Quality improvement, style control | "High quality, detailed, professional" |
| **Least-to-Most Prompting** | Break complex problems into simpler sub-problems | Complex reasoning tasks | Solving multi-step math problems |

#### Dynamic Prompt Engineering
- **Definition**: Adaptive prompting that changes based on context, user behavior, or model responses
- **Components**:
  - **Context-Aware Prompts**: Adjust based on conversation history
  - **Performance-Based Adaptation**: Modify prompts based on output quality
  - **User Personalization**: Customize prompts for individual preferences
- **Implementation**: A/B testing, reinforcement learning, feedback loops
- **Benefits**: Improved relevance, better user experience, higher success rates

### Core Concepts

### Latent Space
- **Definition**: The encoded knowledge or patterns captured by large language models (LLMs) that store relationships between data.
- **Usage**: It represents the internal understanding of language or data that AI models use to generate outputs.

### Embeddings - Comprehensive Overview
- **Definition**: Numerical representations of real-world objects that machine learning (ML) and artificial intelligence (AI) systems use to understand complex knowledge domains like humans

#### Key Characteristics:
- **Dense Vectors**: Compact numerical representations (typically 100-1536 dimensions)
- **Semantic Similarity**: Similar objects have similar embeddings in vector space
- **Contextual**: Meaning depends on surrounding context (contextual embeddings)
- **Learnable**: Automatically generated through training on large datasets

#### Types of Embeddings
| Type | Description | Use Cases | Examples |
|------|-------------|-----------|----------|
| **Word Embeddings** | Individual word representations | NLP tasks, similarity search | Word2Vec, GloVe |
| **Sentence Embeddings** | Entire sentence representations | Document similarity, clustering | Sentence-BERT, Universal Sentence Encoder |
| **Document Embeddings** | Full document representations | Document classification, retrieval | Doc2Vec, transformer-based |
| **Image Embeddings** | Visual content representations | Image search, computer vision | ResNet features, CLIP |
| **Multimodal Embeddings** | Cross-modal representations | Image-text matching, search | CLIP, ALIGN |

#### Embedding Applications
- **Semantic Search**: Find contextually similar content
- **Recommendation Systems**: Suggest similar items based on embeddings
- **Clustering**: Group similar objects together
- **Classification**: Use embeddings as features for ML models
- **Anomaly Detection**: Identify outliers in embedding space
- **Transfer Learning**: Use pre-trained embeddings for new tasks

### Tokens
- **Definition**: The basic units of text (e.g., words, subwords, or characters) that are processed by language models. In the context of LLMs, tokens are used to represent both inputs (text provided) and outputs (text generated).

### Context-Window
- **Definition**: The maximum amount of tokens an LLM model can process at once, including both the input prompt and the output generated by the model. If the number of tokens exceeds the model's context-window, earlier parts of the text may be truncated.
- **Usage**: The context-window is a key factor in determining how much information can be fed into the model at one time and affects tasks like long-form text generation, document analysis, or multi-turn conversations.

### Model Weights
- **Definition**: Parameters learned during training that determine how input features are transformed and combined to produce outputs.
- **Key Concepts**: 
  - Weights represent the strength of connections between neurons/layers
  - Updated through backpropagation during training
  - Fine-tuning adjusts pre-trained weights for new tasks
  - Quantization can reduce weight precision to optimize inference

### Agents
- **Definition**: AI systems that can perceive their environment, make decisions, and take actions to achieve specific goals.
- **Components**: 
  - **Perception**: Ability to receive and interpret inputs
  - **Decision-making**: Logic to choose appropriate actions
  - **Action**: Ability to execute decisions in the environment
  - **Memory**: Store and recall past experiences
- **AWS Implementation**: Bedrock Agents for multi-step workflows

### Hallucinations
- **Definition**: Hallucinations occur when a language model generates incorrect or nonsensical information that may sound plausible but is not grounded in factual data or the provided input.
- **Mitigations**: 
  - Retrieval Augmented Generation (RAG): Mitigates hallucinations by retrieving relevant external data during the generation process, ensuring the model generates responses based on accurate information.
  - Fine-Tuning: Training the model on more relevant, accurate data can help reduce hallucinations by aligning the model's output with factual knowledge.
  - Human-in-the-Loop (HITL): Incorporating human review in low-confidence areas can prevent hallucinated outputs from being used in critical applications.

### Multi-Modal Models
- **Definition**: Models that work across multiple data types, embedding text, images, or even audio into a shared space. These models are commonly used for multimodal generation tasks, such as creating captions for images or generating visuals from textual descriptions, by leveraging different types of input to produce richer, more context-aware outputs.

### Classifier-Free Guidance (CFG)
- **Definition**: Parameter controlling how closely diffusion models follow text prompts during image generation
- **CFG Value Impact**:
  - **Low CFG (1-3)**: More creative, diverse outputs with less prompt adherence
  - **Medium CFG (7-10)**: Balanced creativity and prompt following
  - **High CFG (15-20)**: Strict prompt adherence but may reduce image quality
- **Use Cases**: Image generation refinement, artistic control, prompt precision

## Generative AI Models and Metrics

### Types of Diffusion Models
| Diffusion Model Type | Description | When to Use | Notes |
|----------------------|-------------|-------------|-------|
| **Forward Diffusion** | Adds noise to data progressively. | Not often used (it adds noise) | |
| **Reverse Diffusion** | Reconstructs original data from noise. | Creating detailed images from distorted inputs. | Image restoration tools |
| **Stable Diffusion** | Works in reduced latent space, not directly in pixels. | Better then Reverse Diffusion | Midjourney, DALL-E |

### Generative AI Models
| Generative AI Model | Examples | Use Case/What It's Good For |
|---------------------|----------|----------------------------|
| **Generative Adversarial Networks (GANs)** | StyleGAN, CycleGAN, ProGAN | Image generation, face synthesis, video creation |
| **Variational Autoencoders (VAEs)** | Kingma & Welling VAE, Beta-VAE | Image denoising, anomaly detection, image compression |
| **Transformers** | GPT-4, BERT, T5 | Text generation, language translation, content generation |
| **Recurrent Neural Networks (RNNs)** | LSTMs, GRUs | Sequential data, time series forecasting, language models |
| **Reinforcement Learning for Generative Tasks** | AlphaGo, DQN, OpenAI Five | Game AI, autonomous control, optimizing generative tasks |
| **Diffusion Models** | Stable Diffusion, DALL·E 2, Imagen | Image and text-to-image generation |
| **Flow-Based Models** | Glow, RealNVP | High-quality image generation, density estimation |

### Generative AI Performance Metrics
| Metric Name | Explanation |
|-------------|-------------|
| **Recall Oriented Understudy for Gisting Evaluation (ROUGE)** | Measures overlap between generated and reference text, good for summaries. |
| **Bilingual Evaluation Understudy (BLEU)** | Evaluates translation quality by comparing n-grams between outputs and references. |
| **General Language Understanding Evaluation (GLUE)** | Assesses model performance on multiple natural language understanding tasks. |
| **Holistic Evaluation of Language Models (HELM)** | Provides broad, task-specific evaluation of language model capabilities. |
| **Massive Multitask Language Understanding (MMLU)** | Tests model knowledge across a variety of domains and topics. |
| **Beyond the Imitation Game Benchmark (BIG-bench)** | Evaluates models on creative and difficult AI tasks not covered by standard benchmarks. |
| **Perplexity** | Measures how well a model predicts the likelihood of the next token or word. |

---

# 3. Applications of Foundation Models

## AWS Services

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
- **AWS Intelligent Document Processing** (IDP)
  - A group of AWS services that together automate the extraction, classification, and processing of data from various document types. It leverages AI technologies such as Optical Character Recognition (OCR), Natural Language Processing (NLP), and Machine Learning (ML) to handle unstructured data found in documents like PDFs, invoices, and legal contracts.

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

#### Amazon Q

- **Amazon Q Business**
  - A generative AI-powered assistant that helps with tasks like answering questions, generating content, and automating workflows by accessing enterprise data sources like S3, SharePoint, and Salesforce.
- **Amazon Q Developer**
  - A tool for developers that includes features like code generation and security scanning to help automate tasks related to development and testing.
- **Amazon Q in QuickSight**
  - Integrated with Amazon QuickSight for natural language querying, allowing users to ask business intelligence-related questions and generate insights from their QuickSight data with AI.
- **Amazon Q in Connect**
  - Integrated with Amazon Connect, Amazon Q helps improve customer service by answering customer inquiries, automating responses, and managing tickets using natural language AI.
- **Amazon Q in AWS Supply Chain**
  - Integrated into AWS Supply Chain, Amazon Q assists in optimizing and automating supply chain management by generating insights from supply chain data, streamlining inventory management, and forecasting demand.

### Amazon SageMaker

Amazon SageMaker is an integrated machine learning service that enables developers and data scientists to build, train, and deploy machine learning models at scale. Users can create custom models from scratch or use and fine-tune existing ones through SageMaker JumpStart. This platform offers more control than high-level AI services like AWS Rekognition, allowing for detailed customization and optimization to meet specific project requirements.

#### SageMaker Studio

SageMaker Studio is an integrated development environment (IDE) for machine learning, providing a single interface for preparing data, building models, training, tuning, and deploying them. It offers features like Jupyter notebooks for code development, integrated debugging tools, and experiment management, all within a collaborative, cloud-based environment. Studio also supports real-time collaboration and easy access to various SageMaker capabilities, including model monitoring and data preparation.

#### Amazon SageMaker Notebook Instances
- **Definition**: Fully managed ML compute instances running Jupyter notebooks for data science and machine learning development
- **Key Features**:
  - **Pre-configured Environment**: Comes with popular ML frameworks and libraries pre-installed
  - **Multiple Instance Types**: Range from cost-effective instances for development to high-performance instances for intensive workloads
  - **Lifecycle Configuration**: Custom scripts to automatically configure notebook instances
  - **Git Integration**: Direct integration with Git repositories for version control
- **Use Cases**:
  - **Data Exploration**: Interactive data analysis and visualization
  - **Model Development**: Building and testing ML models
  - **Prototyping**: Rapid experimentation and proof-of-concepts
  - **Education**: Learning and teaching machine learning concepts
- **Management**:
  - **Start/Stop Control**: Manage costs by stopping instances when not in use
  - **Automatic Scaling**: Configure auto-scaling based on usage patterns
  - **Security**: Integrated with IAM for access control and VPC for network isolation
- **Migration Note**: AWS recommends migrating to SageMaker Studio for enhanced collaboration and integrated ML workflows

#### Training Process

The typical SageMaker training process includes several key elements that help configure and manage the training jobs:

- **Training Data Locations**: Data is typically stored in Amazon S3 and accessed via S3 URLs.
- **ML Compute Instances**: SageMaker leverages EC2 instances (ECR instances) for scalable compute power.
- **Training Images**: The training process is run using Docker container images specifically designed for machine learning.
- **Hyperparameters**: Parameters that guide the learning process (e.g., learning rate, batch size).
- **S3 Output Bucket**: The trained model artifacts are stored in an S3 bucket for later use.

#### Parameters

- **Inference Parameters** (supported by Amazon Bedrock):
  - **Randomness and Diversity**:
    - Temperature:
      - Controls randomness—higher values lead to more diverse, creative outputs; lower values produce more predictable, deterministic results.
    - Top K:
      - Limits the model to selecting from the top K most probable tokens; smaller K values result in safer, more predictable choices.
      - Top K is not a terribly useful parameter - use Temperature or Top-P instead. 
    - Top P:
      - Uses cumulative probability to choose tokens, focusing on the smallest set of tokens with a combined probability of P, balancing randomness and diversity.
      - Higher Top P values (closer to 1) reduce randomness by restricting the token pool to only the most probable choices. 
    - Top-K vs Top-P: Top K fixes the number of tokens considered, while Top P uses a variable number of tokens based on their combined probability, making Top P more adaptive and flexible in balancing randomness.
    - Temperature vs. Top P: Temperature adjusts the overall randomness by scaling probabilities, allowing more or less randomness across all possible tokens. Top P narrows the token choices to those that collectively add up to a certain probability threshold, balancing randomness with accuracy.
    - Use Cases:
      - Use Top-P when you want adaptive diversity but want to stay closer to more likely outcomes.
      - Use Temperature when you need consistent randomness control across the board.  
  - Response length:
    - Specifies the maximum length of generated output, affecting how verbose or concise the response is.
  - Penalties:
    - Applies penalties to repeated tokens or sequences to encourage variety in the generated text.
  - Stop sequences:
    - Defines specific sequences where the model will stop generating text, ensuring controlled output length or format.
- **Model Training Parameters** (Hyperparameters):
  - **Epoch**: The number of iterations through the entire dataset.
    - Increasing epochs generally improves the model's learning but can lead to overfitting if too high. More epochs help the model learn better but might also result in diminishing returns after a certain point.
  - **Batch Size**: Number of samples before updating model parameters.
    - Smaller batch sizes provide more frequent updates, which can help in converging quickly but can introduce more noise. Larger batch sizes are more stable but may require more computation and memory.
  - **Learning Rate**: Controls how fast the model learns.
    - A high learning rate speeds up training but may skip optimal solutions, while a low learning rate leads to slower but more precise convergence, though it risks getting stuck in local minima.

- **Model Training Issues**:
  - **Overfitting**: Too much training on the same data, causing the model to be overly specific.
    - Solution: Use more diverse data during training.
  - **Underfitting**: The model doesn't learn enough patterns from the data.
    - Solution: Train the model for more epochs or with more data.
  - **Bias and Fairness**: Lack of diversity in training data leading to biased predictions.
    - Solution: Ensure diverse and representative training data; include fairness constraints.

- **Fine-Tuning** (BedRock and SageMaker):
  - Adjust the weights of a pre-trained model with your specific and _labelled_ data to adapt it for new tasks. 
  - Be aware that if you only provide instructions for a single task, the model may lose its more general purpose capability and experience _catastrophic forgetting_.
  - **Domain adaptation fine-tuning** (SageMaker): Tailors a pre-trained foundation model to a specific domain (e.g., legal, medical) using a small amount of domain-specific data. This helps the model perform better on tasks related to that particular domain.
  - **Instruction-based fine-tuning** (SageMaker): Involves providing labeled examples of specific tasks (e.g., summarization, classification) to improve a model's performance on that particular task. This type of fine-tuning is useful for making the model better at tasks where precise outputs are needed.

- **Continued-Pretraining** (BedRock):
  - Using _unlabeled_ data to expand the model's overall knowledge without narrowing its scope to a specific task.

#### Continued Pre-Training vs DAFT (Domain Adaptive Fine-Tuning)
| Aspect | Continued Pre-Training | DAFT |
|--------|----------------------|------|
| **Data Type** | Unlabeled domain-specific text | Labeled task-specific examples |
| **Purpose** | Expand knowledge base | Improve specific task performance |
| **Model Capability** | Maintains general abilities | May lose general capabilities |
| **Training Method** | Unsupervised learning | Supervised learning |
| **Use Case** | Domain adaptation (medical, legal) | Task specialization (classification) |

- **Transfer Learning**:
  - Fine-tuning an existing model that has learned general features and applying it to a new problem, speeding up training and improving accuracy.

- **Tools**:
  - **SageMaker Training Jobs**: Manage training processes, specify training data, hyperparameters, and compute resources.
  - **SageMaker Experiments**: Track model runs and hyperparameter tuning.
  - **Automatic Model Tuning (AMT)**: Automatically tune hyperparameters using the specified metric.

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
- **SageMaker Autopilot**
  - Automatically builds, trains, and tunes machine learning models while giving you full visibility and control over the process.
- **Amazon Augmented AI (A2I)**
  - Allows you to add human review for low-confidence predictions or random samples, ensuring more accurate results.
- **SageMaker Managed Spot Training**
  - Reduces the cost of training models by using spare AWS EC2 capacity.
- **SageMaker Profiler**
  - Identifies resource inefficiencies in training jobs to minimize cost without sacrificing speed or accuracy.

### Amazon Bedrock

Amazon Bedrock is a fully managed, serverless service that provides access to high-performing foundation models (FMs) from leading AI companies through a single API. It is designed to facilitate the creation of generative AI applications, prioritizing security, privacy, and responsible AI.

#### Pricing Models
| Pricing Model | Description | Best For | Cost Structure |
|---------------|-------------|----------|----------------|
| **On-Demand** | Pay per input/output token | Variable, low usage | Per-token pricing |
| **Provisioned Throughput** | Reserved model capacity | Consistent, high usage | Hourly reservation + per-token |
| **Model Customization** | Fine-tuning and training costs | Custom models | Training + storage + inference |

#### Fine-tuning with Amazon Bedrock API for Data Privacy
- **Scenario**: Company needs to fine-tune a foundation model while ensuring data remains within the AWS Region and avoids internet exposure
- **Solution Benefits**:
  - **Regional Data Residency**: All fine-tuning operations occur within the specified AWS Region
  - **No Internet Exposure**: Data never leaves AWS network infrastructure
  - **Private Connectivity**: Use VPC endpoints and PrivateLink for secure access
  - **Managed Security**: AWS handles infrastructure security and compliance
- **Implementation**:
  - **VPC Integration**: Deploy Bedrock API calls through VPC endpoints
  - **S3 Regional Storage**: Store training data in S3 buckets within the same region
  - **IAM Controls**: Implement fine-grained access controls for data and model access
  - **Encryption**: Data encrypted in transit and at rest using customer-managed keys
- **Process Flow**:
  1. Upload training data to regional S3 bucket
  2. Configure VPC endpoint for Bedrock API access
  3. Initiate fine-tuning job through secure API calls
  4. Monitor training progress within AWS console
  5. Deploy fine-tuned model for inference within the same region

#### Features
- **Model Choice**  
  - Access a variety of foundation models from AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI, and Amazon, allowing for optimal model selection based on specific application needs.
  - Amazon Titan Models
    - Exclusive to Amazon Bedrock, Amazon Titan models are pretrained, high-performing foundation models created by AWS, designed for a wide range of use cases with responsible AI practices.
- **Customization**  
  - Customize foundation models privately with your data using techniques such as fine-tuning and Retrieval Augmented Generation (RAG), enhancing the model's relevance and accuracy.
- **Foundation Model Evaluation**
  - Model Evaluation on Amazon Bedrock allows you to evaluate, compare, and select the best foundation models for your specific use case. You can assess models based on custom metrics such as accuracy, robustness, and toxicity, ensuring they meet your performance requirements. Integration with Amazon SageMaker Clarify and fmeval further supports model evaluation by checking for bias and explainability.
- **Bedrock Knowledge Bases**
  - Uses Retrieval Augmented Generation (RAG) to fetch data from private sources, delivering more relevant and accurate responses with full support for the RAG workflow—handling data ingestion, retrieval, and prompt augmentation.
  - Enhances outputs by integrating retrieval processes that pull relevant external knowledge into the generative model.

#### Knowledge Bases vs RAG
| Aspect | Bedrock Knowledge Bases | Generic RAG |
|--------|-------------------------|-------------|
| **Management** | Fully managed by AWS | Manual setup and management |
| **Integration** | Native Bedrock integration | Custom integration required |
| **Vector Storage** | Automatic vector database management | Manual vector database setup |
| **Data Sources** | S3, web crawlers, databases | Custom data connectors |
| **Maintenance** | AWS handles updates and scaling | Manual maintenance required |

- **Bedrock Agents**  
  - Create agents capable of planning and executing multistep tasks using company systems and data sources—streamlining complex tasks such as customer inquiries and order processing.
- **Model Invocation Logging**
  - Comprehensive logging of all model invocations including inputs, outputs, and metadata for audit and compliance purposes.
  - Integration with CloudWatch and CloudTrail for monitoring and governance.
- **Serverless**  
  - Eliminates the need for infrastructure management, simplifying the deployment and scaling of AI capabilities.

#### Amazon Bedrock Guardrails

- **Definition**: Safety features that help prevent inappropriate content generation and ensure responsible AI usage
- **Purpose**: Filter harmful content, protect sensitive information, and maintain brand safety

#### Steps to Create Bedrock Guardrails

| Step | Component | Description | Configuration Options |
|------|-----------|-------------|----------------------|
| **1** | **Contextual Grounding Check** | Validates model outputs against source material | - Relevance threshold settings<br>- Source document validation<br>- Factual consistency checks |
| **2** | **Word Filters** | Blocks or allows specific words and phrases | - Profanity filters<br>- Custom word lists<br>- Regular expression patterns<br>- Severity levels |
| **3** | **Sensitive Information Filters** | Detects and masks PII and sensitive data | - Social Security Numbers<br>- Credit card numbers<br>- Email addresses<br>- Phone numbers<br>- Custom patterns |
| **4** | **Denied Topics** | Prevents discussion of specific subject areas | - Legal advice<br>- Medical diagnosis<br>- Financial recommendations<br>- Illegal activities<br>- Custom topic definitions |
| **5** | **Content Filters** | Controls harmful content generation | - Hate speech detection<br>- Violence filters<br>- Sexual content blocks<br>- Self-harm prevention<br>- Toxicity thresholds |

#### Guardrails Implementation Process
1. **Policy Definition**: Create guardrail policies with specific rules and thresholds
2. **Testing Phase**: Validate guardrail effectiveness with test prompts
3. **Integration**: Apply guardrails to model invocations through API parameters
4. **Monitoring**: Track guardrail activations and effectiveness metrics
5. **Optimization**: Adjust rules based on performance and business requirements

- **Security and Privacy Guardrails**  
  - Features robust controls to restrict AI outputs, preventing the generation of inappropriate content and restricting sensitive advice like financial recommendations, ensuring ethical and compliant AI usage.
- **PartyRock**
  - A hands-on AI app-building playground powered by Amazon Bedrock, where users can quickly build generative AI apps and experiment with models without writing code.

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

## Comparisons

### SageMaker vs Bedrock Inference Options
| Feature | SageMaker | Bedrock |
|---------|-----------|---------|
| **Model Types** | Custom models, pre-trained models | Foundation models only |
| **Customization** | Full model training and fine-tuning | Fine-tuning and prompt engineering |
| **Infrastructure** | Managed endpoints, custom instances | Serverless, fully managed |
| **Pricing** | Instance-based + usage | Token-based |
| **Use Cases** | Custom ML workflows | Generative AI applications |
| **Technical Expertise** | High (ML/Data Science) | Medium (Application Development) |

### SageMaker JumpStart vs Amazon Bedrock
| Aspect | SageMaker JumpStart | Amazon Bedrock |
|--------|---------------------|----------------|
| **Model Types** | Pre-trained models across domains | Foundation models for generative AI |
| **Customization** | Full fine-tuning capabilities | Fine-tuning with guardrails |
| **Infrastructure** | Requires endpoint management | Serverless, no infrastructure |
| **Cost Model** | Instance hours + storage | Pay-per-token usage |
| **Target User** | Data scientists, ML engineers | Application developers |
| **Model Variety** | Hundreds of models across tasks | Curated foundation models |

### Amazon SageMaker Inference Methods

| Inference Type | Deployed To | Characteristics |
|----------------|-------------|-----------------|
| **Batch** | EC2 | Cost-effective for infrequent, large jobs |
| **Asynchronous** | EC2 | Suitable for non-time-sensitive applications and large payload |
| **Serverless** | Lambda | Intermittent traffic, periods of no traffic, auto-scaling out of the box |
| **Real-Time** | EC2 | Live predictions, sustained traffic, low latency, consistent performance |

---

# 4. Guidelines for Responsible AI

## Core Frameworks

### Six GenAI Perspectives Framework - BP,GP,SO
'In a kingdom, the wise council focused on six pillars: Business to grow riches, People to train and care for citizens, Governance to set just rules, Platform to build strong castles, Security to guard the gates, and Operations to keep daily life running smoothly. Together, they balanced strategy, skill, oversight, tech, safety, and routine action.'

| Perspective | Description |
|-------------|-------------|
| **Business** | Value creation and strategic alignment |
| **People** | Human impact and workforce transformation |
| **Governance** | Oversight and decision-making frameworks |
| **Platform** | Technical infrastructure and capabilities |
| **Security** | Protection and risk management |
| **Operations** | Day-to-day management and optimization |

### Seven Capabilities of GenAI- ARS,CDPS
'GenAI was like a versatile artist with Adaptability to learn new styles, Responsiveness to answer instantly, Simplicity to make hard tasks easy, Creativity to invent new masterpieces, Data Efficiency to learn from little, Personalization to tailor art for each viewer, and Scalability to paint many canvases at once—all making it invaluable.'

| Capability | Description | 
|------------|-------------|
| **Adaptability** | Ability to adjust to new tasks and domains |
| **Responsiveness** | Real-time interaction and quick responses, generate content in real time |
| **Simplicity** | Simplify complex tasks. |
| **Creativity & Exploration** | Generate novel ideas, designs, or solutions. |
| **Data efficiency** | Learn from relatively small amounts of data |
| **Personalization** | Create personalized content tailored to individual preferences or characteristics. |
| **Scalability** | Generate large amounts of content quickly and handle varying workloads efficiently |

### AI's Foundational Security Capabilities Framework - BP, GP, SO
'Guardians of the AI realm watched over six gates: Business aligned the AI quest with goals, People trained the knights, Governance enforced laws, Platform held the fortress's tech, Security defended against threats, and Operations ensured smooth, ongoing patrols—protecting AI and all it served.'

| Capability | Description |
|------------|-------------|
| **Business** | AI strategy alignment with business objectives |
| **People** | Human-centered AI development and usage |
| **Governance** | Oversight frameworks for AI systems |
| **Platform** | Technical infrastructure and model management |
| **Security** | Protection of AI systems and data |
| **Operations** | Day-to-day AI system management |

### Core Dimensions of Responsible AI - FESP,VCTG
'In the land of AI, the Code of Honor had eight tenets: Fairness to treat all justly, Explainability to show clear reasons, Safety to prevent harm, Privacy & Security to guard secrets, Veracity & Robustness to ensure truth and strength, Controllability to keep humans in charge, Transparency to open the AI book, and Governance to uphold the laws—ensuring AI serves well and wisely.'

| Dimension | Description |
|-----------|-------------|
| **Fairness** | Ensuring AI systems treat all individuals and groups equitably |
| **Explainability** | Making AI decisions transparent and understandable |
| **Safety** | Preventing AI systems from causing harm |
| **Privacy & Security** | Protecting personal data and system integrity |
| **Veracity & Robustness** | Ensuring AI systems are reliable and accurate |
| **Controllability** | Maintaining human control over AI systems |
| **Transparency** | Providing clear information about AI system capabilities and limitations |
| **Governance** | Establishing frameworks for responsible AI development and deployment |

## Governance Strategies for Responsible AI

### Development Governance  
Involves forming diverse, multidisciplinary teams including ethicists and domain experts to build AI responsibly. It includes systematic risk assessments, embedding ethical design principles early, and comprehensive testing for bias, fairness, and robustness.

### Deployment Governance  
Focuses on carefully releasing AI systems through phased rollouts with continuous monitoring. It involves assessing real-world impacts on stakeholders and having clear procedures to respond promptly to any failures or harms caused by the AI.

### Usage Governance  
Ensures users are well-trained on AI capabilities and limitations. It includes setting access controls to restrict inappropriate use, maintaining audit trails to track AI operations, and enabling feedback channels for issue reporting and improvements.

### Organizational Governance  
Establishes oversight bodies such as AI ethics committees to guide AI use. It develops policies and compliance frameworks to meet legal and ethical standards while engaging regularly with stakeholders and affected communities for transparency and trust.

In AI land, building smart helpers needs careful rules:  
- **Development:** A team of experts (ethics, domain, diverse voices) craft AI with safety checks and fairness tests from day one.  
- **Deployment:** The AI is gently introduced step-by-step, watched closely for impact, and ready for quick fixes if anything goes wrong.  
- **Usage:** Users are trained well, given the right access, and can give feedback while every use is logged for trust.  
- **Organization:** A council of ethics oversees everything, makes clear policies, ensures laws are followed, and keeps the community in the loop.

## Disadvantages & Challenges of Generative AI

### Disadvantages of using gen AI in Production
'GenAI loves telling tales but sometimes hallucinates with fake facts. It forgets new info (knowledge cutoff), drinks too much power (costly), spills secrets (privacy risk), and confuses folks with its secretive mind (black box). People depend on it too much, and it's sometimes unfair (bias).'

| Disadvantage | Description |
|--------------|-------------|
| **Hallucination** | Generation of false or nonsensical information |
| **Knowledge Cutoff** | Limited to training data timeframe |
| **Resource and Cost Intensity** | High resource requirements for training and inference |
| **Security and Privacy Concerns** | Risks of data leaks and misuse of sensitive information |
| **Overreliance on AI** | Excessive dependence leading to skill atrophy |
| **Lack of Explainability (Black Box Problem)** | Difficulty in understanding AI decision processes |
| **Bias in Outputs** | Amplification of biases from training data |

### Challenges of genAI
'GenAI can trick with falsehoods (hallucinations), copy others' work (plagiarism), upset the workplace (work disruption), say offensive things (toxicity), and cause fights over who owns its creations (intellectual property).'

| Challenge | Description |
|-----------|-------------|
| **Hallucinations** | Generation of false or misleading information |
| **Plagiarism and Cheating** | Unauthorized copying or misuse of generated content |
| **Disruption of the Nature of Work** | Changes in job roles and workflows causing uncertainty |
| **Toxicity** | Generation of harmful, offensive, or biased content |
| **Intellectual Property** | Issues surrounding ownership and rights of AI-generated work |

### Ethical and Legal Challenges
| Challenge | Description | Impact |
|-----------|-------------|--------|
| **Toxicity** | Generation of harmful, biased, or inappropriate content | Brand damage, user harm |
| **Intellectual Property** | Potential copyright infringement in generated content | Legal liability, licensing issues |
| **Privacy Concerns** | Inadvertent exposure of sensitive training data | Regulatory violations, trust issues |
| **Misinformation** | Spreading false or misleading information | Social harm, credibility loss |
| **Job Displacement** | Automation replacing human workers | Economic disruption, social unrest |

---

# 5. Security, Compliance, and Governance for AI Solutions

## Security and Adversarial Attacks

### Adversarial Attacks on AI Systems
| Attack Type | Description |
|-------------|-------------|
| **Jailbreaking** | Bypassing safety restrictions |
| **Prompt Injection** | Malicious instructions within prompts |
| **Prompt Leaking** | Extracting original system prompts |
| **Token Smuggling** | Hiding malicious content in tokens |
| **Payload Splitting** | Breaking malicious content across inputs |

## Model Governance and Explainability

- **Description**: Ensure transparency, accountability, and regulatory compliance for ML models, while making their behavior interpretable to stakeholders.
- **Key Activities**:
  - Implement governance frameworks for tracking model usage and lineage.
  - Use tools to detect and mitigate bias, ensure fairness, and explain predictions.
  - Provide clear documentation of model history and performance for audits.
- **Tools**:
  - **SageMaker Clarify**: _Detect bias_, explain model predictions, and increase transparency.
  - **SageMaker Model Cards**: Create documentation for trained models, including performance metrics and intended use.
  - **ML Governance from SageMaker**: Provides tools for tighter control and visibility over ML models, helping track model information and monitor behavior like bias.
  - **SageMaker ML Lineage Tracking**: Capture the entire workflow, tracking model lineage for reproducibility and governance.
  - **Glue DataBrew**: Simplify data governance with visual data preparation and quality rules.
  - **AWS Audit Manager**: Automates the auditing of AWS services, ensuring continuous compliance and audit readiness for industry regulations.
  - **AWS Artifact**: Provides on-demand access to compliance reports and agreements, helping organizations meet compliance requirements.
  - **AWS AI Service Cards**: Enhance transparency by providing information on use cases, limitations, responsible AI practices, and performance best practices for AI services and models.
  - **AWS Trusted Advisor for AI Governance**: Provides recommendations for AI/ML workload optimization, security, and compliance best practices.

## MLOps and Automation

- **Description**: Apply DevOps principles to manage machine learning models throughout their lifecycle, focusing on automation, version control, and monitoring.
- **Key Activities**:
  - Automate deployment, monitoring, and retraining of models.
  - Ensure continuous integration and delivery (CI/CD) for model updates.
  - Implement version control to manage multiple model versions.
- **Tools**:
  - **SageMaker Pipelines**: Automate and manage the ML workflow end-to-end.
  - **AWS CodePipeline**: Automate the build, test, and deploy phases for models.
  - **SageMaker Model Registry**: Manage and track model versions and metadata.
  - **Amazon S3**: Used to store trained model artifacts after training for both SageMaker and Bedrock. Model outputs, including weights and artifacts, are exported to S3 for easy access during deployment and evaluation.

## Cost and Performance Optimization

- **Description**: Optimize resource usage and model performance without inflating costs.
- **Key Activities**:
  - Use managed spot training for lower-cost training jobs.
  - Select appropriate instance types for the job and leverage auto-scaling capabilities.
  - Use resource monitoring to detect inefficiencies in resource utilization.

### Amazon EC2 Spot Instances for AI Model Training
- **Definition**: Spare EC2 capacity available at up to 90% discount compared to On-Demand prices
- **Use Cases for AI Training**:
  - **Fault-tolerant workloads**: Training jobs that can handle interruptions
  - **Flexible timing**: Non-urgent training experiments
  - **Large-scale training**: Cost optimization for extensive computational requirements
  - **Hyperparameter tuning**: Multiple parallel training runs
- **Best Practices**:
  - Implement checkpointing to resume interrupted training
  - Use mixed instance types for better availability
  - Combine with SageMaker Managed Spot Training for automatic management
- **Tools**: SageMaker Managed Spot Training, AWS Batch with Spot instances

- **Tools**:
  - **AWS Trusted Advisor**: Provides recommendations for cost and performance improvements.
  - **SageMaker Managed Spot Training**: Reduce training costs by utilizing spare AWS EC2 capacity.
  - **SageMaker Profiler**: Identify inefficient resource use during model training.
  - **Amazon Inspector**: Automates security assessments of ML applications, identifying vulnerabilities that can lead to performance degradation or security issues.

## Continual Learning and Retraining

- **Description**: Continuously retrain models to account for new data and changing conditions, preventing performance degradation.
- **Key Activities**:
  - Schedule regular model retraining based on new data.
  - Use tools to detect performance drops and initiate automatic retraining workflows.
  - Handle model drift by monitoring for concept and data drift.
- **Tools**:
  - **SageMaker Model Monitor**: Detect data drift and trigger retraining workflows.
  - **SageMaker Pipelines**: Automate retraining processes end-to-end.

## Security

- **Description**: Implement best security practices to safeguard machine learning models, data, and related infrastructure.

- **Key Activities**:
  - **Least Privilege Principle**: Ensure that IAM roles and policies grant only the permissions required for specific jobs or functions.
  - **PrivateLink and VPC Endpoints**: Lock down **SageMaker** to prevent exposure to the internet. Use **PrivateLink** and **VPC endpoints** to securely access resources within your private network.
  - **Encryption at Rest and in Transit**: By default, **SageMaker** encrypts data at rest and in transit using **KMS** (Key Management Service).
  - **IAM Roles and Policies**: Create and manage IAM roles and policies to ensure secure access to model data and resources.
  - **S3 Block Public Access**: Prevent model data from being exposed by ensuring **S3 Block Public Access** settings override any potential public access.
  - **AWS IAM Identity Center**: Centralize identity management, allowing access to multiple AWS accounts, and integrate with Active Directory for identity management.

### VPC Endpoints for Secure Data Transfer
| Endpoint Type | Description | Use Cases | AWS Services |
|---------------|-------------|-----------|--------------|
| **Gateway Endpoints** | Route traffic through AWS network | S3, DynamoDB access | S3, DynamoDB |
| **Interface Endpoints** | Private IP addresses in VPC | Most AWS services | SageMaker, Bedrock, EC2 |

### Using Gateway Endpoint for S3
- **Purpose**: Enable private access to S3 from within a VPC without internet gateway
- **Benefits**:
  - **No Internet Exposure**: Data transfers between VPC and S3 remain within AWS network
  - **Cost Efficiency**: No NAT gateway costs for S3 access
  - **Improved Security**: Traffic never leaves AWS infrastructure
- **Implementation**:
  - Create Gateway Endpoint in VPC route table
  - Configure routing for S3 traffic through the endpoint
  - Applies to all S3 buckets in the region
- **Use Cases**:
  - ML training data access from SageMaker instances
  - Model artifact storage and retrieval
  - Data processing workflows requiring S3 access

### VPC Endpoints vs AWS PrivateLink
| Aspect | VPC Endpoints | AWS PrivateLink |
|--------|---------------|-----------------|
| **Definition** | AWS-managed service for private connectivity | Underlying technology enabling private connectivity |
| **Implementation** | Specific endpoint configurations | Broader networking technology |
| **Management** | Managed by AWS | Technology platform |
| **Use Case** | Direct service access | Foundation for private connections |
| **Relationship** | Uses PrivateLink technology | Enables VPC Endpoints functionality |

### Data Privacy and AWS Network Benefits
- **PrivateLink Advantages**:
  - **No Internet Exposure**: All traffic remains within AWS network backbone
  - **Enhanced Security**: Eliminates exposure to internet-based threats
  - **Compliance**: Meets strict data residency and privacy requirements
  - **Performance**: Lower latency through AWS direct network paths
- **Gateway Endpoint Advantages**:
  - **Cost Optimization**: No additional charges for data transfer
  - **Simplified Architecture**: Direct routing without complex networking
  - **Regional Coverage**: Access to all S3 buckets in the region
  - **Seamless Integration**: Works with existing VPC infrastructure

### AWS Regional vs Global Services
| Service Type | Examples | Characteristics |
|--------------|----------|----------------|
| **Regional** | EC2, S3, SageMaker, Bedrock | Data stays in specific region, region-specific endpoints |
| **Global** | CloudFront, Route 53, IAM | Globally distributed, single global endpoint |

- **Tools**:
  - **AWS Config**: Continuously monitors and records configuration changes across AWS resources to ensure compliance and security.
  - **AWS CloudTrail**: Logs API calls and tracks user activity for auditing and compliance.
  - **Amazon Inspector**: Automatically scans for vulnerabilities in machine learning environments to ensure that deployed models are secure.
  - **AWS Audit Manager**: Automates auditing and ensures compliance with industry regulations by generating audit reports for validation.
  - **AWS Artifact**: Provides access to compliance documents and security reports to ensure that your ML environments meet security and regulatory standards.
  - **SageMaker Role Manager**: Simplifies the management of permissions and roles for SageMaker resources and services.

## Study Sheets

| **Term/Concept** | **Description** |
|------------------|-----------------|
| **Top-P** | 0 to 1 value — cumulative probability, balances randomness and accuracy, 1 is least random. |
| **Temperature** | Controls randomness—higher values lead to diverse, creative outputs; lower values make results more deterministic. |
| **Epochs** | Number of iterations on training a model—more is generally better, but too many can lead to overfitting. |
| **AWS Rekognition** | Computer vision/image recognition, used for object detection, facial analysis, and content moderation. |
| **AWS Textract** | **OCR** service — converts scanned images into text, structured data extraction from documents. |
| **Amazon Comprehend** | Extracts key phrases, entities, sentiment, and PII data from text. |
| **AWS Intelligent Document Processing (IDP)** | Automates the processing of unstructured documents (e.g., PDFs, invoices) using Textract, Comprehend, and A2I. |
| **Amazon Lex** | Service for **building chatbots** with natural conversational text or voice interfaces. |
| **Amazon Transcribe** | **Speech-to-text** service, including audio captioning. |
| **Amazon Polly** | **Text-to-speech** service for converting written text into spoken words. |
| **Amazon Kendra** | Intelligent document search engine with semantic understanding. |
| **Amazon Personalize** | Service for **personalized product recommendations**. |
| **Amazon Translate** | **Language translation** service. |
| **Amazon Forecast** | Provides time-series **forecasting**, e.g., inventory levels prediction. |
| **Amazon Fraud Detection** | Detects fraudulent activity, including online transactions and account takeovers. |
| **Amazon Q Business** | Generative AI-powered assistant for enterprise data processing and tasks. |
| **Amazon Macie** | **PII data** detection and anonymization service for data security. |
| **SageMaker Clarify** | **Bias detection** and explainability for machine learning models. |
| **SageMaker Ground Truth** | Provides **human labeling** for **model training** datasets. |
| **Amazon Augmented AI (A2I)** | **Human review** for low-confidence predictions during **inference**. |
| **AWS Data Exchange** | Access **third-party datasets** securely. |
| **AWS Glue Transformations** | ETL transformations like removing duplicates and filling missing values. |
| **Amazon SageMaker JumpStart** | Hub with pre-trained models and one-click deployments. |
| **Amazon SageMaker Canvas** | No-code tool for building and training machine learning models. |
| **SageMaker Notebook Instances** | Fully managed Jupyter notebook environments for ML development and experimentation. |
| **Fine-Tuning** | **Adjusting model weights** using **labeled data** to improve task performance. |
| **Domain adaptation fine-tuning** | Tailor a model for a **specific domain** like legal or medical using small datasets. |
| **Instruction-based fine-tuning** | Fine-tuning a model to **perform better on specific tasks**, e.g., classification. |
| **Continued-Pretraining** | Using **unlabelled data** to **expand a model's knowledge base**. |
| **Automatic Model Tuning (AMT)** | Automatically tunes hyperparameters to improve model performance. |
| **Data-Drift** | Input data changes, but the relationship between inputs and outputs stays the same (e.g., new demographic). |
| **Concept-Drift** | The relationship between inputs and outputs changes, meaning the model's learned patterns no longer apply (e.g., new fraud patterns). |
| **AWS Trusted Advisor** | Provides recommendations for cost, performance, and security improvements. |
| **Amazon Inspector** | Automated security assessments for application workloads. |
| **AWS PrivateLink** | Secure private connections between VPCs and AWS services. |
| **AWS Config** | Monitors and records AWS configuration changes for compliance. |
| **AWS CloudTrail** | Logs and tracks AWS API calls for auditing. |
| **BedRock GuardRails** | Prevents inappropriate foundation model outputs and restricts risky content. |
| **Gateway Endpoint for S3** | VPC endpoint that routes S3 traffic through AWS network without internet exposure. |
| **Responsible AI Dimensions** | Eight core areas: Fairness, Explainability, Safety, Privacy & Security, Veracity & Robustness, Controllability, Transparency, Governance. |
| **Postgres (Aurora or RDS)** | SQL database with vector database support for similarity search. |
| **Amazon DocumentDB** | JSON store, MongoDB-compatible with vector database support. |
| **Amazon Neptune** | Graph database with vector search capabilities. |
| **Amazon Neptune ML** | Uses Graph Neural Networks (GNNs) to predict outcomes from graph data. |
| **Amazon MemoryDB** | In-memory database with fast vector search capabilities. |
| **Amazon OpenSearch Service** | Search service with vector database support for similarity search. |
| **MSE (Mean Squared Error)** | Average squared difference between predicted and actual values, lower MSE indicates better model performance. |
| **RMSE (Root Mean Squared Error)** | Square root of MSE, more interpretable; lower RMSE is better. |
| **MAE (Mean Absolute Error)** | Average absolute differences, less sensitive to outliers than MSE. |
| **MAPE (Mean Absolute Percentage Error)** | Scale-independent percentage error metric for comparing models across scales. |
| **AWS Trainium** | **ML training** chips optimized for cost-effective, high-performance model training workloads. |
| **AWS Inferentia** | **ML inference** chips optimized for low-latency, high-throughput, cost-effective inference. |
| **Model Weights** | Parameters learned during training that determine feature transformation and output generation. |
| **Small Language Models (SLM)** | Compact models designed for **edge computing** with lower latency and resource requirements. |
| **Agents** | AI systems that **perceive, decide, and act** in environments to achieve specific goals. |
| **Cross-Validation** | Technique to assess model stability by training on multiple data splits. |
| **Regularization** | Methods to prevent overfitting (L1, L2, dropout, early stopping). |
| **Ensemble Methods** | Combine multiple models to improve performance and reduce variance. |
| **Feature Engineering** | Process of selecting, creating, and transforming features to improve model performance. |
| **Data Augmentation** | Techniques to increase dataset size by creating modified versions of existing data. |
| **Gateway Endpoints** | VPC endpoints for S3 and DynamoDB that route traffic through AWS network. |
| **Interface Endpoints** | VPC endpoints with private IP addresses for accessing most AWS services securely. |
| **Regional Services** | AWS services that operate within specific regions (EC2, S3, SageMaker). |
| **Global Services** | AWS services that operate globally with single endpoints (CloudFront, Route 53, IAM). |
| **CFG (Classifier-Free Guidance)** | Parameter controlling prompt adherence in diffusion models for image generation. |
| **Dynamic Prompt Engineering** | Adaptive prompting that changes based on context, user behavior, or model responses. |
| **Confusion Matrix** | Table showing TP, FP, TN, FN for classification evaluation. |
| **AUC-ROC** | Area under ROC curve measuring model's ability to distinguish between classes. |
| **Spot Instances** | Spare EC2 capacity at up to 90% discount for cost-effective AI model training. |
