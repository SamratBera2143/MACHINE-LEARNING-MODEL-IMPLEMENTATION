# %% [markdown]
# # Spam Email Detection Using Machine Learning
# 
# ## Internship Task 4 - ML Model Implementation
# 
# ### Objectives:
# - Build a predictive model to classify emails as spam or ham (not spam)
# - Use Scikit-learn for model implementation
# - Evaluate model performance using appropriate metrics

# %% [markdown]
# ## Step 1: Import Required Libraries 

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

# Warning suppression
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("Libraries imported successfully!")

# %% [markdown]
# ## Step 2: Load and Explore the Dataset
# 
# We'll use the popular SMS Spam Collection dataset.

# %%
# Load the dataset
url = 'https://raw.githubusercontent.com/justmarkham/pydata-dc-2016-tutorial/master/sms.tsv'
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print(f"Dataset shape: {df.shape}")
df.head(10)

# %%
# Check dataset information
df.info()

# %%
# Check class distribution
df['label'].value_counts()

# %%
# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=df, palette='viridis')
plt.title('Distribution of Email Classes', fontsize=16)
plt.xlabel('Label', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Add percentage labels
total = len(df)
for p in plt.gca().patches:
    percentage = f'{100 * p.get_height() / total:.1f}%'
    plt.gca().annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height() + 5), 
                       ha='center', va='bottom', fontsize=12)
plt.show()

# %%
# Check for missing values
df.isnull().sum()

# %%
# Convert labels to binary (0 = ham, 1 = spam)
df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})
df.head()

# %%
# Basic text statistics
df['message_length'] = df['message'].apply(len)
df['word_count'] = df['message'].apply(lambda x: len(x.split()))

# Statistics by class
print("Message Statistics by Class:")
df.groupby('label')[['message_length', 'word_count']].describe().T

# %%
# Visualize message length distribution by class
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True, palette=['green', 'red'])
plt.title('Message Length Distribution by Class', fontsize=14)
plt.xlabel('Message Length (characters)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.boxplot(x='label', y='message_length', data=df, palette=['green', 'red'])
plt.title('Message Length Boxplot by Class', fontsize=14)
plt.xlabel('Class')
plt.ylabel('Message Length')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 3: Text Preprocessing

# %%
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean and preprocess text data:
    - Convert to lowercase
    - Remove special characters and digits
    - Remove stopwords
    - Apply stemming
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits (keep only alphabets)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization and stopword removal with stemming
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Apply text preprocessing
print("Applying text preprocessing...")
df['cleaned_message'] = df['message'].apply(clean_text)

print("Sample original message:")
print(df['message'].iloc[0])
print("\nCleaned message:")
print(df['cleaned_message'].iloc[0])

# %% [markdown]
# ## Step 4: Feature Extraction (TF-IDF Vectorization)

# %%
# Split data into features and target
X = df['cleaned_message']
y = df['label_binary']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# %%
# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,  # Limit to top 5000 features
    ngram_range=(1, 2),  # Unigrams and bigrams
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.95  # Ignore terms that appear in more than 95% of documents
)

# Fit and transform training data, transform test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"TF-IDF feature matrix shape (train): {X_train_tfidf.shape}")
print(f"TF-IDF feature matrix shape (test): {X_test_tfidf.shape}")

# %%
# Display top features
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"Top 20 features: {feature_names[:20]}")

# %% [markdown]
# ## Step 5: Model Implementation

# %%
# Define models to compare
models = {
    'Multinomial Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Support Vector Machine': SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate each model
results = {}
predictions = {}

for name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Training {name}...")
    
    # Train the model
    model.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Store predictions for later analysis
    predictions[name] = (y_pred, y_pred_proba)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# %% [markdown]
# ## Step 6: Model Evaluation and Comparison

# %%
# Create DataFrame for results comparison
results_df = pd.DataFrame(results).T
results_df = results_df.round(4)

print("Model Performance Comparison:")
results_df

# %%
# Visualize model comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

metrics = ['accuracy', 'precision', 'recall', 'f1_score']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

for i, metric in enumerate(metrics):
    results_df[metric].sort_values().plot(kind='barh', ax=axes[i], color=colors[i], 
                                          edgecolor='black', linewidth=1)
    axes[i].set_title(f'{metric.capitalize()} Comparison', fontsize=14)
    axes[i].set_xlabel(metric.capitalize(), fontsize=12)
    axes[i].set_xlim(0, 1)
    axes[i].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for j, v in enumerate(results_df[metric].sort_values()):
        axes[i].text(v + 0.02, j, f'{v:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.show()

# %%
# Confusion Matrix for the best model
best_model_name = results_df['accuracy'].idxmax()
best_model = models[best_model_name]
y_pred_best, _ = predictions[best_model_name]

print(f"Best Model: {best_model_name}")
print(f"Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Ham', 'Spam'], 
            yticklabels=['Ham', 'Spam'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.show()

# %%
# Classification Report for the best model
print(f"Classification Report - {best_model_name}:")
print("-" * 60)
print(classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam']))

# %% [markdown]
# ## Step 7: ROC-AUC Analysis

# %%
# ROC-AUC for models with probability predictions
plt.figure(figsize=(10, 8))

for name, (_, y_pred_proba) in predictions.items():
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves for All Models', fontsize=16)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.show()

# %%
# ROC-AUC scores
roc_auc_scores = {}
for name, (_, y_pred_proba) in predictions.items():
    if y_pred_proba is not None:
        auc = roc_auc_score(y_test, y_pred_proba)
        roc_auc_scores[name] = auc

print("ROC-AUC Scores:")
for name, auc in roc_auc_scores.items():
    print(f"{name}: {auc:.4f}")

# %% [markdown]
# ## Step 8: Cross-Validation

# %%
# Perform cross-validation on the best model
cv_scores = cross_val_score(best_model, X_train_tfidf, y_train, cv=5, scoring='accuracy')

print(f"Cross-Validation Results for {best_model_name}:")
print(f"CV Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# %% [markdown]
# ## Step 9: Feature Importance Analysis (for Random Forest)

# %%
# If Random Forest is the best model, analyze feature importance
if best_model_name == 'Random Forest':
    feature_importances = best_model.feature_importances_
    
    # Get top 20 most important features
    indices = np.argsort(feature_importances)[::-1][:20]
    top_features = [feature_names[i] for i in indices]
    top_importances = feature_importances[indices]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances, color='skyblue')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 20 Most Important Features (Random Forest)', fontsize=16)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Step 10: Testing the Model on New Emails

# %%
def predict_email(message, model=tfidf_vectorizer, classifier=best_model):
    """
    Predict whether an email is spam or ham.
    """
    # Clean the message
    cleaned = clean_text(message)
    
    # Transform using the fitted TF-IDF vectorizer
    message_tfidf = model.transform([cleaned])
    
    # Make prediction
    prediction = classifier.predict(message_tfidf)[0]
    proba = classifier.predict_proba(message_tfidf)[0][1] if hasattr(classifier, 'predict_proba') else None
    
    label = "SPAM" if prediction == 1 else "HAM"
    confidence = f"{proba*100:.2f}%" if proba is not None else "N/A"
    
    return label, confidence

# Test with sample messages
test_messages = [
    "Congratulations! You've won a free iPhone. Click here to claim your prize.",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your bank account has been compromised. Please verify your details now.",
    "Thanks for your email. I'll get back to you shortly.",
    "FREE entry to the VIP concert. Text WIN to 55555 now!",
    "Hi Mom, I'll be home for dinner tonight. Can you pick up some milk?"
]

print(f"\n{'='*80}")
print(f"Testing Best Model: {best_model_name}")
print(f"{'='*80}")

for msg in test_messages:
    label, confidence = predict_email(msg)
    print(f"\nMessage: {msg}")
    print(f"Prediction: {label} (Confidence: {confidence})")

# %% [markdown]
# ## Step 11: Save the Model (Optional)

# %%
# Uncomment to save the model using joblib or pickle
# import joblib
# 
# # Save the best model and vectorizer
# joblib.dump(best_model, 'spam_classifier_model.pkl')
# joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
# 
# print("Model saved successfully!")

# %% [markdown]
# ## Summary and Conclusion

# %%
print("\n" + "="*80)
print("SUMMARY AND CONCLUSION")
print("="*80)

print(f"""
📊 **Dataset**: SMS Spam Collection
   - Total messages: {len(df)}
   - Training samples: {len(X_train)}
   - Testing samples: {len(X_test)}

🤖 **Models Evaluated**: {', '.join(models.keys())}

🏆 **Best Model**: {best_model_name}
   - Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f}
   - Precision: {results_df.loc[best_model_name, 'precision']:.4f}
   - Recall: {results_df.loc[best_model_name, 'recall']:.4f}
   - F1 Score: {results_df.loc[best_model_name, 'f1_score']:.4f}

📈 **Key Insights**:
   1. The {best_model_name} achieved the highest accuracy of {results_df.loc[best_model_name, 'accuracy']*100:.2f}%
   2. All models performed well, with accuracy ranging from {results_df['accuracy'].min()*100:.2f}% to {results_df['accuracy'].max()*100:.2f}%
   3. TF-IDF vectorization effectively captured important textual features for classification
   4. The model can reliably distinguish between spam and ham messages

✅ **Task Completed**: A robust spam detection model has been successfully implemented and evaluated using Scikit-learn.
""")

# %%
print("\n" + "="*80)
print("INTERNSHIP TASK 4 - COMPLETED")
print("="*80)
