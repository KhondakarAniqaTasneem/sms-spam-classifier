SMS Spam Classifier (TF-IDF + Linear SVM)

A machine learning system that reads an SMS message and decides whether it is Spam or Ham (a normal message).

1. Problem

Spam SMS messages try to trick people with fake prizes, links and phone numbers. The goal is to build a model that detects them automatically.

Type of problem: binary text classification
Classes: Ham = 0, Spam = 1
Main challenge: the data is imbalanced (only about 13% of messages are spam), so accuracy alone is misleading.
2. Dataset
Name: SMS Spam Collection (UCI)
Source: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
Size: 5,572 messages (about 5,169 after removing duplicates)
Columns used: v1 (label: ham/spam) and v2 (message text)

The dataset file is not included in this repository. Download it from the Kaggle link and save it as SMS.csv.

3. How the system works (big picture)
Raw SMS
   |
   v
Text cleaning (keep spam clues as tokens)
   |
   v
Feature extraction (word + character TF-IDF)
   |
   v
Linear SVM classifier
   |
   v
Spam or Ham
4. Step-by-step explanation
Step 1: Data preprocessing

Computers cannot understand raw text, so the messages are cleaned first. The key idea: do not delete spam clues, turn them into tokens (simple words) so the model can still learn from them.

In the SMS	Replaced with	Why it helps
Links (www.win.com)	urltoken	Spam often has links
Emails	emailtoken	Spam often has emails
Numbers with 5+ digits	longnumtoken	Phone numbers and shortcodes
Other numbers	numtoken	Prices, counts
£ $ €	currencytoken	Money is a spam signal
!	exclamtoken	Spam uses many exclamation marks
ALL-CAPS words (FREE, URGENT)	capstoken	Spam "shouts"

Example

Before: URGENT! Call 09061701461 now to claim your £900 prize. Visit www.win.com
After : urgent exclamtoken call longnumtoken now to claim your currencytoken
        numtoken prize visit urltoken capstoken

Other cleaning steps:

Convert to lowercase (after counting the ALL-CAPS words)
Remove other punctuation and extra spaces
Remove duplicate messages
No stopword removal or lemmatization (the SVM handles these fine, and they can carry style clues)
Step 2: Train/test split
80% of the data for training, 20% for testing
Stratified, so both parts keep the same spam ratio
random_state=42 so results are repeatable
Step 3: Feature extraction (TF-IDF)

TF-IDF converts text into numbers. A word gets a high score if it is frequent in a message but rare across all messages, which makes it a useful clue.

Two TF-IDF vectorizers are combined (FeatureUnion):

Type	Settings	What it catches
Word TF-IDF	1–2 word n-grams, min_df=2, sublinear_tf=True	Words and phrases like "free entry", "call now"
Character TF-IDF	char_wb, 2–5 character n-grams	Misspellings and tricks like "fr33", "txt4u"
Step 4: Classifier (Linear SVM)
Model: LinearSVC (Support Vector Machine with a linear kernel)
Idea: it finds the best dividing line (hyperplane) between spam and ham with the widest possible margin.
Why it suits this problem: it works very well on high-dimensional, sparse text data, and it trains fast.
class_weight='balanced': mistakes on spam count more, because spam is the rare class.
Step 5: Hyperparameter tuning
GridSearchCV with 5-fold stratified cross-validation
Tuned settings: C (0.1, 0.3, 1, 3, 10) and the character n-gram range
Scored on F1 (not accuracy), because F1 balances precision and recall on imbalanced data
Cross-validation uses the training data only, so the test set stays unseen (no data leakage)
Step 6: Evaluation
Metric	Meaning
Accuracy	% of all messages classified correctly
Precision	Of messages flagged as spam, how many really were spam
Recall	Of all real spam, how many were caught
F1-score	Balance of precision and recall
Confusion matrix	Counts of right and wrong predictions per class
ROC-AUC	How well the model separates spam from ham (1.0 = perfect)

Extras included: error analysis (the misclassified messages), decision-threshold tuning, and the top spam/ham features learned by the SVM.


6. Results

Run the notebook to get your own scores, then fill in this table:

Metric	Baseline	Improved
Accuracy		
Precision		
Recall		
F1		

Libraries used: pandas, numpy, matplotlib, seaborn, nltk, wordcloud, scikit-learn, joblib

8. Using the model on a new message
python
print(predict_sms("URGENT! Call 09061701461 now to claim your £900 prize"))
# Spam (score: ...)

print(predict_sms("Hi, are we meeting tomorrow at 3 PM?"))
# Ham (score: ...)

A score above 0 means Spam. A bigger number means the model is more confident.

9. Short explanation (for presentations)

"I built an SMS spam detector. First, I clean the text but keep important spam clues like links, phone numbers, money symbols and capital letters by converting them into tokens. Then I convert the text to numbers using word and character TF-IDF. Finally, I classify with a Linear SVM using balanced class weights, tuned for F1 score with cross-validation on the training data only. The system is evaluated with precision, recall, F1, a confusion matrix and ROC-AUC."

10. Possible future improvements
Try other models (Logistic Regression, Naive Bayes, ensembles) or DistilBERT
Build a web app with Streamlit, Gradio or Flask
Show a spam probability instead of only Spam/Ham
Support more languages
Add a feedback loop where users flag mistakes and the model retrains
