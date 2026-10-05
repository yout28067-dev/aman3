# Phishing Email Detection Model

A Scikit-learn classifier that labels email content as **Phishing** or **Safe**. It combines word and character TF-IDF features with signals often found in malicious mail: URL count, IP-address URLs, risky domains, URL shorteners, urgency/account keywords, punctuation, and unusual capitalisation.

## Run

```powershell
python -m pip install -r requirements.txt
python phishing_detector.py
```

The script creates `confusion_matrix.png`, prints accuracy, a confusion matrix, and the per-class precision/recall report. To test a message while evaluating:

```powershell
python phishing_detector.py --predict "Verify your password immediately at http://account-check.top"
```

## Use your own data

Replace or extend `data/emails.csv`. It must contain these headers:

```csv
text,label
"Email body or subject and body",phishing
"A normal message",safe
```

Accepted safe labels include `safe`, `legitimate`, `ham`, and `0`; accepted phishing labels include `phishing`, `spam`, and `1`.

The included data is intentionally small and illustrative. For a production-quality model, train and test on a much larger, representative labelled corpus and keep examples from the same campaign in only one split to avoid overly optimistic accuracy.
