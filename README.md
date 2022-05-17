# Feedback Natural Language Processing Analysis

## About

This project was developed as a means of taking common user feedbacks and being able to quickly understand 1) the percentage of sentiments (positive vs negative) and 2) what type of comment was provided.

## Model Explanation
As you may see, the only model actively used is the intent analysis, which helps explain what type of comment was provided (ie. question, comment, rant, etc). Older models of analysis (topic analysis and sentiment analysis) that were used for preliminary research can be viewed under the **Archived Models** folder.

# Environment Setup

## Initial venv Installation

Navigate to your source directory, and run the following command.

```
python -m venv ./venv
```

## Running venv

Before running any code, run the following:

```
venv/Scripts/activate
```

# Development

## Package Updates

To enable package updates and install packages, run the following command ONCE: 
```
pip install pipreqs
```

To update packages, run the following from the home directory:
```
pipreqs . --force
```

## Package Management

To install packages, run the following:

```
pip install -r requirements.txt
```
