# Recommendations with IBM

This repository contains the implementation of a recommendation system for IBM's Watson Studio platform, developed as part of the Udacity Data Science Nanodegree Program. The project leverages various recommendation techniques to analyze user-article interactions and suggest articles to users based on their preferences and behaviors.

## GitHub Repository

The source code for this project is hosted on GitHub:  
[Recommendations with IBM App Repository](https://github.com/Makarand7/Recommendations_with_IBM)

## Web Application 

Web Application for this project is hosted on Streamlit:  
[Recommendations with IBM Web App](https://recommendationswithibm-kbi4ibaoccgd6wyt5soorr.streamlit.app/)

## Project Overview

The goal of this project is to build a recommendation system that improves user engagement by suggesting relevant articles on the IBM Watson Studio platform. The project includes:

- **Exploratory Data Analysis (EDA)**: Understanding user-article interactions, article popularity, and dataset characteristics.
- **Rank-Based Recommendations**: A simple yet effective technique for new users based on article popularity.
- **User-User Collaborative Filtering**: Leveraging user similarities to recommend articles.
- **Content-Based Recommendations**: Recommending articles based on their textual content using Natural Language Processing (NLP) techniques.
- **Matrix Factorization**: Utilizing Singular Value Decomposition (SVD) to uncover latent patterns in the user-item matrix and improve recommendations.
- **A/B Testing Plan**: A detailed plan to evaluate and validate the effectiveness of the recommendation system.

## Key Features

### Rank-Based Recommendations:
- Recommends the most popular articles based on user interaction counts.

### User-User Collaborative Filtering:
- Uses user similarities to suggest articles that similar users have interacted with.

### Content-Based Recommendations (Optional):
- Suggests articles based on their content similarity, utilizing TF-IDF Vectorization and cosine similarity.

### Matrix Factorization:
- Applies SVD to the user-item interaction matrix to make personalized recommendations.
- Includes an **Accuracy vs. Number of Latent Features** plot to evaluate performance.

### A/B Testing Plan:
- A structured plan to validate the system's performance using randomized controlled trials.


## Project Structure

The project folder is organized as follows:


```
Recommendations_with_IBM
├── app.py
├── data
│   ├── articles_community.csv
│   └── user-item-interactions.csv
├── Recommendations_with_IBM.ipynb
├── Recommendations_with_IBM.html
├── Recommendations_with_IBM.py
├── project_tests.py
├── top_5.p
├── top_10.p
├── top20.p
├── user_item_matrix.p
├── requirements.txt
├── .gitignore
├── images
│   ├── GitHub_logo.jpg
│   ├── Linkedin_logo.jpg
│   └── Udacity_logo.png
└── README.md
```


## Running the Project

### Set Up the Environment:
Install dependencies from `requirements.txt` using:
	pip install -r requirements.txt

### Launch the Streamlit app:

streamlit run app.py

## Explore the Features:

Navigate through the web app to explore the various recommendation techniques.

## Results and Recommendations
### SVD Analysis

This project implements several recommendation techniques:
- **Rank-Based Recommendations**: Suggests top articles based on popularity.
- **User-User Collaborative Filtering**: Recommends articles based on user similarities.
- **Content-Based Recommendations**: Suggests articles similar in content to a given article.
- **Matrix Factorization**: Demonstrates SVD to uncover latent patterns in the user-item matrix.

### SVD Analysis and A/B Testing Plan
For the test data, accuracy decreases as the number of latent features increases. This may occur because only 20 users are shared between the training and test datasets, leading to limited generalization. Additionally, there is a significant imbalance in user-article interactions, with most values being zero (sparse data).

While adding latent features improves accuracy on the training data, it does not significantly enhance the test data's performance. This suggests potential overfitting. To address this, performing cross-validation could help identify the optimal number of latent features that generalize well across different subsets of the data.

### Improvements:
- **Model Enhancements**: Incorporating regularization techniques (e.g., L2 regularization) or reducing the dimensionality of the dataset through feature engineering could mitigate overfitting.
- **Alternative Algorithms**: Exploring other matrix factorization methods, such as Non-Negative Matrix Factorization (NMF), might yield better results for sparse data.

### Evaluation:
- **SVD** provides a clear improvement over rank-based recommendations and user-user collaborative filtering. To further evaluate its effectiveness, conducting A/B testing is recommended.

### Steps for A/B Testing:
1. Ensure that each recommendation system receives an equal and randomized sample of users to avoid bias.
2. Test only one variable at a time, such as comparing two recommendation systems, without testing other features concurrently.
3. Allow the test to run for a sufficient duration to collect meaningful data for statistical significance.
4. Complement quantitative data with qualitative insights by collecting user feedback through surveys or polls.

### Final Recommendations
1. For new users: Use rank-based recommendations.
2. For returning users: Use collaborative filtering for personalized recommendations.
3. To enhance recommendations, integrate content-based filtering for articles not interacted with.


## How to Contribute
We welcome contributions to improve this project! Feel free to fork the repository, raise issues, or submit pull requests.

## Acknowledgments
This project was completed as part of the Udacity Data Science Nanodegree Program. Special thanks to IBM for providing the dataset and Udacity for the learning opportunity.

