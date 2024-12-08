import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Recommendations_with_IBM import (
    get_top_articles,
    get_top_article_ids,
    get_article_names,
    get_user_articles,
    user_user_recs,
    user_user_recs_part2,
    make_content_recs,
    perform_svd_on_user_item_matrix,
    user_item,
    df,
    df_content,
    df_train,
    df_test,
    create_test_and_train_user_item,
    user_item_train,
    user_item_test,
    u_train,
    s_train,
    vt_train,
    #num_latent_feats,
    latent_features_range,
    #sum_errs,
    errors,
    filtered_user_item_test
)

# Title and description
st.title("Recommendations with IBM")
st.write("Explore various types of recommendation techniques implemented as part of the project.")

# Sidebar for navigation
st.sidebar.markdown(
    """
    <style>
        .logo-container {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        .logo-container a {
            text-decoration: none;
            display: flex;
            align-items: center;
            margin-right: 10px;
            font-size: 16px;
            color: #0073e6;
        }
    </style>
    <div class="logo-container">
        <a href="https://github.com/Makarand7/Recommendations_with_IBM" target="_blank">
            GitHub
        </a>
        <a href="https://www.linkedin.com/in/makarand-52b930324/" target="_blank">
            LinkedIn
        </a>
        <a href="https://www.udacity.com/" target="_blank">
            Udacity
        </a>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to:",
    [
        "Project Overview",
        "Rank-Based Recommendations",
        "User-User Collaborative Filtering",
        "Content-Based Recommendations",
        "Matrix Factorization",
        "Project Analysis and Final Recommendations",
    ],
)

# Project Overview
if section == "Project Overview":
    st.header("Project Overview")
    st.write(
        """
        This project involves building a recommendation system for IBM's Watson Studio platform. It explores different types of recommendation techniques:
        1. **Rank-Based Recommendations**: Recommends articles based on popularity.
        2. **User-User Collaborative Filtering**: Suggests articles based on user similarity.
        3. **Content-Based Recommendations**: Uses Natural Language Processing (NLP) to recommend articles similar in content.
        4. **Matrix Factorization**: Applies Singular Value Decomposition (SVD) to analyze user-item interactions.
        
        Each technique has been implemented with its unique strengths and weaknesses to provide tailored recommendations for various scenarios.
        """
    )

# Rank-Based Recommendations
elif section == "Rank-Based Recommendations":
    st.header("Rank-Based Recommendations")
    n = st.number_input("How many top articles do you want?", min_value=1, step=1, value=10)
    if st.button("Get Top Articles"):
        top_articles = get_top_articles(n)
        st.write("### Top Article Titles")
        for idx, title in enumerate(top_articles, start=1):
            st.write(f"{idx}. {title}")

    if st.button("Get Top Article IDs"):
        top_article_ids = get_top_article_ids(n)
        st.write("### Top Article IDs")
        st.write(top_article_ids)

# User-User Collaborative Filtering
elif section == "User-User Collaborative Filtering":
    st.header("User-User Collaborative Filtering")
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    m = st.number_input("How many recommendations do you want?", min_value=1, step=1, value=10)
    if st.button("Get Recommendations"):
        try:
            rec_ids, rec_titles = user_user_recs_part2(user_id, m)
            st.write("### Recommendations")
            st.write("#### Article IDs:")
            st.write(rec_ids)
            st.write("#### Article Titles:")
            st.write(rec_titles)
        except Exception as e:
            st.error(f"Error: {e}")

    article_ids_input = st.text_input(
        "Enter Article IDs (comma-separated) to get article names:", value=""
    )
    if st.button("Get Article Names"):
        try:
            article_ids = article_ids_input.split(",")
            article_ids = [id.strip() for id in article_ids]
            article_names = get_article_names(article_ids)
            st.write("### Article Names:")
            for name in article_names:
                st.write(name)
        except Exception as e:
            st.error(f"Error: {e}")

# Content-Based Recommendations
elif section == "Content-Based Recommendations":
    st.header("Content-Based Recommendations")
    article_id = st.number_input("Enter Article ID for recommendations:", min_value=0.0, step=1.0)
    top_n = st.number_input("How many similar articles do you want?", min_value=1, step=1, value=10)
    if st.button("Get Content-Based Recommendations"):
        recommendations = make_content_recs(df_content, article_id, top_n)
        if recommendations.empty:
            st.warning(f"No recommendations available for Article ID {article_id}.")
        else:
            st.write("### Content-Based Recommendations")
            st.write(recommendations)

# Matrix Factorization
elif section == "Matrix Factorization":
    st.header("Matrix Factorization")
    st.write(
        "This section demonstrates the application of Singular Value Decomposition (SVD) "
        "on the user-item matrix."
    )
    if st.button("Perform SVD"):
        try:
            u, s, vt = perform_svd_on_user_item_matrix(user_item)
            st.write("### SVD Results")
            st.write("U Matrix:")
            st.write(u)
            st.write("Sigma (Diagonal):")
            st.write(s)
            st.write("V Transpose:")
            st.write(vt)

            # Plot Accuracy vs. Number of Latent Features
            fig, ax = plt.subplots()
            #ax.plot(num_latent_feats, 1 - np.array(sum_errs) / df.shape[0])
            #ax.plot(latent_features_range, 1 - np.array(errors) / df.shape[0])
            ax.plot(latent_features_range, 1 - np.array(errors) / filtered_user_item_test.size)
            ax.set_xlabel("Number of Latent Features")
            ax.set_ylabel("Accuracy")
            ax.set_title("Accuracy vs. Number of Latent Features")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

# Project Analysis and Final Recommendations
elif section == "Project Analysis and Final Recommendations":
    st.header("Project Analysis and Final Recommendations")
    st.write(
        """
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
        """
    )
