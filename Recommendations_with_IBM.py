#########################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import project_tests as t
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from subprocess import call

#%matplotlib inline

df = pd.read_csv('data/user-item-interactions.csv')
df_content = pd.read_csv('data/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

########################################################


user_article_counts = df.groupby("email")["article_id"].count()
# Fill in the median and maximum number of user_article interactios below
median_val =  np.median(user_article_counts) # 50% of individuals interact with ____ number of articles or fewer.
max_views_by_user =  user_article_counts.max() # The maximum number of user-article interactions by any 1 user is ______.

#######################################################
# Remove any rows that have the same article_id - only keep the first

# Remove duplicates based on article_id
df_interactions_cleaned = df.drop_duplicates(subset=["email", "article_id"], keep="first")

# Check the shape of the cleaned DataFrame
#print(f"Original df shape: {df.shape}")
#print(f"Cleaned df shape: {df_interactions_cleaned.shape}")

# Remove duplicates based on article_id and keep the first occurrence
df_content_cleaned = df_content.drop_duplicates(subset=["article_id"], keep="first")

# Check the shape of the cleaned DataFrame
#print(f"Original df_content shape: {df_content.shape}")
#print(f"Cleaned df_content shape: {df_content_cleaned.shape}")

unique_articles = df_interactions_cleaned["article_id"].nunique() # The number of unique articles that have at least one interaction
total_articles = df_content_cleaned.shape[0] # The number of unique articles on the IBM platform
unique_users = df_interactions_cleaned["email"].nunique() # The number of unique users
user_article_interactions = df.shape[0] # The number of user-article interactions

#####################################################
# Group by 'article_id' and count the number of interactions ('email') for each article
article_view_counts = df.groupby("article_id")["email"].count()

most_viewed_article_id = str(article_view_counts.idxmax()) # The most viewed article in the dataset as a string with one value following the decimal 
max_views = article_view_counts.max() # The most viewed article in the dataset was viewed how many times?

#######################################################
## No need to change the code here - this will be helpful for later parts of the notebook
# Run this cell to map the user email to a user_id column and remove the email column

def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

#newly_added####################################################
def get_top_articles(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    Description:
    This function identifies the top `n` articles with the highest number of interactions 
    and retrieves their titles.
    '''
    # Group the dataframe by article titles and count the interactions
    title_interactions = df.groupby('title').size().reset_index(name='interaction_count')
    
    # Sort the articles by interaction count in descending order
    sorted_titles = title_interactions.sort_values(by='interaction_count', ascending=False)
    
    # Select the top n titles
    top_articles = sorted_titles['title'].head(n).tolist()
    
    return top_articles

def get_top_article_ids(n, df=df):
    '''
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_article_ids - (list) A list of the top 'n' article IDs 
    
    Description:
    This function identifies the top `n` articles with the highest number of interactions 
    and retrieves their unique IDs.
    '''
    # Group the dataframe by article IDs and count the interactions
    article_interactions = df.groupby('article_id').size().reset_index(name='interaction_count')
    
    # Sort the articles by interaction count in descending order
    sorted_ids = article_interactions.sort_values(by='interaction_count', ascending=False)
    
    # Select the top n article IDs as strings
    top_article_ids = sorted_ids['article_id'].head(n).astype(str).tolist()
    
    return top_article_ids
########################################################################
# Create the user-article matrix with 1's and 0's using pivot table and NumPy

def create_user_item_matrix(df):
    '''
    INPUT:
    df - pandas dataframe with article_id, title, user_id columns
    
    OUTPUT:
    user_item - user-item matrix 
    
    Description:
    This function creates a matrix where each row corresponds to a user_id, 
    and each column corresponds to an article_id. The matrix contains 1 if the 
    user interacted with the article, and 0 otherwise.
    '''
    # Use pivot_table to construct the user-item matrix
    user_item = df.pivot_table(index='user_id', columns='article_id', aggfunc='size', fill_value=0)
    
    # Save the original index and columns
    index = user_item.index
    columns = user_item.columns
    
    # Convert all non-zero values to 1 using NumPy
    user_item = (user_item.values > 0).astype(int)
    
    # Reconstruct the DataFrame with the original index and columns
    user_item = pd.DataFrame(user_item, index=index, columns=columns)
    
    return user_item

# Generate the user-item matrix
user_item = create_user_item_matrix(df)
#newly_added######################################################################
def find_similar_users(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user_id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    similar_users - (list) an ordered list where the closest users (largest dot product users)
                    are listed first
    
    Description:
    Computes the similarity of every pair of users based on the dot product.
    Returns an ordered list of users based on similarity, excluding the user_id itself.
    '''
    # Compute similarity for all users with the given user
    user_similarity_scores = user_item @ user_item.loc[user_id]
    
    # Convert the Series to a DataFrame
    similarity_df = user_similarity_scores.reset_index()
    similarity_df.columns = ['user_id', 'similarity']
    
    # Exclude the target user and sort by similarity in descending order
    similarity_df = similarity_df[similarity_df['user_id'] != user_id].sort_values(by='similarity', ascending=False)
    
    # Extract the sorted user IDs as a list
    similar_users = similarity_df['user_id'].tolist()
    
    return similar_users  # Return the list of user_ids sorted by similarity
#newly_added3#############################################################################
def get_article_names(article_ids, df=df):
    '''
    INPUT:
    article_ids - (list) a list of article ids
    df - (pandas dataframe) df as defined at the top of the notebook
    
    OUTPUT:
    article_names - (list) a list of article names associated with the list of article ids 
                    (this is identified by the title column)
    
    Description:
    Returns the article titles corresponding to the provided article IDs.
    '''
    # Create a dictionary to map article IDs to titles
    id_to_title = df.set_index('article_id')['title'].to_dict()
    
    # Use list comprehension to retrieve titles for the given article IDs
    article_names = [id_to_title.get(float(article_id), "Unknown") for article_id in article_ids]
    
    return article_names  # Return the article names


def get_user_articles(user_id, user_item=user_item):
    '''
    INPUT:
    user_id - (int) a user id
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    article_ids - (list) a list of the article ids seen by the user
    article_names - (list) a list of article names associated with the list of article ids 
    
    Description:
    Extracts the article IDs and titles that the user has interacted with.
    '''
    # Identify the row corresponding to the user
    user_row = user_item.loc[user_id]
    
    # Find article IDs where the user has interacted
    article_ids = [col for col, val in zip(user_row.index, user_row.values) if val == 1]
    
    # Convert article IDs to strings
    article_ids = list(map(str, article_ids))
    
    # Retrieve article names using the helper function
    article_names = get_article_names(article_ids)
    
    return article_ids, article_names


def user_user_recs(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user
    
    Description:
    Finds articles the user hasn't seen before and provides recommendations 
    based on similar users.
    '''
    # Step 1: Fetch the articles already seen by the user
    user_articles, user_articles_names = get_user_articles(user_id)
    
    # Step 2: Find similar users and initialize recommendation tracking
    similar_users = find_similar_users(user_id)
    unseen_articles = []
    
    # Step 3: Traverse through similar users to gather recommendations
    for similar_user in similar_users:
        articles_by_user, articles_by_user_name = get_user_articles(similar_user)
        
        # Append unseen articles to the list
        unseen_articles.extend([article for article in articles_by_user if article not in user_articles])
        
        # Stop if enough recommendations are found
        if len(set(unseen_articles)) >= m:
            break
    
    # Step 4: Ensure consistent ordering and limit recommendations to 'm'
    unseen_articles = list(dict.fromkeys(unseen_articles))  # Remove duplicates while preserving order
    recs = unseen_articles[:m]
    
    return recs


#newl_added2########################################################################################
def get_top_sorted_users(user_id, df=df, user_item=user_item):
    '''
    INPUT:
    user_id - (int)
    df - (pandas dataframe) df as defined at the top of the notebook 
    user_item - (pandas dataframe) matrix of users by articles: 
                1's when a user has interacted with an article, 0 otherwise
    
    OUTPUT:
    neighbors_df - (pandas dataframe) a dataframe with:
                    neighbor_id - is a neighbor user_id
                    similarity - measure of the similarity of each user to the provided user_id
                    num_interactions - the number of articles viewed by the user
    
    Other Details - sort the neighbors_df by the similarity and then by number of interactions where 
                    highest of each is higher in the dataframe
    '''
    # Compute the similarity using numpy for efficiency and clarity
    user_similarity_scores = np.dot(user_item, user_item.loc[user_id].values)

    # Create a DataFrame to store the similarity scores
    similarity_data = pd.DataFrame({
        'user_id': user_item.index,
        'similarity': user_similarity_scores
    }).query('user_id != @user_id')  # Filter out the current user

    # Calculate the number of interactions for each user
    user_interaction_counts = df.groupby('user_id').size()
    user_interaction_counts.name = 'num_interactions'  # Assign name to the Series

    # Merge similarity scores with the number of interactions
    neighbors_df = pd.merge(similarity_data, user_interaction_counts, left_on='user_id', right_index=True, how='left')
    neighbors_df.rename(columns={'user_id': 'neighbor_id'}, inplace=True)

    # Sort by similarity and number of interactions in descending order.
    # Use neighbor_id as a tiebreaker in ascending order for consistent results.
    neighbors_df.sort_values(by=['similarity', 'num_interactions', 'neighbor_id'], 
                         ascending=[False, False, True], inplace=True)
    
    return neighbors_df

def user_user_recs_part2(user_id, m=10):
    '''
    INPUT:
    user_id - (int) a user id
    m - (int) the number of recommendations you want for the user
    
    OUTPUT:
    recs - (list) a list of recommendations for the user by article id
    rec_names - (list) a list of recommendations for the user by article title
    
    Description:
    Loops through users who have similar behaviors to the input user_id.
    Collects articles not seen by the user, prioritizing those with higher interaction counts.
    Stops after gathering 'm' recommendations.
    '''
    # Get articles already seen by the user
    user_articles = set(get_user_articles(user_id)[0])
    
    # Fetch sorted neighbors and their interactions
    sorted_neighbors = get_top_sorted_users(user_id)
    recommended = []
    top_articles_full_list = get_top_article_ids(df['article_id'].nunique())
    
    # Loop over sorted neighbors to find unseen articles
    for _, row in sorted_neighbors.iterrows():
        neighbor_articles = set(get_user_articles(row['neighbor_id'])[0])
        unseen = neighbor_articles - user_articles
        
        # Sort unseen articles by their global popularity ranking
        unseen_sorted = sorted(unseen, key=lambda x: top_articles_full_list.index(x))
        
        # Add unseen articles to the recommendations list while retaining order
        recommended.extend(article for article in unseen_sorted if article not in recommended)
        
        if len(recommended) >= m:
            break

    # Select the top 'm' recommendations while retaining their sorted order
    recs = recommended[:m]
    
    rec_names = get_article_names(recs)
    
    return recs, rec_names
##########################################################################
def make_content_recs(df_content, article_id, top_n=10):
    '''
    INPUT:
    df_content - (DataFrame) DataFrame containing article details (e.g., doc_body, doc_description, doc_full_name)
    article_id - (int) The article_id for which recommendations will be made
    top_n - (int) The number of top articles to return (default is 10)

    OUTPUT:
    recommendations - (DataFrame) DataFrame containing article IDs and doc_full_name of the most similar articles
                      Returns an empty DataFrame if no recommendations are available.
    '''
    
    # Check if the article_id exists in df_content
    if article_id not in df_content['article_id'].values:
        print(f"Error: Article ID {article_id} not found in df_content.")
        return pd.DataFrame()  # Return an empty DataFrame if the article ID is not found
    
    # Combine relevant content columns into a single string
    df_content['content'] = (
        df_content['doc_body'].fillna('') 
        + " " 
        + df_content['doc_description'].fillna('') 
        + " " 
        + df_content['doc_full_name'].fillna('')
    )
    
    # Remove irrelevant content like "Skip navigation" (if present)
    df_content['content'] = df_content['content'].replace(r"Skip navigation.*", "", regex=True)
    
    # Initialize TF-IDF vectorizer and fit on the combined content
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_content['content'])
    
    # Find the index of the article_id in df_content
    article_idx = df_content[df_content['article_id'] == article_id].index[0]
    
    # Compute the cosine similarity of the article with all other articles
    cosine_sim = cosine_similarity(tfidf_matrix[article_idx], tfidf_matrix).flatten()
    
    # Get the indices of the top_n most similar articles
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]
    
    # Get the article_ids and doc_full_name of the top_n most similar articles
    recommendations = df_content.iloc[similar_indices][['article_id', 'doc_full_name']].drop_duplicates()
    
    # Reset the index to start from 1 instead of 0
    recommendations.reset_index(drop=True, inplace=True)
    recommendations.index = recommendations.index + 1

    # Handle the case where no recommendations are found
    if recommendations.empty:
        print(f"No recommendations available for article ID {article_id}.")
    return recommendations
#################################################################################
#Function for brand new user
################################################################################
# For Matrix factorization
def perform_svd_on_user_item_matrix(user_item_matrix):
    '''
    INPUT:
    user_item_matrix - (DataFrame) The user-item matrix to perform SVD on
    
    OUTPUT:
    u, s, vt - SVD decomposition matrices (U, Sigma, V Transpose)
    '''
    u, s, vt = np.linalg.svd(user_item_matrix, full_matrices=False)
    return u, s, vt

################################################################################
# Splitting data into training and testing sets
df_train = df.head(40000)
df_test = df.tail(5993)

def create_test_and_train_user_item(df_train, df_test):
    '''
    INPUT:
    df_train - DataFrame containing the training data
    df_test - DataFrame containing the testing data
    
    OUTPUT:
    user_item_train - User-item matrix created from the training data
    user_item_test - User-item matrix created from the testing data
    test_idx - List of unique user IDs from the testing data
    test_arts - List of unique article IDs from the testing data
    '''
    # Create the user-item matrices
    user_item_train = create_user_item_matrix(df_train)
    user_item_test = create_user_item_matrix(df_test)
    
    # Extract user IDs and article IDs as lists
    test_idx = list(user_item_test.index.values)
    test_arts = list(user_item_test.columns.values)
    
    return user_item_train, user_item_test, test_idx, test_arts

# Calling the function to create matrices and retrieve lists of IDs
user_item_train, user_item_test, test_idx, test_arts = create_test_and_train_user_item(df_train, df_test)
###################################################################################
# fit SVD on the user_item_train matrix
u_train, s_train, vt_train = np.linalg.svd(user_item_train) # fit svd similar to above then use the cells below
###################################################################################
# Use these cells to see how well you can use the training 
# decomposition to predict on test data
# Identify common users and articles between training and test datasets
common_users = np.intersect1d(test_idx, user_item_train.index)
common_articles = np.intersect1d(test_arts, user_item_train.columns)

# Filter test dataset to include only common users and articles
filtered_user_item_test = user_item_test.loc[
    user_item_test.index.isin(common_users), 
    user_item_test.columns.isin(common_articles)
]

# Get indices for the common users and articles in the training dataset
train_user_indices = [np.where(user_item_train.index == user)[0][0] for user in common_users]
train_article_indices = [np.where(user_item_train.columns == article)[0][0] for article in common_articles]

# Initialize a list to store errors and define the latent feature range
latent_features_range = np.arange(10, 710, 20)
errors = []

# Loop over different numbers of latent features
for k in latent_features_range:
    # Create reduced matrices for the current number of latent features
    u_reduced = u_train[:, :k]
    s_reduced = np.diag(s_train[:k])
    vt_reduced = vt_train[:k, :]
    
    # Get the corresponding reduced matrices for the test dataset
    test_user_matrix = u_reduced[train_user_indices, :]
    test_article_matrix = vt_reduced[:, train_article_indices]
    
    # Predict user-item interactions using matrix multiplication
    predicted_matrix = np.around(np.dot(np.dot(test_user_matrix, s_reduced), test_article_matrix))
    
    # Compute the error between actual and predicted values
    differences = np.abs(filtered_user_item_test.values - predicted_matrix)
    total_error = differences.sum()
    errors.append(total_error)

# Compute accuracy and plot the curve
accuracy = 1 - np.array(errors) / filtered_user_item_test.size
plt.plot(latent_features_range, accuracy)
plt.xlabel('Number of Latent Features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Latent Features')
plt.show()
####################################################################################
####################################################################################
