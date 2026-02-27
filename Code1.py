import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
os.chdir("MLOPS2") # chaning dir to CWD
def create_dataframe():
    # create a dict and convert that into dataframe.
    data={
        "id":[1,2,3,4,5,6,7,8,9,10],
        "review":[
            "greate food and ambiene.",
            "terrible service.",
            "amazing experience!",
            "Food was cold.",
            "Loved the desserts.",
            "Not worth the money.",
            "Excellent customer service.",
            "THe place was too crowded.",
            "Best restaurant in town.",
            "Average experice."
        ]
    }
    df=pd.DataFrame(data)
    return df

# check if 'data folder exist, create if not and
# save the dataframe as data.csv
def save_dataframe(df):
    if not os.path.exists("data"):
        os.makedirs("data")
    df.to_csv("data/data.csv",index=False)
    print("data.csv is saved inside data folder")

# load data.csv vectorize and create 'k' new column

def process_data(k):
    df=pd.read_csv("data/data.csv")

    # apply vectorization
    
    vectorizer=CountVectorizer(max_features=k)
    vectorized_data=vectorizer.fit_transform(df["review"])
    feature_names=vectorizer.get_feature_names_out()

    # create a new dataframe with k new colllllumn

    vectorizer_df=pd.DataFrame(vectorized_data.toarray(),columns=feature_names)
    processed_df=pd.concat([df,vectorizer_df],axis=1)

    #save datafra,e to data directory
    processed_df.to_csv("data/processed_data.csv",index=False)
    print(f"processed_data.csv saved in data foler with {k} new column")
    return processed_df

#main execution
if __name__=="__main__":
    df=create_dataframe()

    save_dataframe(df)
    k=3
    processed_df=process_data(k)

    print(f"data shape:{df.shape}")
    print(f"processed data shape: {processed_df.shape}")
