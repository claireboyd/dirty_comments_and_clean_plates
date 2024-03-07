import pandas as pd
import os
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sample_n", type=int, required=False)
args = parser.parse_args()

def concatenante_reviews(sample_n: int = 0) -> pd.DataFrame:
    '''
    Joins human and GPT-generated reviews
    '''
    real_df = pd.read_csv("data/yelp/yelp_verified_slim.csv")
    real_df['label'] = 'HUMAN'
    real_df['source'] = 'Yelp'
    real_df['num_shot'] = 0

    real_df = real_df[['text', 'label', 'source', 'num_shot']]
    if sample_n:
        real_df = real_df.sample(n=sample_n)

    gpt_dir = 'chat_gpt/gpt_reviews/'
    gpt_files = os.listdir(gpt_dir)

    dfs = []
    for gpt_file in gpt_files: 
        gpt_df = pd.read_csv(f'{gpt_dir}/{gpt_file}')
        gpt_df = gpt_df.rename(columns={'REVIEW' : 'text', 'LABEL': 'label', 'MODEL': 'source', 'NUM_SHOTS' : 'num_shot'})
        dfs.append(gpt_df)

    dfs.append(real_df)
    all_df = pd.concat(dfs)
    return all_df 


def test_train_val_split(df: pd.DataFrame, out_folder: str):
    '''
    Split dataframe
    '''
    os.makedirs(out_folder, exist_ok=True)

    df = df.reset_index().rename(columns={"index": "uuid"})
    val = df.sample(frac=0.10)
    leftover = df[~df["uuid"].isin(val["uuid"].to_list())]

    train, test = train_test_split(leftover, train_size=0.90, shuffle=True)

    val.drop(columns=["uuid"]).to_csv(f"{out_folder}/val.csv", index=False)
    train.drop(columns=["uuid"]).to_csv(f"{out_folder}/train.csv", index=False)
    test.drop(columns=["uuid"]).to_csv(f"{out_folder}/test.csv", index=False)
  

if __name__ == '__main__':
    test_train_val_split(concatenante_reviews(args.sample_n), out_folder='data/fake_reviews')


    



