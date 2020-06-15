import sys
import pandas as pd
from sqlalchemy import create_engine

# Sample Call :
#. python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

def load_data(messages_filepath, categories_filepath):
    
    # Load messages
    messages = pd.read_csv(messages_filepath)
    
    # Load categories
    categories = pd.read_csv(categories_filepath)
    
    # Merge two datasets
    df = messages.merge(categories, on = ['id'], how = 'inner')
    
    # Return merged dataframe
    return df


def clean_data(df):
    
    # Split category column
    categories = df['categories'].str.split(pat=';',expand=True)
    
    # Get all category names in a list and rename the split columns
    row = categories.iloc[1].values.tolist()
    category_colnames = []
    for val in row:
        category_colnames.append(val.split('-')[0])
    categories.columns = category_colnames
    
    # set each value to be the last character of the string
    categories = categories.apply(lambda x: x.astype(str).str[-1] if x.name in category_colnames else x)
    
    # convert column from string to numeric
    categories = categories.apply(lambda x: x.astype(int) if x.name in category_colnames else x)
    
    #Drop the categories column from the df dataframe since it is no longer needed
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1, join = 'inner')
    
    # drop duplicate messages
    df.drop_duplicates(subset='message', inplace=True)
    
    # Drop null features
    df.dropna(subset=category_colnames, inplace=True)
    
    # Cleanup 2 to 0 as found later in replace column
    df.related.replace(2, 0, inplace=True)
    
    return df

def save_data(df, database_filename):
    
    # Create a connection string
    connection_string = 'sqlite:///'+ database_filename
    
    # Create engine object
    engine = create_engine(connection_string)
    
    # Drop table if exists, then create
    sql = 'DROP TABLE IF EXISTS messages;'
    result = engine.execute(sql)
    
    # Load the dataframe into the table : messages
    df.to_sql('messages', engine, index=False)  

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()