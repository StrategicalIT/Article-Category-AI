from gravityai import gravityai as grav
import pickle
import pandas as pd

model = pickle.load(open('financial_text_classifier.pkl', 'rd' ))
tfidf_vectorsizer = pickle.load(open('financial_text_vectorizer.pkl', 'rd' ))
label_encoder = pickle.load(open('financial_text_encoder.pkl', 'rd' ))

def process(inpath, outpath):
    # read the file
    input_df = pd.read_csv(inpath)
    # vectorize the input
    features = tfidf_vectorsizer.transform(input_df['body'])
    # predict the classes
    predictions = model.predict(features)
    # convert the predictions to categories
    input_df['category'] = label_encoder.inverse_transform(predictions)
    # save the predictions
    output_df = input_df[['id', 'category']]
    output_df.to_csv(outpath, index=False)

    grav.wait_for_request(process)
    # grav.wait_for_requests(process)
