import os
import cv2
import easyocr
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from drug_named_entity_recognition.drugs_finder import find_drugs
from difflib import get_close_matches
import pandas as pd
import openai


import html

app = Flask(__name__)

# Folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
DATA_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_area(bbox):
    # bbox is a list of 4 points: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    # We can use the width and height of the bounding box to calculate the area
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    x3, y3 = bbox[2]
    x4, y4 = bbox[3]

    # Calculate the width and height of the bounding box
    width = max(x1, x2, x3, x4) - min(x1, x2, x3, x4)
    height = max(y1, y2, y3, y4) - min(y1, y2, y3, y4)

    return width * height

def get_top_n_largest_text(image_path, n):
  # Load your bottle image using OpenCV'  # Replace with the path to your bottle image
  img = cv2.imread(image_path)

  if img is None:
    print(f"Error loading image: {image_path}")
    return []

  # Convert to grayscale
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # Optionally, convert to binary (black and white) using a threshold
  _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

  # Perform OCR on the image
  results = reader.readtext(binary_img)

  # Print and display the detected text
  detected_text = []

  # Reorder the results by the size of the bounding box
  sorted_results = sorted(results, key=lambda x: calculate_area(x[0]), reverse=True)

  detected_text = []
  # Now sorted_results contains the OCR results ordered by the bounding box area
  for result in sorted_results:
      bbox, text, _ = result  # Unpack the result (bbox, text, and confidence)
      #print(f"Detected Text: {text}, Bounding Box: {bbox}, Area: {calculate_area(bbox)}")
      if(result[2] > 0.5):
        detected_text.append(text)
  return detected_text[:n]




def find_close_matches(word, word_list, cutoff=0.6, n=3):
    """
    Find close matches for a potentially misspelled word in a given list.

    :param word: The word to search for (potentially misspelled)
    :param word_list: The list of words to search in
    :param cutoff: The similarity threshold (default: 0.6)
    :param n: Maximum number of close matches to return (default: 3)
    :return: A list of close matches
    """
    print('inside find close matches with word: ', word)
    return get_close_matches(word, word_list, n=n, cutoff=cutoff)

def get_name_from_picture(image_path, df):
    detected_text = get_top_n_largest_text(image_path, 3)
    print('Got detected Text: ')
    print(detected_text)
    drug_results = find_drugs(detected_text)
    names_list = []
    for drug_top in drug_results:
        #print(drug_top)
        for drug in drug_top:
            if drug:
            #combine name and synonyms
                if 'name' in drug.keys():
                    names_list += [drug['name'].lower()]
                if 'synonyms' in drug.keys():
                    names_list += [syn.lower() for syn in drug['synonyms']]
    for name in names_list:
        print('got names to compare against', name)
    matches = []
    if names_list:
        for word in detected_text:
            if word.lower() in names_list:
                matches += [word]
                break
        
    else:
        matches = []
        for word in detected_text:
            print('looking for close match of ', word)
            matches = find_close_matches(word, df.drugName.values.tolist(), cutoff=0.7)
            print('Matches are: ', matches)
            if matches:
                break

        print('returning matches')
    return matches

def filter_drug(df, drugName):
  df['drugName'] = df['drugName'].apply(lambda x: x.lower())
  df['review'] = df['review'].apply(html.unescape)
  drugName = drugName.lower()
  subset = df[df['drugName'] == drugName]
  positive_subset = pd.DataFrame()
  negative_subset = pd.DataFrame()
  if not subset.empty:
    subset = subset.sort_values(by= ['rating', 'usefulCount'], ascending=[False, False])
    positive_subset = subset[subset.rating > 7].sort_values(by = ['usefulCount'], ascending=False)
    negative_subset = subset[subset.rating < 5].sort_values(by = ['usefulCount'], ascending=False)

  return subset, positive_subset, negative_subset

def get_alternative_drug_names(df, drugName):

    df['drugName_lower'] = df['drugName'].apply(lambda x: x.lower())
    conditon_selected_df = df[df['drugName_lower'] == drugName.lower()]
    conditon_selected = ''
    alternatives_text = ''
    if not conditon_selected_df.empty:
        conditon_selected = conditon_selected_df.condition.mode().tolist()[0]
    if conditon_selected:
        df_condition = df[df['condition'] == conditon_selected]
        df_condition = df_condition[df_condition['drugName_lower']!=drugName]
        alternatives_text = ', '.join(df_condition.drugName.unique().tolist())

    return alternatives_text

def get_summary_of_reviews(reviews, drugName):
  #return 'this is example text for summary'
  # Set up OpenAI API key
  api_key_ = os.getenv('OPENAI_API_KEY')

  #b['review'] = b.review.str.lower().replace('it', 'zyclara')

  # Concatenate all reviews into a single string
  all_reviews_text = ".".join(reviews)

  # Create a prompt to summarize all reviews
  prompt = f"A drug named {drugName} is being discussed. What are its uses and opinions in the text: \n{all_reviews_text}. End with a proper sentence"

  # Create an OpenAI client instance (OpenAI Client interface)
  client = openai.Client(api_key = api_key_)

  # Call the chat completion method
  response = client.chat.completions.create(
      model="gpt-4",  # You can also use "gpt-3.5-turbo" for faster responses
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt}
      ],
      max_tokens=150,
      temperature=0.5
  )


  return response.choices[0].message.content


# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    print('inside search with ')
    drug_name = request.form.get('drug_name')
    print('inside search with name: ', drug_name)
    # Process the image and extract text using EasyOCR
    data_relative_path = 'drugReviews/drugsComTrain_raw.tsv'
    review_data_path = os.path.join(app.config['DATA_FOLDER'], data_relative_path)
    df = pd.read_csv(review_data_path, sep='\t')
    df['review'] = df['review'].apply(html.unescape)

    # Assuming df is a valid DataFrame loaded earlier
    review_subset, review_positive, review_negative = filter_drug(df, drug_name)

    pos_reviews = []
    neg_reviews = []

    average_rating = 0
    if not review_subset.empty:
        average_rating = review_subset.rating.mean().round(2)

    if not review_positive.empty:
        pos_reviews = review_positive.head(2).review.tolist()

    if not review_negative.empty:
        neg_reviews  = review_negative.head(2).review.tolist()
    
    all_reviews = pos_reviews + neg_reviews

    summary = ''
    pr_1 = ''
    pr_2 = ''
    pr_3 = ''

    nr_1 = ''
    nr_2 = ''
    nr_3 = ''

    if len(all_reviews) > 0 : 
        summary = get_summary_of_reviews(all_reviews, drug_name)

    if review_positive.shape[0] > 1 : 
        pr_1 = review_positive.review.iloc[0]
    if review_positive.shape[0] > 2 :
        pr_2 = review_positive.review.iloc[1]
    if review_positive.shape[0] > 3 :
        pr_3 = review_positive.review.iloc[2]

    # Populate negative reviews if available
    if review_negative.shape[0] > 0:
        nr_1 = review_negative.review.iloc[0]
    if review_negative.shape[0] > 1:
        nr_2 = review_negative.review.iloc[1]
    if review_negative.shape[0] > 2:
        nr_3 = review_negative.review.iloc[2]


    # Render the template with positive and negative reviews as available
    return render_template('index.html', 
                            extracted_text=drug_name + ' (Average Rating ' + str(average_rating) + ')', 
                            filename=f'',
                            positive_review="Yes" if not review_positive.empty else "No",
                            positive_review_1=pr_1, 
                            positive_review_2=pr_2, 
                            positive_review_3=pr_3,
                            negative_review="Yes" if not review_negative.empty else "No",
                            negative_review_1=nr_1,
                            negative_review_2=nr_2,
                            negative_review_3=nr_3,
                            alternative_drugs_text = get_alternative_drug_names(df, drug_name),
                            summary_text = summary)



@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400
    
    if file and allowed_file(file.filename):
        # Save file to the static folder (or a subfolder inside static)
        filename = secure_filename(file.filename)
        file_save_path = os.path.join('static/uploads', filename)
        
        # Make sure the subdirectory exists
        if not os.path.exists(os.path.dirname(file_save_path)):
            os.makedirs(os.path.dirname(file_save_path))
        
        file.save(file_save_path)
    
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # file.save(file_path)

        # Process the image and extract text using EasyOCR
        data_relative_path = 'drugReviews/drugsComTrain_raw.tsv'
        review_data_path = os.path.join(app.config['DATA_FOLDER'], data_relative_path)
        df = pd.read_csv(review_data_path, sep='\t')
        df['review'] = df['review'].apply(html.unescape)

        possible_names = get_name_from_picture(file_save_path, df)

        print('Found Name: ', possible_names)
        # img = cv2.imread(file_path)
        # result = reader.readtext(img)
        
        # Extract text from OCR result
        extracted_text = 'Could Not Read Drug Name'
        if(possible_names):
            extracted_text = possible_names[0]
        

        # Assuming df is a valid DataFrame loaded earlier
        review_subset, review_positive, review_negative = filter_drug(df, extracted_text)

        average_rating = 0
        if not review_subset.empty:
            average_rating = review_subset.rating.mean().round(2)

        print('Average Rating is: ', average_rating)
        print('subset shape: ', review_subset.shape)
        print('positive subset shape: ', review_positive.shape)
        print('negative subset shape: ', review_negative.shape)

        pos_reviews = review_positive.head(2).review.tolist()
        neg_reviews  = review_negative.head(2).review.tolist()

        all_reviews = pos_reviews + neg_reviews

        summary = ''
        if len(all_reviews) > 0 : 
            summary = get_summary_of_reviews(all_reviews, extracted_text)

        pr_1 = ''
        pr_2 = ''
        pr_3 = ''

        if review_positive.shape[0] > 1 : 
            pr_1 = review_positive.review.iloc[0]
        if review_positive.shape[0] > 2 :
            pr_2 = review_positive.review.iloc[1]
        if review_positive.shape[0] > 3 :
            pr_3 = review_positive.review.iloc[2]


        nr_1 = ''
        nr_2 = ''
        nr_3 = ''

        # Populate negative reviews if available
        if review_negative.shape[0] > 0:
            nr_1 = review_negative.review.iloc[0]
        if review_negative.shape[0] > 1:
            nr_2 = review_negative.review.iloc[1]
        if review_negative.shape[0] > 2:
            nr_3 = review_negative.review.iloc[2]

        # Render the template with positive and negative reviews as available
        return render_template('index.html', 
                               extracted_text=extracted_text + ' (Average Rating ' + str(average_rating) + ')', 
                               filename=f'uploads/{filename}',
                               positive_review="Yes" if not review_positive.empty else "No",
                               positive_review_1=pr_1, 
                               positive_review_2=pr_2, 
                               positive_review_3=pr_3,
                               negative_review="Yes" if not review_negative.empty else "No",
                               negative_review_1=nr_1,
                               negative_review_2=nr_2,
                               negative_review_3=nr_3,
                               alternative_drugs_text = get_alternative_drug_names(df, extracted_text),
                               summary_text = summary)
    return 'File not allowed', 400
if __name__ == '__main__':
    app.run(debug=True)


