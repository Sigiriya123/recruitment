import logging
import base64
import azure.functions as func
import fitz
import numpy as np
import pdfplumber
import os
from openai import AzureOpenAI
from io import BytesIO
import json
from sentence_transformers import SentenceTransformer, util

os.environ['OPENAI_API_KEY'] = "1c18b8371a78404d9183cffeeb87042d"
os.environ['OPENAI_ENDPOINT'] = "https://mihcm-openai-australiaeast.openai.azure.com/"

client = AzureOpenAI(
    azure_endpoint = os.environ.get('OPENAI_ENDPOINT'),
    api_key = os.environ.get('OPENAI_API_KEY'),
    api_version = "2024-02-15-preview"
)

   
# Load prompt files once at the beginning
script_directory = os.path.dirname(os.path.abspath(__file__))
extract_details_file = os.path.join(script_directory, 'extractdetailsPrompt.txt')

with open(extract_details_file, 'r') as text_file:
    extract_details_content = text_file.read()

extract_detailsPrompt_main = f""" {extract_details_content} """


def estimate_columns(pdf_bytes, page_number=0):
    try:
        # Open the PDF from bytes
        pdf_document = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
        # Select the specific page
        page = pdf_document.load_page(page_number)

        # Get text blocks from the page
        text_blocks = page.get_text("blocks")

        # Get the x-coordinates of the text blocks
        x_coords = [block[:2] for block in text_blocks]  # (x0, x1)

        # Determine the column positions by clustering x-coordinates
        x_coords = np.array(x_coords)
        x_start_coords = x_coords[:, 0]

        # Determine unique column positions with some tolerance
        tolerance = 20
        columns = []

        for x in sorted(x_start_coords):
            if not any(abs(x - col) < tolerance for col in columns):
                columns.append(x)

        return len(columns)

    except Exception as e:
        print(f"Error: {e}")
        return 0
    

def extract_text_from_two_columns_pdfplumber(pdf_bytes):
    text = []

    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            width = page.width
            height = page.height

            # Define the bounding boxes for the left and right columns
            left_bbox = (0, 0, width / 2, height)
            right_bbox = (width / 2, 0, width, height)

            # Extract text from each column
            left_text = page.within_bbox(left_bbox).extract_text()
            right_text = page.within_bbox(right_bbox).extract_text()

            # Combine the text from both columns
            combined_text = (left_text or "") + "\n" + (right_text or "")
            text.append(combined_text)

    return "\n".join(text)


def convert_pdf_to_text(pdf_bytes):
    try:
        text = ""
        # Open the bytes as a PDF file
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''  # Ensure we handle None values

        return text

    except Exception as e:
        print(f"Error: {e}")
        return ""
    


def extract_details(user_input):
  
  prompt = extract_detailsPrompt_main.format(user_input=user_input)
  # Make a request to the OpenAI GPT-3 model
  response = client.chat.completions.create(
      model = "gpt-35-turbo",
      messages = [
          {"role":"system", "content":prompt},
          {"role":"user", "content":user_input}],
      max_tokens = 2000,
      n = 1,
      temperature = 0.2,
      top_p = 0.1
  )

  #Extract the generated content from the response
  generated_content = response.choices[0].message.content

  return generated_content

############## Qualifications JSON Format

# Function to encode text into embeddings
def encode_text(text, model):
    return model.encode(text, convert_to_tensor=True).unsqueeze(0)


# Function to compute cosine similarity between two embeddings
def compute_cosine_similarity(embedding1, embedding2):
    return util.pytorch_cos_sim(embedding1, embedding2).item()


# Function to update CV qualifications with the most similar degree
def update_cv_qualifications(CV_json_dict, qual_list_json_str, model_name='all-MiniLM-L6-v2', similarity_threshold=0.65):
    
    qual_list_json = json.loads(qual_list_json_str)
    
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)

    # Iterate through CV qualifications and find the most similar degree
    for qualification in CV_json_dict['Qualifications']:
        qualification_name = qualification['QualificationName']
        max_similarity = -1
        best_match = qualification_name

        # Encode qualification code into embedding
        qualification_embedding = encode_text(qualification_name, model)

        # Calculate similarity with each degree in the degree list
        for qual in qual_list_json['QualificationList']:
            qual_name = qual['QualificationName']
            qual_type = qual['QualificationType']
            qual_embedding = encode_text(qual_name, model)
            similarity = compute_cosine_similarity(qualification_embedding, qual_embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = qual_name
                best_type = qual_type

        # Only replace if max similarity exceeds 0.65
        if max_similarity > 0.65:
            qualification['BestMatchingQualificationName'] = best_match
            qualification['QualificationType'] = best_type 
        else:
            qualification['BestMatchingQualificationName'] = ""
            qualification['QualificationType'] = best_type

    return CV_json_dict





def compare_and_update_json(input_json):
    required_json_structure = {
    "TitleCode": "",
    "Initials": "",
    "FirstName": "",
    "Surname": "",
    "FullName": "",
    "DateOfBirth": "",
    "GenderCode": "",
    "NicNumber": "",
    "PAddress1": "",
    "PAddress2": "",
    "PAddress3": "",
    "DistrictCode": "",
    "PCountry": "",
    "PPostalCode": "",
    "Email": "",
    "RTelephone": "",
    "MobileNumber": "",
    "ReligionCode": "",
    "RaceCode": "",
    "NationalityCode": "",
    "MaritalStatusCode": "",
    "FatherName": "",
    "FatherOccupation": "",
    "MotherName": "",
    "MotherOccupation": "",
    "DateAvaliableForWork": "",
    "DesiredSalary": "",
    "ReferredBy": "",
    "CategoryCode": "",
    "PictureName": "",
    "PictureContentBase64String": "",
    "Spouses": [
        {
            "SpouseCode": "",
            "SpouseName": "",
            "DateOfBirth": "",
            "Occupation": "",
            "Employer": "",
            "Alive": ""
        }
    ],
    "Childrens": [
        {
            "ChildCode": "",
            "ChildName": "",
            "DateOfBirth": "",
            "ChildType": "",
            "School": "",
            "PassportNumber": "",
            "Alive": ""
        }
    ],
    "Qualifications": [
        {
            "QualificationType": "",
            "QualificationName": "",
            "BestMatchingQualificationName":"",
            "InstituteCode": "",
            "FromYear": "",
            "FromMonth": "",
            "ToYear": "",
            "ToMonth": "",
            "Description" : ""
        }
    ],
    "WorkExperiences": [
        {
            "CompanyName": "",
            "Address": "",
            "JoinedDate": "",
            "JoinedAs": "",
            "PositionHeld": "",
            "Resigned": "",
            "ResignedDate": "",
            "LastDrawnSalary": "",
            "CurrencyCode": "",
            "PreviousEpfNumber": "",
            "ExperienceType": "",
            "ReasonToLeave": ""
        }
    ],
    "Achievements": [
        {
            "AwardName": "",
            "Year": "",
            "Comments": ""
        }
    ],
    "Referee": [
        {
            "RefereeName": "",
            "RefereeCompanyName": "",
            "RefereeDesignation": "",
            "PhoneNumber": "",
            "Email": "",
            "RefereeAddress": "",
            "HowRefereeKnows": ""
        }
    ]
    }
    
    def update_dict(required_dict, input_dict):
        
        updated_dict = {}
        
        for key, value in required_dict.items():
            if key in input_dict:

                # If the value is a list, handle list comparison
                if isinstance(value, list):
                    updated_dict[key] = update_list(value, input_dict[key])

                # If the value is a dictionary, handle dictionary comparison
                elif isinstance(value, dict):
                    updated_dict[key] = update_dict(value, input_dict[key])
                else:
                    updated_dict[key] = input_dict[key]
            else:
                updated_dict[key] = value

        return updated_dict

    def update_list(required_list, input_list):
        # Create a new list to store the updated input list
        updated_list = []

        for item in input_list:
            updated_item = update_dict(required_list[0], item)
            updated_list.append(updated_item)

        return updated_list

    # Start the update process with the top-level dictionary
    updated_json = update_dict(required_json_structure, input_json)
    
    return updated_json



def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    current_dir = os.path.dirname(__file__)

    try:
       
        data = req.get_json()
        content = data.get('cvb64')
        qual_list = data.get('quallist')
        decode = base64.b64decode(content)
        num_columns = estimate_columns(decode)

        if num_columns == 2:
            extracted_text = extract_text_from_two_columns_pdfplumber(decode)
        else:
            extracted_text = convert_pdf_to_text(decode)
        

        CV_json = extract_details(extracted_text)
        CV_json_dict = json.loads(CV_json)
       
        qual_list_json = json.dumps(qual_list)
        updated_cv_json = update_cv_qualifications(CV_json_dict,qual_list_json)
        updated_json = compare_and_update_json(updated_cv_json)
       
        # Create the response dictionary with the key "CVDetails"
        response_data = {
            "CVDetails": updated_json
        }

        # Return the response as JSON
        return func.HttpResponse(json.dumps(response_data), mimetype="application/json", status_code=200)

       

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return func.HttpResponse(f"Error processing request: {e}", status_code=500)