import logging
import azure.functions as func
import os
from openai import AzureOpenAI
import json
from datetime import datetime

os.environ['OPENAI_API_KEY'] = "1c18b8371a78404d9183cffeeb87042d"
os.environ['OPENAI_ENDPOINT'] = "https://mihcm-openai-australiaeast.openai.azure.com/"

client = AzureOpenAI(
    azure_endpoint=os.environ.get('OPENAI_ENDPOINT'),
    api_key=os.environ.get('OPENAI_API_KEY'),
    api_version="2024-02-15-preview"
)

# Load prompt files once at the beginning
script_directory = os.path.dirname(os.path.abspath(__file__))
degree_compare_file = os.path.join(script_directory, 'degreecomparePrompt.txt')

with open(degree_compare_file, 'r') as text_file:
    degree_compare_content = text_file.read()

degree_comparePrompt_main = f""" {degree_compare_content} """

duration_work_file = os.path.join(script_directory, 'durationworkPrompt.txt')

with open(duration_work_file, 'r') as text_file:
    duration_work_content = text_file.read()

duration_workPrompt_main = f""" {duration_work_content} """


relation_work_file = os.path.join(script_directory, 'relationworkPrompt.txt')

with open(relation_work_file, 'r') as text_file:
    relation_work_content = text_file.read()

relation_workPrompt_main = f""" {relation_work_content} """

# Separate JD rules
def separate_rules(JD_dict):
    ol_rules = [rule for rule in JD_dict['Rules'] if "Ordinary Level" in rule['Rule']]
    al_rules = [rule for rule in JD_dict['Rules'] if "Advanced Level" in rule['Rule']]
    degree_rules = [rule for rule in JD_dict['Rules'] if "University or Higher Educational Institute" in rule['Rule']]
    work_rules = [rule for rule in JD_dict['Rules'] if "Work Experience" in rule['RuleType']]
    return ol_rules, al_rules, degree_rules, work_rules

def Create_JSON_structures(ol_rules, al_rules, degree_rules, work_rules):
    ol_json_strings = [json.dumps(item, indent=4) for item in ol_rules]
    ol_json = ",\n".join(ol_json_strings)
    al_json = json.dumps(al_rules[0], indent=4)
    degree_json = {"Rules": degree_rules}
    work_json = {"Rules": work_rules}
    return al_json, work_json, degree_json

# Grade comparison mapping
grade_mapping = {
    'A': 75,
    'B': 65,
    'C': 55,
    'S': 45,
    'W': 30
}

# Subject name variations mapping
subject_variations = {
    "Mathematics": ["Mathematics", "Math", "Maths"],
    "Science": ["Science"],
    "English": ["English"]
}

def extract_ol_results(CV_dict):
    ol_description = ""
    for qualification in CV_dict["Qualifications"]:
        if qualification['QualificationCode'] in ['G.C.E. Ordinary Level Examination', 'G.C.E. O/Level', 'G.C.E. O/Level Exam']:
            ol_description = qualification["Description"]
            OL_text = f'Qualification: {qualification["QualificationCode"]}. Description: {ol_description}'
            break

    if not ol_description:
        return {}, ""

    try:
        # Parse the description to get subject-grade pairs
        ol_results = {item.split()[0].strip(): item.split()[1].strip() for item in ol_description.split(', ')}

        # Normalize subjects in results based on variations mapping
        normalized_results = {}
        for standard_subject, variations in subject_variations.items():
            for variation in variations:
                if variation in ol_results:
                    normalized_results[standard_subject] = ol_results[variation]
                    break
                
        return normalized_results, OL_text
    except Exception as e:
        # Handle parsing errors
        OL_text = "Invalid result format"
        return {}, OL_text

# Check if the O-Level rules are satisfied
def check_ol_rule(subject, required_grade, logic, normalized_results):
    if subject not in normalized_results:
        return False
    actual_grade = normalized_results[subject]
    return grade_mapping[actual_grade] >= grade_mapping[required_grade]


# Process O-Level rules
def process_ol_rules(ol_rules, normalized_results, OL_text):
    output = {"Rules": []}
    rule_satisfaction = {}
    for rule in ol_rules:
        subject, required_grade = rule["Entry"].split(' - ')
        rule_satisfaction[rule["Entry"]] = check_ol_rule(subject, required_grade, rule["Logic"], normalized_results)
        
    for rule in ol_rules:
        subject = rule["Entry"]
        rule_status = "Fulfilled" if rule_satisfaction.get(subject, False) else "Not Fulfilled"
        output_rule = {
            "RuleType": "Qualification",
            "Rule": rule["Rule"],
            "Entry": subject,
            "Logic": rule["Logic"],
            "CorrespondingText": OL_text,
            "RuleStatus": rule_status
        }
        output["Rules"].append(output_rule)
       
    return output

# Extract and compare A-Level results
def process_al_rules(CV_dict, JD_dict, output):
    AL_rule = {
        "RuleType": "Qualification",
        "Rule": "G.C.E. Advanced Level",
        "Entry": "",
        "Logic": "equal or better",
        "CorrespondingText": "",
        "RuleStatus": "Fulfilled"
    }

    result_mark = []
    for qualification in CV_dict['Qualifications']:
        if qualification['QualificationCode'] in ['G.C.E. Advanced Level Examination', 'G.C.E. A/Level', 'G.C.E. A/Level Exam']:
            results = qualification['Description']
    
            AL_rule['CorrespondingText'] = f'Qualification: {qualification["QualificationCode"]}. Description: {results}'
            results_list = results.split(',')
            results_list = [x.strip()[-1] for x in results_list]
    
            for grade in results_list:
                if grade == 'A':
                    result_mark.append(75)
                elif grade == 'B':
                    result_mark.append(65)
                elif grade == 'C':
                    result_mark.append(55)
                elif grade == 'S':
                    result_mark.append(45)
                elif grade in ['F', 'W']:
                    result_mark.append(30)

    cut_off_value = 0
    for i in JD_dict['Rules']:

      if i['Rule'] == 'G.C.E. Advanced Level':
        AL_rule['Entry'] = i['Entry']
        cut_off_grade = i['Entry'][0]
    
        if cut_off_grade == 'A':
          cut_off_value = 75
    
        if cut_off_grade == 'B':
          cut_off_value = 65
    
        if cut_off_grade == 'C':
          cut_off_value = 55
    
        if cut_off_grade == 'S':
          cut_off_value = 45
    
        if cut_off_grade == 'F' or cut_off_grade == 'W':
          cut_off_value = 30


    for i in result_mark:
        if i < cut_off_value:
            AL_rule['RuleStatus'] = "Not Fulfilled"
            break

    output["Rules"].append(AL_rule)
    return output

def degree_relation(rule_relation, cv_relation):
  
  prompt = degree_comparePrompt_main.format(rule_relation=rule_relation,cv_relation=cv_relation)
  
  # Make a request to the OpenAI GPT-3 model
  response = client.chat.completions.create(
      model = "gpt-35-turbo",
      messages = [
          {"role":"system", "content":prompt},
          {"role":"user", "content":rule_relation},
          {"role":"user", "content":cv_relation}
          ],
      max_tokens = 600,
      n = 1,
      temperature = 0.2,
      top_p = 0.1
  )

  #Extract the generated content from the response
  generated_content = response.choices[0].message.content

  return generated_content


# Process Degree rules
def process_degree_rules(degree_json, CV_dict, output):
    # Extract the first rule's Entry and Logic
   
    deg_entry = degree_json['Rules'][0]['Entry']

    # Combine them into a natural language text
    rule_deg_relation = f"The applicant has a {deg_entry} or in a similar field."

    # Initialize a list to store sentences and qualification codes
    cv_deg_relation_sentences = []
    deg_sim_scores = []
    qualification_codes = []

    # Threshold for similarity
    deg_threshold = 50.0

   
    # Iterate over each qualification in the qualifications list
    for qualification in CV_dict["Qualifications"]:
        qualification_code = qualification["QualificationCode"]
    
        # Check if qualification is not "G.C.E. Advanced Level Examination" or "G.C.E. Ordinary Level Examination"
        if "Level" not in qualification_code:
            qualification_codes.append(qualification_code)


    # Iterate through each work experience
    for i, qualification in enumerate(qualification_codes, start=1):
        qualification_code = qualification
    
        cv_deg_relation = f"The applicant has a {qualification_code}."
    
        # Append the sentence to the list for optional future use
        cv_deg_relation_sentences.append(cv_deg_relation)
    
        deg_score = degree_relation(rule_deg_relation, cv_deg_relation)
    
        # Convert str "%" to float
        numeric_str = deg_score.strip('%')
        deg_percent_float = float(numeric_str)
    
        deg_sim_scores.append(deg_percent_float)
    
    
        if deg_percent_float >= deg_threshold:
          Rule_Status_1 = "Fulfilled"
    
        else:
          Rule_Status_1 = "Not Fulfilled"


    # Save the final output to a variable
    deg_Rule_Status = Rule_Status_1

    # Use list comprehension to create formatted strings
    formatted_strings = [
        f"Degree: {deg}"
        for deg in qualification_codes
    ]

    # Join the list items into a single string separated by a comma and space
    deg_corresponding_text = ', '.join(formatted_strings)

    rule_type = "Qualification"
    rule = degree_json['Rules'][0]['Rule']
    entry = degree_json['Rules'][0]['Entry']
    logic = degree_json['Rules'][0]['Logic']
    corresponding_text = deg_corresponding_text
    rule_status = deg_Rule_Status

    # Create the dictionary for the JSON output
    deg_dict = {
        "RuleType": rule_type,
        "Rule": rule,
        "Entry": entry,
        "Logic": logic,
        "CorrespondingText": corresponding_text,
        "RuleStatus": rule_status
    }

    output["Rules"].append(deg_dict)
    return output


# Work experience comparison function using Azure's OpenAI service
def compare_work(rule, cv, prompt_type):
    if prompt_type == "relation":
        
        prompt = relation_workPrompt_main.format(rule=rule,cv=cv)
        
    elif prompt_type == "duration":
        
        prompt = duration_workPrompt_main.format(rule=rule,cv=cv)
        
    else:
        raise ValueError("Invalid prompt type. Choose either 'relation' or 'duration'.")

    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": rule},
            {"role": "user", "content": cv}
        ],
        max_tokens=600,
        n=1,
        temperature=0.2,
        top_p=0.1
    )

    generated_content = response.choices[0].message.content.strip()

    return generated_content

def calculate_duration(date1, date2):
    # Convert the date strings to datetime objects
    dt1 = datetime.strptime(date1, "%d/%m/%Y")
    dt2 = datetime.strptime(date2, "%d/%m/%Y")

    # Calculate the difference between the dates
    delta = dt2 - dt1

    # Calculate number of years and months in the difference
    years = delta.days // 365
    months = (delta.days % 365) // 30

    return years, months
    
def aggregate_durations(durations):
    total_years = 0
    total_months = 0

    for years, months in durations:
        total_years += years
        total_months += months

    # Convert months to years and months
    total_years += total_months // 12
    total_months = total_months % 12

    return total_years, total_months

# Process work experience rules
def process_work_experience_rules(work_json, CV_dict, output):

    entry = work_json['Rules'][0]['Entry']
    # Combine them into a natural language text
    rule_work_relation = f"The applicant has worked in the field of {entry} or similar."
    # Initialize a list to store sentences
    cv_relation_sentences = []
    sim_scores = []
    durations = []

    # Threshold for similarity
    threshold = 50.0


    for i, experience in enumerate(CV_dict["WorkExperiences"], start=1):
        company_name = experience["CompanyName"]
        position_held = experience["PositionHeld"]
        joined_as = experience["JoinedAs"]

        if position_held == "":
            cv_work_relation = f"""The applicant has worked as a {joined_as} at {company_name}."""
        else:
            cv_work_relation = f"""The applicant has worked as a {position_held} at {company_name}."""
       # Append the sentence to the list for optional future use
        cv_relation_sentences.append(cv_work_relation)

        if position_held == "" and joined_as =="":
            score = "0%"
        else:
            score = compare_work(rule_work_relation, cv_work_relation, "relation")
    
        # Convert str "%" to float
        numeric_str = score.strip('%')
        percent_float = float(numeric_str)
    
        sim_scores.append(percent_float)
    
    
        if percent_float >= threshold:
    
              Rule_Status_1 = "The rule is fulfilled"
        
              joined_date = experience["JoinedDate"]
        
              if experience["Resigned"] == False:
                 today = datetime.today()
                 # Format the date as day/month/year
                 resigned_date = today.strftime("%d/%m/%Y")
        
              else:
                 resigned_date = experience["ResignedDate"]
        
              duration = calculate_duration(joined_date, resigned_date)
              durations.append(duration)
        
              total_years, total_months = aggregate_durations(durations)
        
              work_logic = work_json['Rules'][0]['Logic']
              rule_duration = f"The applicant has worked for {work_logic}."
        
              cv_duration = f"The applicant has worked for {total_years} years and {total_months} months."
        
              process_duration = compare_work(rule_duration, cv_duration, "duration")
        
              if process_duration == "The rule is fulfilled.":
                Rule_Status_2 = "The rule is fulfilled"
        
              else:
                Rule_Status_2 = "The rule is not fulfilled"
    
        else:
          Rule_Status_1 = "The rule is not fulfilled"

    # Determine the final output based on the overall rule status
    if Rule_Status_1 == "The rule is not fulfilled":
      final_output = "Not Fulfilled"
    elif Rule_Status_1 == "The rule is fulfilled" and Rule_Status_2 == "The rule is not fulfilled":
      final_output = "Not Fulfilled"
    else:
      final_output = "Fulfilled"
    
    # Save the final output to a variable
    Rule_Status = final_output

    # Use list comprehension to create formatted strings
    formatted_strings = [
        f"Company Name: {exp['CompanyName']}, Joined As: {exp['JoinedAs']}, Position Held: {exp['PositionHeld']}, JoinedDate: {exp['JoinedDate']}, ResignedDate: {exp['ResignedDate']}"
        for exp in CV_dict["WorkExperiences"]
    ]

    # Join the list items into a single string separated by a comma and space
    corresponding_text = ', '.join(formatted_strings)
    
    rule_type = work_json['Rules'][0]['RuleType']
    rule = work_json['Rules'][0]['Rule']
    entry = work_json['Rules'][0]['Entry']
    logic = work_json['Rules'][0]['Logic']
    corresponding_text = corresponding_text
    rule_status = Rule_Status


    # Create the dictionary for the JSON output
    work_dict = {
        "RuleType": rule_type,
        "Rule": rule,
        "Entry": entry,
        "Logic": logic,
        "CorrespondingText": corresponding_text,
        "RuleStatus": rule_status
    }

    output["Rules"].append(work_dict)
    
    return output

# Calculate overall score
def calculate_score(output):
    fulfilled_count = sum(1 for rule in output['Rules'] if rule['RuleStatus'] == 'Fulfilled')
    total_rules = len(output['Rules'])
    overall_score = (fulfilled_count / total_rules) * 100
    output["OverallScore"] = overall_score
    return output


def validate_and_clean_compare_result(input_json):
    expected_structure = {
        "RuleType": "",
        "Rule": "",
        "Entry": "",
        "Logic": "",
        "CorrespondingText": "",
        "RuleStatus": ""
        "RuleStatus"
    }

    cleaned_compare_result = []

    for result in input_json.get("Rules", []):
        cleaned_result = {}
        for key in expected_structure:
            cleaned_result[key] = result.get(key, "")
        cleaned_compare_result.append(cleaned_result)

    cleaned_json = {
        "Rules": cleaned_compare_result,
        "OverallScore": input_json.get("OverallScore", "")
    }

    return cleaned_json




def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        data = req.get_json()
        JD_dict = data.get('Rule_json')
        CV_dict = data.get('CV_json')
        

        if not JD_dict or not  CV_dict:
            raise ValueError("Both 'Rule_json' and 'CV_json' must be provided")

        ol_rules, al_rules, degree_rules, work_rules = separate_rules(JD_dict)
        al_json, work_json, degree_json    = Create_JSON_structures(ol_rules, al_rules, degree_rules, work_rules)
        normalized_results, OL_text = extract_ol_results(CV_dict)
        output = process_ol_rules(ol_rules, normalized_results, OL_text)
        output = process_al_rules(CV_dict, JD_dict, output)
        output = process_degree_rules(degree_json, CV_dict, output)
        output = process_work_experience_rules(work_json, CV_dict, output)
        output = calculate_score(output)

        response_data = validate_and_clean_compare_result(output)


        return func.HttpResponse(json.dumps(response_data), mimetype="application/json", status_code=200)
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return func.HttpResponse(f"Error processing request: {e}", status_code=500)
