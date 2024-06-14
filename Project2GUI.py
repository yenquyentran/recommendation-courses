import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import warnings
from gensim import corpora, models, similarities

    
file_path_2 = 'courses.csv'
data_courses = pd.read_csv(file_path_2)

# Title of the app
st.title("COURSE RECOMMENDATION SYSTEM - GROUP 4")

# Define the page navigation menu
menu = ["Home", "Student Login", "Student Search"]
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Current page based on session state or sidebar selection
current_page = st.sidebar.selectbox('Menu', menu, index=menu.index(st.session_state.page))

# Check if the sidebar selection has changed and update session state
if current_page != st.session_state.page:
    st.session_state.page = current_page
    st.rerun()  # Change to st.rerun() if the app is updated post April 2024

def custom_css():
    css = """
    <style>
        .stSelectbox, .css-1l02zno {
            width: 50%;
            margin: 10px auto;
        }
        .css-2trqyj {
            font-size: 16px;
        }
        .st-dm {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        .css-8mokm4 {
            border-color: #f69b4;
            color: #f69b4;
        }
        
        div.stButton > button {
            font-size: 16px;
            color: white !important;
            background-color: #ff69b4 !important;
            height: 2em;
            width: 100%;
            border-radius: 5px;
            border: none;
            outline: none;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        div.stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(255,105,180,0.3);
        }
        div.stButton > button:active {
            background-color: #3D9E9D !important; /* Lighter shade when clicked */
            color: #fff !important; /* Ensuring text is white and readable */
            box-shadow: 0 2px 6px rgba(0,0,0,0.2); /* Soft shadow for depth */
        }
        .info-box {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 4px;
            margin: 10px;
            box-shadow: 5px 5px 5px grey;
            height: 200px;  /* Fixed height for uniformity */
        }
        .info-box-3 {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 4px;
            margin: 10px;
            box-shadow: 5px 5px 5px grey;
            height: 140px;  /* Fixed height for uniformity */
        }
        .info-box-2 {
            border: 2px solid #4CAF50;
            border-radius: 10px;
            padding: 4px;
            margin: 10px;
            box-shadow: 5px 5px 5px grey;
            height: 280px;  /* Fixed height for uniformity */
        }
        .info-box-2 ul {
            padding-left: 5px !important;
        }
        .info-box-2 li {
            font-size: 11px !important;
            margin-bottom: 2px !important;
        }
        .course-name {
            font-size: 13px; /* Smaller font size for course names */
            height: 60px; /* Fixed height to accommodate longer names */
            overflow: hidden; /* Hides text that doesn't fit in the defined height */
            font-weight: bold; /* Make course name bold */
        }
        .compact-info, .review-info {
            font-size: 11px; /* Smaller font size for unit, rating, etc. */
            margin-bottom: 2px; /* Tighter spacing */
        }
        .dashed-divider {
            border-top: 1px dashed #ccc; /* Dashed line style */
            margin: 8px 0; /* Spacing around the divider */
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Initialize custom CSS
custom_css()

    
button_style = """
<style>
div.stButton > button:first-child {
    font-size: 20px;
    color: white;
    background-color: #ff69b4;
    height: 3em;
    width: 100%;
    border-radius: 5px;
    border: none;
    outline: none;
}
</style>
"""
st.markdown(button_style, unsafe_allow_html=True)
def truncate_text(text, max_length=280):
    """Truncate text to a maximum length, adding ellipsis if necessary."""
    if len(text) > max_length:
        return text[:max_length].rstrip() + '...'
    return text

def summarize_results(results):
    """Extract the first two sentences from the results text and truncate each if they are longer than 40 characters."""
    if pd.isna(results):
        return []
    sentences = re.split('\n', results)  # Split by period followed by space
    truncated_sentences = [truncate_text(sentence) for sentence in sentences[:2]]  # Apply truncation only to the first two sentences
    return truncated_sentences

# Display content based on the current page
if st.session_state.page == 'Home':
    st.write("Welcome to our App. Here you can find all recommendations to develop your skills in Data Sciensce. ")
    st.write("Please let us know how you want to proceed")
    # Home page with navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("With account login"):
            st.session_state.page = 'Student Login'
            st.rerun()  # Use st.rerun() as needed
    with col2:
        if st.button("Without login"):
            st.session_state.page = 'Student Search'
            st.rerun()  # Use st.rerun() as needed
elif st.session_state.page == 'Student Login':
    st.header("Student Login Page")
    data = {
            '59' : 'David L',
            '31' : 'Kamlesh C',
            '59' : 'David L',
            '19' : 'Daniel S',
            '29' : 'Daniel P',
            '42' : 'James H',
            '97' : 'Rahul G',
            '7397' : 'mayank s'
    }
    # Tạo DataFrame từ dictionary
    df_KH = pd.DataFrame(list(data.items()), columns=['ID', 'Name'])
    # In 3 khách hàng này ra màn hình
    st.write("##### Courses Recommendation for users")
    # Tạo một điều khiển và đưa khách hàng ngẫu nhiên này vào đó

    selected = st.selectbox("Select your name:", df_KH['Name'])
    st.write("Your account is:", selected)


    file_path_1 = 'last_6_courses_final.csv'
    data_last_courses = pd.read_csv(file_path_1)
    
    file_path_3 = 'recommendation_courses_final.csv'
    data_recommend = pd.read_csv(file_path_3)

# Merge the dataframes on 'CourseName' to include all details
    data_last_courses = data_last_courses.drop_duplicates()
    filtered_data = data_last_courses[data_last_courses['ReviewerName'] == selected].sort_values('DateOfReview', ascending=False).head(6)
    
    filtered_data_recommend = data_recommend[data_recommend['ReviewerName'] == selected]
    
    st.write( f"### Most recent courses for {selected}:")
    cols = st.columns(3)
    for index, row in enumerate(filtered_data.iterrows()):
        _, row = row
        with cols[index % 3]:
            st.markdown(f"""
                <div class='info-box'>
                    <div class='course-name'>{row['CourseName']}</div>
                    <div class='compact-info'>{row['Unit']} | {row['AvgStar']} ⭐</div>
                    <div class='review-info'>Number of reviews: {int(row['ReviewNumber'])}</div>
                    <div class='compact-info'>Level: {row['Level']}</div>
                    <div class='dashed-divider'></div>  <!-- Dashed line -->
                    <div class='compact-info'>Reviewed on: {row['DateOfReview']}</div>
                    <div class='compact-info'>Your rating: {row['RatingStar']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            
    st.write("### Recommended courses for you")
    cols = st.columns(3)
    for index, row in enumerate(filtered_data_recommend.iterrows()):
        _, row = row
        with cols[index % 3]:
            results_sentences = summarize_results(row['Results'])
            results_formatted = "".join(f"<li>{sentence}</li>" for sentence in results_sentences)
            st.markdown(f"""
                <div class='info-box-2'>
                    <div class='course-name'>{row['CourseName']}</div>
                    <div class='compact-info'>{row['Unit']} | {row['AvgStar']} ⭐</div>
                    <div class='compact-info'>Level: {row['Level']}</div>
                    <div class='review-info'>Number of reviews: {int(row['ReviewNumber'])}</div>
                    <ul>{results_formatted}</ul>
                </div>
            """, unsafe_allow_html=True)
    
elif st.session_state.page == 'Student Search':
    st.header("Proceed without login")
    st.write("We provide more than 800 courses for students all over the worlds. How you want to find the best courses for you:")

    ## xử lý dữ liệu
    # Remove the 'CourseID' column as it duplicates 'CourseName'
    data_courses.drop(columns='CourseID', inplace=True)
    # Create a new unique identifier for each course

    # Impute missing values
    data_courses['Level'].fillna('Unknown', inplace=True)
    data_courses['Results'].fillna('No description available', inplace=True)
    # Function to clean text data: remove special characters and extra spaces
    def clean_text(text):
        import re
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Apply text cleaning to the 'Results' column
    data_courses['Results'] = data_courses['Results'].apply(clean_text)
    #check missing data
    data_courses.isnull().sum()
    # Display rows where 'CourseName' is duplicated correctly
    duplicate_course_entries = data_courses[data_courses['CourseName'].duplicated(keep=False)]
    duplicate_course_entries.sort_values(by='CourseName')
    # Append the provider's name to the CourseName for duplicates to make them unique
    data_courses.loc[data_courses['CourseName'].duplicated(keep=False), 'CourseName'] = \
        data_courses['CourseName'] + ' (' + data_courses['Unit'] + ')'
    # Update CourseID since CourseName has changed
    data_courses['CourseID'] = data_courses['CourseName'].factorize()[0]
    # content-based
    # Step 1: Combine CourseName, Unit, and Results into a single text field
    data_courses['Combined_Text'] = data_courses['CourseName'] + " " + data_courses['Unit'] + " " + data_courses['Results']
    # Tokenize(split) the sentences into words
    products_gem = [[text for text in x.split()] for x in data_courses['Combined_Text']]
    # remove some special elements in texts
    products_gem_re = [[re.sub('[0-9]+','', e) for e in text] for text in products_gem] # số
    products_gem_re = [[t.lower() for t in text if not t in ['', ' ', ',', '.', '...', '-',':', ';', '?', '%', '(', ')', '+', '/', 'g', 'ml']] for text in  products_gem_re]
    # Tạo bộ stop words hoặc có thể dùng thư viện bất kỳ
    stop_words = [
    # Articles
    "a", "an", "the",
    # Prepositions
    "in", "on", "at", "to", "from", "by",
    # Conjunctions
    "and", "but", "or", "for", "nor", "so", "yet",
    # Pronouns
    "I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    # Auxiliary verbs
    "be", "have", "do", "does", "did", "was", "were", "will", "would", "shall", "should", "may", "might", "can", "could", "must",
    # Other common stop words
    "that", "this", "which", "what", "their", "these", "those"
    ]

    # Xử lý từng phần tử trong danh sách
    def remove_stopwords(data_list):
        new_data_list = []
        for sentence in data_list:
            new_sentence = [word for word in sentence if word not in stop_words]
            new_data_list.append(new_sentence)
        return new_data_list

    # Áp dụng hàm remove_stopwords
    products_gem_re = remove_stopwords(products_gem_re)
    # Obtain the number of features based on dictionary: Use corpora.Dictionary
    dictionary = corpora.Dictionary(products_gem_re)
    # Numbers of features (word) in dictionary
    feature_cnt = len(dictionary.token2id)
    # Obtain corpus based on dictionary (dense matrix)
    corpus = [dictionary.doc2bow(text) for text in products_gem_re]
    # Use TF-IDF Model to process corpus, obtaining index
    tfidf = models.TfidfModel(corpus)
    # tính toán sự tương tự trong ma trận thưa thớt
    index_model = similarities.SparseMatrixSimilarity(tfidf[corpus],
                                                num_features = feature_cnt)
    ########################################################################################################################################
    col1, col2 = st.columns(2)
    # Adding a radio button in each column
    with col1:
        if st.button("Search by Keywords"):
            # Store the selection in the session state or handle the action immediately
            st.session_state.search_selection = "keywords"

    with col2:
        if st.button("View Top Courses"):
            # Store the selection in the session state or handle the action immediately
            st.session_state.search_selection = "top_courses"
    if 'search_selection' in st.session_state:  
        if st.session_state.search_selection == "top_courses":
            if 'selected_courses' not in st.session_state:
                st.session_state['selected_courses'] = []
            if 'liked_message' not in st.session_state:
                st.session_state['liked_message'] = ""
            if 'selected_course_id' not in st.session_state:
                st.session_state['selected_course_id'] = None
                
            course_id_for_selection = 1  
            top_courses = data_courses.sort_values(by='AvgStar', ascending=False).head(6)
            
            st.write("### Get a head start for your career from today")
            st.write("With these programs, you can build valuable skills, earn career credentials, and make progress toward a degree before you even enroll.")
            cols = st.columns(3)

            for index, row in enumerate(top_courses.iterrows()):
                _, row = row
                with cols[index % 3]:
                    st.markdown(f"""
                        <div class='info-box-3'>
                            <div class='course-name'>{row['CourseName']}</div>
                            <div class='compact-info'>{row['Unit']} | {row['AvgStar']} ⭐</div>
                            <div class='compact-info'>Level: {row['Level']}</div>
                            <div class='review-info'>Number of reviews: {int(row['ReviewNumber'])}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    if st.button("Like This Course", key=row['CourseID']):
                        if row.CourseName not in st.session_state.selected_courses:
                            st.session_state.selected_courses.append(row.CourseName)
                            st.session_state.selected_course_id = row['CourseID']
                            st.session_state.liked_message = f"You liked {row['CourseName']}"
            if st.session_state.liked_message:
                st.success(st.session_state.liked_message)
                if st.session_state.selected_course_id:
                    index_course_id = st.session_state.selected_course_id
                    product_selection = data_courses[data_courses['CourseID'] == index_course_id]
                    name_description_pre = product_selection.iloc[0]['CourseName'] + " " + product_selection.iloc[0]['Unit']+ " "+ product_selection.iloc[0]['Results']
                    view_product = name_description_pre.lower().split()
                    view_product = [char for char in view_product if re.match(r'[a-zA-Z\s]', char)]
                    view_product_new = [word for word in view_product if word not in stop_words]
                    kw_vector = dictionary.doc2bow(view_product_new)
                    sim = index_model[tfidf[kw_vector]]
                    data = []
                    for i in range(len(sim)):
                        data.append((i, sim[i]))
                    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)[:6]
                    id_list_1 = [item[0] for item in sorted_data]
                    id_list = [x for x in id_list_1 if x != index_course_id]
                    product_famillier = data_courses.iloc[id_list]
                    
                    st.write("### Recommended courses for you")
                    cols = st.columns(3)
                    for index, row in enumerate(product_famillier.iterrows()):
                        _, row = row
                        with cols[index % 3]:
                            results_sentences = summarize_results(row['Results'])
                            results_formatted = "".join(f"<li>{sentence}</li>" for sentence in results_sentences)
                            st.markdown(f"""
                                <div class='info-box-2'>
                                    <div class='course-name'>{row['CourseName']}</div>
                                    <div class='compact-info'>{row['Unit']} | {row['AvgStar']} ⭐</div>
                                    <div class='compact-info'>Level: {row['Level']}</div>
                                    <div class='review-info'>Number of reviews: {int(row['ReviewNumber'])}</div>
                                    <ul>{results_formatted}</ul>
                                </div>
                            """, unsafe_allow_html=True)
        
        ############################################################################################################################################################
        elif st.session_state.search_selection == "keywords":
            st.subheader("Search key:")
            content =  st.text_input("Your search:", placeholder="Enter your keywords:")
            #content = st.text_area("Your search:")
            if st.button('Search'):
                if content:
                    tukhoa = content.lower().split()
                    tukhoa = [char for char in tukhoa if re.match(r'[a-zA-Z\s]', char)]
                    tukhoa_new = [word for word in tukhoa if word not in stop_words]
                    kw_vector_2 = dictionary.doc2bow(tukhoa_new)
                    sim_2 = index_model[tfidf[kw_vector_2]]
                    data_2 = []
                    for i in range(len(sim_2)):
                        data_2.append((i, sim_2[i]))
                        
                    sorted_data = sorted(data_2, key=lambda x: x[1], reverse=True)
                    course_recommend = data_courses.iloc[[tup[0] for tup in sorted_data[:5]]]
                    st.write("### Recommended courses for you")
                    cols = st.columns(3)
                    for index, row in enumerate(course_recommend.iterrows()):
                        _, row = row
                        with cols[index % 3]:
                            results_sentences = summarize_results(row['Results'])
                            results_formatted = "".join(f"<li>{sentence}</li>" for sentence in results_sentences)
                            st.markdown(f"""
                                <div class='info-box-2'>
                                    <div class='course-name'>{row['CourseName']}</div>
                                    <div class='compact-info'>{row['Unit']} | {row['AvgStar']} ⭐</div>
                                    <div class='compact-info'>Level: {row['Level']}</div>
                                    <div class='review-info'>Number of reviews: {int(row['ReviewNumber'])}</div>
                                    <ul>{results_formatted}</ul>
                                </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("Please enter a keyword to search.")
