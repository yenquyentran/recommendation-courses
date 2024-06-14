import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# Using menu
st.title("Trung Tâm Tin Học")
# menu
menu = ["Home", "Capstone Project", "Sử dụng các điều khiển", "Gợi ý điều khiển project 1", "Gợi ý điều khiển project 2", "Gợi ý điều khiển project 3"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':    
    st.subheader("[Trang chủ](https://csc.edu.vn)")  
elif choice == 'Capstone Project':    
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")
    st.write("""### Có 3 chủ đề trong khóa học:
    - Topic 1: Sentiment Analysis
    - Topic 2: Recommendation System
    - Topic 3: RFM & Clustering
    - ...""")