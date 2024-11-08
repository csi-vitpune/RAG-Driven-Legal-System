import streamlit as st
import sys
sys.path.append(r'Developers') 
from  Aditi.Law import Feature_1_RAG
from Anvesha.Law_Information_System import search_laws_with_generation
from Prasad.Action_Assessment_and_Law_Violation_Detection import RAG_feature_3
from Saket.Draft_Making_CSI import RAG_feature_4
from Shireen.Shireen_main import RAG_feature_5

options = [
    "Legal Document Chat and Understanding System",
    "Law Information System",
    "Action Assessment and Law Violation Detection",
    "Case Drafting and Relevant Law Suggestions",
    "Personal Legal Advice"
]

st.sidebar.title("Select a feature")
selected_option = None

for i, option in enumerate(options):
    if st.sidebar.button(option):
        selected_option = i

if selected_option is not None:
    st.title("Legal AI Helper")
    st.markdown("Emphasizes the AI-driven legal support system.")
    st.header(options[selected_option])
    input_text = st.text_input("Enter input:")
    if st.button("Submit"):
        if selected_option == 0:
            output = Feature_1_RAG(input_text)
        elif selected_option == 1:
            output = search_laws_with_generation(input_text)
        elif selected_option == 2:
            output = RAG_feature_3(input_text)
        elif selected_option == 3:
            output = RAG_feature_4(input_text)
        elif selected_option == 4:
            output = RAG_feature_5(input_text)
        
        st.write(output)
