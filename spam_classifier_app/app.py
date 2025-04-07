import streamlit as st
import pickle

# Load the model and vectorizer
try:
    with open("spam_classifier_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open("vectorizer.pkl", "rb") as vec_file:
        vectorizer = pickle.load(vec_file)

except FileNotFoundError:
    st.error("Model or vectorizer file not found! Please make sure 'spam_classifier_model.pkl' and 'vectorizer.pkl' are in the same folder as this app.")
    st.stop()

# Page config
st.set_page_config(page_title="Spam Email Classifier", page_icon="📧")

st.title("📬 Spam Email Classifier")
st.markdown("Enter an email message below, and the model will tell you if it's **Spam** or **Not Spam**.")

# User input
email_text = st.text_area("✉️ Enter Email Text:", height=200)

if st.button("🔍 Classify"):
    if email_text.strip() == "":
        st.warning("Please enter some email text.")
    else:
        input_vector = vectorizer.transform([email_text])
        prediction = model.predict(input_vector)

        if prediction[0] == 1:
            st.error("🚨 This is **Spam**!")
        else:
            st.success("✅ This is **Not Spam**.")

st.markdown("---")
st.caption("Made with ❤️ by Karan Tiwari | Powered by Machine Learning")
