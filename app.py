import streamlit as st

st.set_page_config(
    page_title="VoxGuard AI",
    page_icon="🛡️"
)

st.title("🛡️ VoxGuard AI")
st.subheader("Detect. Verify. Protect.")

st.write(
    "AI-Powered Voice Cloning Detection System"
)

st.divider()

audio = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3", "flac", "ogg", "m4a"]
)

if audio:

    st.audio(audio)

    if st.button("🔍 ANALYZE AUDIO"):

        st.success("Audio received successfully!")

        st.write("Prediction: Processing...")
