import streamlit as st

class Onboarding:
    def __init__(self):
        self.steps = [
            "Step 1: Upload your data",
            "Step 2: Fetch weather data",
            "Step 3: Generate forecast",
            "Step 4: Analyze results",
            "Step 5: Get recommendations",
        ]
        self.current_step = 0

    def start(self):
        if 'onboarding_step' not in st.session_state:
            st.session_state.onboarding_step = 0

        if st.session_state.onboarding_step < len(self.steps):
            st.info(self.steps[st.session_state.onboarding_step])
            if st.button("Next"):
                st.session_state.onboarding_step += 1
        else:
            st.success("Onboarding complete!")