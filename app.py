# app.py (Version 3 - With Conversational Chatbot)

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import openai

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Exit Interview Intelligence Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD ANALYZED DATA ---
@st.cache_data
def load_data():
    try:
        with open('analyzed_data.json', 'r') as f:
            data = json.load(f)
        df = pd.json_normalize(data, sep='_')
        return df
    except FileNotFoundError:
        st.error("`analyzed_data.json` not found. Please run the `pre_analyze.py` script first.", icon="🚨")
        return None

df = load_data()

# Initialize OpenAI client from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]


if df is None:
    st.stop()

# --- APP LAYOUT ---
st.title("Optum - Exit Interview Intelligence Platform 📈")

tab1, tab2, tab3 = st.tabs(["📊 Aggregate Dashboard", "🧑‍💼 Individual Interview Analysis", "💬 Conversational AI Assistant"])

# --- TAB 1: AGGREGATE DASHBOARD ---
with tab1:
    st.header("Overall Exit Trends")

    # --- KPIs ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Interviews Analyzed", len(df))
    with col2:
        neg_sentiment_count = df[df['analysis_overallSentiment'] == 'Negative'].shape[0]
        neg_sentiment_perc = (neg_sentiment_count / len(df)) * 100
        st.metric("Negative Sentiment Rate", f"{neg_sentiment_perc:.1f}%")
    with col3:
        top_theme = df['analysis_keyThemes'].explode().value_counts().idxmax()
        st.metric("Top Reason for Leaving", top_theme)

    st.markdown("---")

    # --- CHARTS ---
    col1_charts, col2_charts = st.columns(2)
    with col1_charts:
        st.subheader("Top Reasons for Leaving")
        themes_df = df['analysis_keyThemes'].explode().value_counts().reset_index()
        themes_df.columns = ['Theme', 'Count']
        # --- CHANGE HERE: Removed orientation='h', swapped x and y ---
        fig_themes = px.bar(themes_df, x='Theme', y='Count', title="Frequency of Key Themes")
        st.plotly_chart(fig_themes, use_container_width=True)
        
    with col2_charts:
        st.subheader("Sentiment by Department")
        sentiment_by_dept = df.groupby('department')['analysis_overallSentiment'].value_counts().unstack().fillna(0)

        sentiment_color_map = {
            'Negative': '#E74C3C',  # A clear red
            'Positive': '#2ECC71',  # A nice green
            'Mixed': '#F39C12',     # A warm orange
            'Neutral': '#BDC3C7'   # A neutral grey
        }

        fig_sentiment_dept = px.bar(
            sentiment_by_dept,
            barmode='stack', # <-- CHANGED TO 'stack'
            title="Sentiment Distribution Across Departments",
            color_discrete_map=sentiment_color_map
        )
        st.plotly_chart(fig_sentiment_dept, use_container_width=True)

        # Pass the color map to the plotting function
        fig_sentiment_dept = px.bar(
            sentiment_by_dept,
            barmode='group',
            title="Sentiment Distribution Across Departments",
            color_discrete_map=sentiment_color_map # <-- This is the magic line!
        )
        
        st.plotly_chart(fig_sentiment_dept, use_container_width=True)

# --- TAB 2: INDIVIDUAL ANALYSIS ---
with tab2:
    st.header("Drill Down into Individual Interviews")
    
    df['display_name'] = df['employeeName'] + " (" + df['employeeID'] + ")"
    selected_interview_display = st.selectbox(
        label="Choose an interview to analyze:",
        options=df['display_name']
    )

    selected_interview = df[df['display_name'] == selected_interview_display].iloc[0]

    col1_ind, col2_ind = st.columns(2)
    with col1_ind:
        st.subheader("Interview Details & Transcript")
        st.info(f"""
        **Employee:** {selected_interview['employeeName']} ({selected_interview['employeeID']})
        **Designation:** {selected_interview['designation']}
        **Department:** {selected_interview['department']}
        **Stated Reason for Leaving:** {selected_interview['exitReason']}
        """)
        st.text_area(
            "Full Transcript",
            value=selected_interview['interviewTranscript'],
            height=400,
            disabled=True
        )

    with col2_ind:
        st.subheader("AI-Powered Insights")
        sentiment_color = {
            "Positive": "green", "Negative": "red", "Mixed": "orange", "Neutral": "blue"
        }.get(selected_interview['analysis_overallSentiment'], "blue")

        st.markdown(f"**Overall Sentiment:** :{sentiment_color}[{selected_interview['analysis_overallSentiment']}]")
        st.markdown("**Key Themes:**")
        for theme in selected_interview['analysis_keyThemes']:
            st.markdown(f"- {theme}")
        st.markdown("**AI-Generated Summary:**")
        st.markdown(selected_interview['analysis_summary'])
        if 'analysis_extractedEntities' in selected_interview and selected_interview['analysis_extractedEntities']:
             st.markdown("**Extracted Entities:**")
             for entity in selected_interview['analysis_extractedEntities']:
                 st.markdown(f"- **{entity.get('type', 'N/A')}:** {entity.get('name', 'N/A')}")


# --- TAB 3: CONVERSATIONAL AI ASSISTANT ---
with tab3:
    st.header("Ask the AI Assistant")
    st.markdown("Ask questions about the exit interview data in plain English. The AI will answer based on the analyzed results.")

    # --- The System Prompt: The "Brain" of the Chatbot ---
    # We convert our DataFrame of analyzed data into a JSON string to pass to the model
    data_as_json = df.to_json(orient='records', indent=2)

    SYSTEM_PROMPT = f"""
    You are a quantitative HR Analyst AI assistant for an HR leader at Optum.
    Your primary goal is to provide data-driven, statistical insights from the provided exit interview data.
    You must answer questions based ONLY on the provided JSON data below. Do not make up information.

    **RESPONSE GUIDELINES:**
    1.  **Quantify First:** Always lead with statistics. Use counts, percentages, or fractions (e.g., "The top reason is 'Career Growth', mentioned in 4 out of 8 Engineering departures (50%).").
    2.  **Synthesize, Don't Just List:** Do not list individual summaries one-by-one. Instead, synthesize trends and use individual cases as brief, supporting examples.
    3.  **Structure Your Answers:** Provide a clear headline finding, followed by quantitative evidence, and then a brief qualitative example if relevant.
    4.  **Be Direct and Actionable:** Frame your answers to help the HR leader make decisions.

    Here is the exit interview data:
```json
    {data_as_json}
    ```
    """

    # --- Initialize Chat History ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hello! How can I help you analyze the exit interview data today?"}
        ]

    # --- Display Chat Messages ---
    for message in st.session_state.messages:
        if message["role"] not in ["system"]: # Don't display the system prompt
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # --- Handle User Input ---
    if prompt := st.chat_input("e.g., Why are people in the Engineering department leaving?"):
        # Add user message to session state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = openai.chat.completions.create(
                    model="gpt-4-turbo", # GPT-4 is better at reasoning over structured data
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                )
                response_content = response.choices[0].message.content
                st.markdown(response_content)
        
        # Add AI response to session state
        st.session_state.messages.append({"role": "assistant", "content": response_content})
