import streamlit as st
import os
import time
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

# Set page configuration
st.set_page_config(page_title="Programming TA", page_icon="💻", layout="wide")

# Initialize session state variables if they don't exist
if 'feedback_history' not in st.session_state:
    st.session_state.feedback_history = []
if 'student_profiles' not in st.session_state:
    st.session_state.student_profiles = {}
if 'current_student' not in st.session_state:
    st.session_state.current_student = None
if 'submissions' not in st.session_state:
    st.session_state.submissions = {}


# Groq API configuration
def configure_groq_api():
    api_key = st.session_state.get("groq_api_key", "")
    if not api_key:
        st.sidebar.error("Please enter your Groq API key")
    return api_key


# Function to format code with syntax highlighting
def format_code(code, language):
    try:
        lexer = get_lexer_by_name(language)
        formatter = HtmlFormatter(style="default", linenos=True)
        result = highlight(code, lexer, formatter)
        css = formatter.get_style_defs('.highlight')
        return f"<style>{css}</style>{result}"
    except Exception:
        return f"<pre>{code}</pre>"


# Function to analyze code using Groq API
def analyze_code(code, language, student_id, assignment_name):
    api_key = configure_groq_api()
    if not api_key:
        return None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Get student's history for context
    student_history = []
    if student_id in st.session_state.student_profiles:
        student_history = st.session_state.student_profiles[student_id].get("history", [])

    # Format prompt for the LLM
    prompt = f"""
    You are an expert programming teacher's assistant. Analyze the following {language} code submission for a student.

    STUDENT INFORMATION:
    - Student ID: {student_id}
    - Assignment: {assignment_name}
    - Previous feedback patterns: {json.dumps(student_history[-3:] if len(student_history) > 0 else [])}

    CODE:
    ```{language}
    {code}
    ```

    Provide a comprehensive analysis in the following JSON format:
    {{
        "syntax_errors": [
            {{"line": <line_number>, "description": "<description>", "suggestion": "<suggestion>"}}
        ],
        "logic_errors": [
            {{"description": "<description>", "affected_lines": [<line_numbers>], "suggestion": "<suggestion>"}}
        ],
        "style_issues": [
            {{"line": <line_number>, "description": "<description>", "suggestion": "<suggestion>"}}
        ],
        "efficiency_concerns": [
            {{"description": "<description>", "affected_lines": [<line_numbers>], "suggestion": "<suggestion>"}}
        ],
        "conceptual_misunderstandings": [
            {{"concept": "<concept_name>", "description": "<description>", "resources": ["<resource_url>", "<resource_description>"]}}
        ],
        "positive_aspects": [
            "<positive_comment>"
        ],
        "overall_feedback": "<general feedback>",
        "suggested_resources": [
            {{"title": "<resource_title>", "url": "<resource_url>", "reason": "<why this is helpful>"}}
        ],
        "grade_estimate": "<estimated grade out of 100>"
    }}

    Only respond with the JSON. Do not include any other text in your response.
    """

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": "llama3-70b-8192",  # Using Llama 3.3 via Groq
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2
            },
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            feedback = result["choices"][0]["message"]["content"]

            # Clean up the response to ensure valid JSON
            feedback = re.sub(r'```json', '', feedback)
            feedback = re.sub(r'```', '', feedback)
            feedback = feedback.strip()

            try:
                feedback_json = json.loads(feedback)

                # Add timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                feedback_json["timestamp"] = timestamp
                feedback_json["language"] = language
                feedback_json["assignment"] = assignment_name

                # Update student profile and history
                if student_id not in st.session_state.student_profiles:
                    st.session_state.student_profiles[student_id] = {
                        "history": [],
                        "submissions": 0,
                        "common_issues": {},
                        "strengths": {},
                        "progress": []
                    }

                profile = st.session_state.student_profiles[student_id]
                profile["submissions"] += 1
                profile["history"].append({
                    "timestamp": timestamp,
                    "assignment": assignment_name,
                    "grade_estimate": feedback_json["grade_estimate"],
                    "key_issues": [issue["description"] for issue in feedback_json.get("logic_errors", [])] +
                                  [issue["concept"] for issue in feedback_json.get("conceptual_misunderstandings", [])]
                })

                # Track progress
                profile["progress"].append({
                    "timestamp": timestamp,
                    "assignment": assignment_name,
                    "grade": int(feedback_json["grade_estimate"].split("/")[0]) if "/" in feedback_json[
                        "grade_estimate"] else
                    int(re.search(r'\d+', feedback_json["grade_estimate"]).group()) if re.search(r'\d+', feedback_json[
                        "grade_estimate"]) else 0
                })

                # Store submission
                if student_id not in st.session_state.submissions:
                    st.session_state.submissions[student_id] = []

                st.session_state.submissions[student_id].append({
                    "code": code,
                    "language": language,
                    "assignment": assignment_name,
                    "feedback": feedback_json,
                    "timestamp": timestamp
                })

                # Add to feedback history
                st.session_state.feedback_history.append({
                    "student_id": student_id,
                    "assignment": assignment_name,
                    "timestamp": timestamp,
                    "grade_estimate": feedback_json["grade_estimate"]
                })

                return feedback_json
            except json.JSONDecodeError as e:
                st.error(f"Error parsing JSON response: {e}")
                st.code(feedback)
                return None
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        return None


# Main app layout
def main():
    st.title("💻 AI Programming Teaching Assistant")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # API Key input
        if "groq_api_key" not in st.session_state:
            st.session_state.groq_api_key = ""

        api_key = st.text_input("Enter Groq API Key:", value=st.session_state.groq_api_key, type="password")
        if api_key != st.session_state.groq_api_key:
            st.session_state.groq_api_key = api_key

        st.divider()

        # Student selection
        st.subheader("Student Management")

        # Add new student
        new_student = st.text_input("Add new student ID:")
        if st.button("Add Student") and new_student:
            if new_student not in st.session_state.student_profiles:
                st.session_state.student_profiles[new_student] = {
                    "history": [],
                    "submissions": 0,
                    "common_issues": {},
                    "strengths": {},
                    "progress": []
                }
                st.success(f"Added student: {new_student}")
            else:
                st.warning(f"Student {new_student} already exists")

        # Select existing student
        students = list(st.session_state.student_profiles.keys())
        if students:
            selected_student = st.selectbox("Select Student:", options=students)
            if selected_student != st.session_state.current_student:
                st.session_state.current_student = selected_student
                st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
        else:
            st.info("No students added yet")

        st.divider()

        # App navigation
        st.subheader("Navigation")
        app_mode = st.radio("Select Mode:", ["Code Submission", "Student Analytics", "Class Overview"])

    # Main content area based on selected mode
    if app_mode == "Code Submission":
        display_code_submission()
    elif app_mode == "Student Analytics":
        display_student_analytics()
    else:  # Class Overview
        display_class_overview()


# Code submission interface
def display_code_submission():
    st.header("Code Submission & Analysis")

    if not st.session_state.current_student:
        st.warning("Please select or add a student from the sidebar")
        return

    student_id = st.session_state.current_student
    st.subheader(f"Student: {student_id}")

    col1, col2 = st.columns([3, 2])

    with col1:
        # Code input
        st.subheader("Submit Code")
        language = st.selectbox("Programming Language:",
                                ["python", "java", "javascript", "cpp", "c", "csharp", "go", "ruby", "php"])

        assignment_name = st.text_input("Assignment Name:", "Assignment 1")

        code = st.text_area("Code:", height=300, placeholder="Paste your code here...")

        if st.button("Analyze Code") and code:
            with st.spinner("Analyzing code..."):
                feedback = analyze_code(code, language, student_id, assignment_name)
                if feedback:
                    st.success("Analysis complete!")
                    st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

    with col2:
        # Feedback display
        st.subheader("Previous Submissions")

        if student_id in st.session_state.submissions and len(st.session_state.submissions[student_id]) > 0:
            submissions = st.session_state.submissions[student_id]
            submission_options = [f"{sub['assignment']} - {sub['timestamp']}" for sub in submissions]

            selected_submission = st.selectbox("Select submission:", submission_options,
                                               index=len(submission_options) - 1)
            selected_index = submission_options.index(selected_submission)

            submission = submissions[selected_index]
            feedback = submission["feedback"]

            st.subheader("Feedback Summary")

            # Display estimated grade
            grade_col, timestamp_col = st.columns(2)
            with grade_col:
                st.metric("Estimated Grade", feedback["grade_estimate"])
            with timestamp_col:
                st.info(f"Submitted: {feedback['timestamp']}")

            # Display feedback tabs
            feedback_tab, code_tab = st.tabs(["Detailed Feedback", "Submitted Code"])

            with feedback_tab:
                # Positive aspects
                if feedback.get("positive_aspects"):
                    st.success("**Positive Aspects:**")
                    for item in feedback["positive_aspects"]:
                        st.write(f"✓ {item}")

                # Overall feedback
                st.info(f"**Overall Feedback:** {feedback['overall_feedback']}")

                # Errors and issues
                if feedback.get("syntax_errors") and len(feedback["syntax_errors"]) > 0:
                    with st.expander(f"Syntax Errors ({len(feedback['syntax_errors'])})", expanded=True):
                        for error in feedback["syntax_errors"]:
                            st.warning(f"**Line {error['line']}:** {error['description']}")
                            st.write(f"**Suggestion:** {error['suggestion']}")
                            st.divider()

                if feedback.get("logic_errors") and len(feedback["logic_errors"]) > 0:
                    with st.expander(f"Logic Errors ({len(feedback['logic_errors'])})", expanded=True):
                        for error in feedback["logic_errors"]:
                            affected = ", ".join(map(str, error["affected_lines"]))
                            st.error(f"**Lines {affected}:** {error['description']}")
                            st.write(f"**Suggestion:** {error['suggestion']}")
                            st.divider()

                if feedback.get("style_issues") and len(feedback["style_issues"]) > 0:
                    with st.expander(f"Style Issues ({len(feedback['style_issues'])})", expanded=False):
                        for issue in feedback["style_issues"]:
                            st.info(f"**Line {issue['line']}:** {issue['description']}")
                            st.write(f"**Suggestion:** {issue['suggestion']}")
                            st.divider()

                if feedback.get("efficiency_concerns") and len(feedback["efficiency_concerns"]) > 0:
                    with st.expander(f"Efficiency Concerns ({len(feedback['efficiency_concerns'])})", expanded=False):
                        for concern in feedback["efficiency_concerns"]:
                            affected = ", ".join(map(str, concern["affected_lines"]))
                            st.warning(f"**Lines {affected}:** {concern['description']}")
                            st.write(f"**Suggestion:** {concern['suggestion']}")
                            st.divider()

                # Conceptual misunderstandings
                if feedback.get("conceptual_misunderstandings") and len(feedback["conceptual_misunderstandings"]) > 0:
                    with st.expander("Conceptual Misunderstandings", expanded=True):
                        for concept in feedback["conceptual_misunderstandings"]:
                            st.error(f"**Concept:** {concept['concept']}")
                            st.write(f"**Description:** {concept['description']}")
                            if concept.get("resources"):
                                st.write("**Resources:**")
                                for resource in concept["resources"]:
                                    st.write(f"- {resource}")
                            st.divider()

                # Suggested resources
                if feedback.get("suggested_resources") and len(feedback["suggested_resources"]) > 0:
                    with st.expander("Suggested Resources", expanded=False):
                        for resource in feedback["suggested_resources"]:
                            st.write(f"**{resource['title']}**")
                            st.write(f"- {resource['reason']}")
                            if resource.get("url"):
                                st.markdown(f"- [Link]({resource['url']})")
                            st.divider()

            with code_tab:
                st.markdown(f"**Language:** {submission['language']}")
                st.markdown(f"**Assignment:** {submission['assignment']}")
                st.markdown(format_code(submission["code"], submission["language"]), unsafe_allow_html=True)
        else:
            st.info("No submissions found for this student")


# Student analytics interface
def display_student_analytics():
    st.header("Student Analytics Dashboard")

    if not st.session_state.current_student:
        st.warning("Please select or add a student from the sidebar")
        return

    student_id = st.session_state.current_student

    if student_id not in st.session_state.student_profiles or not st.session_state.student_profiles[student_id][
        "history"]:
        st.info("No data available for this student yet")
        return

    profile = st.session_state.student_profiles[student_id]

    # Overview metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Submissions", profile["submissions"])
    with col2:
        if profile["progress"]:
            latest_grade = profile["progress"][-1]["grade"]
            st.metric("Latest Grade", f"{latest_grade}/100")
    with col3:
        if len(profile["progress"]) > 1:
            first_grade = profile["progress"][0]["grade"]
            latest_grade = profile["progress"][-1]["grade"]
            improvement = latest_grade - first_grade
            st.metric("Overall Improvement", f"{improvement} points", delta=improvement)

    # Progress over time
    st.subheader("Grade Progress")
    if profile["progress"]:
        progress_df = pd.DataFrame(profile["progress"])
        fig = px.line(progress_df, x="timestamp", y="grade", markers=True,
                      labels={"timestamp": "Submission Date", "grade": "Grade"},
                      title="Grade Progress Over Time")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    # Submission history
    st.subheader("Submission History")
    if profile["history"]:
        history_df = pd.DataFrame(profile["history"])

        # Get unique key issues
        all_issues = []
        for entry in profile["history"]:
            all_issues.extend(entry.get("key_issues", []))

        # Count issue occurrences
        issue_counts = {}
        for issue in all_issues:
            if issue in issue_counts:
                issue_counts[issue] += 1
            else:
                issue_counts[issue] = 1

        # Common issues chart
        if issue_counts:
            st.subheader("Common Issues")
            issue_df = pd.DataFrame({"Issue": list(issue_counts.keys()), "Count": list(issue_counts.values())})
            issue_df = issue_df.sort_values("Count", ascending=False).head(5)

            fig = px.bar(issue_df, x="Count", y="Issue", orientation="h",
                         title="Top 5 Most Common Issues")
            st.plotly_chart(fig, use_container_width=True)

        # Submission details
        st.subheader("Recent Submissions")
        for i, entry in enumerate(reversed(profile["history"])):
            if i >= 5:  # Show only the 5 most recent submissions
                break

            with st.expander(f"{entry['assignment']} - {entry['timestamp']}"):
                st.write(f"**Grade:** {entry['grade_estimate']}")
                if entry.get("key_issues"):
                    st.write("**Key Issues:**")
                    for issue in entry["key_issues"]:
                        st.write(f"- {issue}")


# Class overview interface
def display_class_overview():
    st.header("Class Overview Dashboard")

    if not st.session_state.student_profiles:
        st.info("No student data available yet")
        return

    # General class metrics
    total_students = len(st.session_state.student_profiles)
    total_submissions = sum(profile["submissions"] for profile in st.session_state.student_profiles.values())

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Students", total_students)
    with col2:
        st.metric("Total Submissions", total_submissions)

    # Student activity
    st.subheader("Student Activity")

    activity_data = []
    for student_id, profile in st.session_state.student_profiles.items():
        if profile["progress"]:
            latest_grade = profile["progress"][-1]["grade"]
            avg_grade = sum(entry["grade"] for entry in profile["progress"]) / len(profile["progress"])
            activity_data.append({
                "Student ID": student_id,
                "Submissions": profile["submissions"],
                "Latest Grade": latest_grade,
                "Average Grade": round(avg_grade, 1)
            })

    if activity_data:
        activity_df = pd.DataFrame(activity_data)
        st.dataframe(activity_df.sort_values("Latest Grade", ascending=False), use_container_width=True)

        # Grade distribution
        st.subheader("Grade Distribution")

        fig = px.histogram(activity_df, x="Latest Grade", nbins=10,
                           title="Distribution of Latest Grades")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No grade data available yet")

    # Recent feedback
    st.subheader("Recent Feedback")

    if st.session_state.feedback_history:
        history = sorted(st.session_state.feedback_history, key=lambda x: x["timestamp"], reverse=True)[:10]
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True)


if __name__ == "__main__":
    main()