import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to make the app more attractive
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2130;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5 !important;
        color: white !important;
    }
    div[data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, #4e8df5, #8c54ff);
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stDataFrame {
        border: 1px solid #4e8df5;
        border-radius: 10px;
    }
    .css-1544g2n {
        padding-top: 3rem;
    }
    .stExpander {
        border: 1px solid #4e8df5;
        border-radius: 10px;
    }
    .css-1v3fvcr {
        background-color: #0e1117;
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    .css-184tjsw p {
        font-size: 16px;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading to prevent reloading on each interaction
@st.cache_data
def load_data(file_path):
    """Load and preprocess the data"""
    # Read CSV with no preprocessing to preserve original column names
    df = pd.read_csv(file_path, skipinitialspace=True)
    return df

# Define the path to the CSV file - use relative path for better portability
file_path = 'student_performance.csv'

# Main application
try:
    # Load the data with caching
    df = load_data(file_path)

    # Define numerical columns with exact names from the CSV
    attendance_col = 'Attendance (%)'  # Exact column name from CSV
    numerical_cols = ['Hours_Studied', attendance_col, 'Assignments_Submitted', 'Test_Score']

    # Verify all columns exist
    for col in numerical_cols:
        if col not in df.columns:
            st.error(f"Column '{col}' not found in the CSV. Available columns: {list(df.columns)}")
            st.stop()

    # Convert columns to numeric
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values
    df = df.dropna(subset=numerical_cols)

except FileNotFoundError:
    st.error(f"Error: The file '{file_path}' was not found. Please ensure the file is in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while reading the CSV file: {str(e)}")
    st.stop()

# Title and introduction with attractive header
st.markdown("""
<div style="background-color:#4e8df5; padding:10px; border-radius:10px; margin-bottom:20px">
    <h1 style="color:white; text-align:center; font-size:3em">📚 Student Performance Analysis 📊</h1>
    <p style="color:white; text-align:center; font-size:1.2em">
        Comprehensive analysis of student performance metrics including study hours, attendance, assignments, and test scores.
    </p>
</div>
""", unsafe_allow_html=True)

# Add a dashboard summary with metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Average Study Hours", value=f"{df['Hours_Studied'].mean():.1f}",
              delta=f"{df['Hours_Studied'].mean() - 3:.1f} vs Target")
with col2:
    st.metric(label="Average Attendance", value=f"{df['Attendance (%)'].mean():.1f}%",
              delta=f"{df['Attendance (%)'].mean() - 75:.1f}%" if df['Attendance (%)'].mean() - 75 > 0 else f"{df['Attendance (%)'].mean() - 75:.1f}%")
with col3:
    st.metric(label="Average Assignments", value=f"{df['Assignments_Submitted'].mean():.1f}",
              delta=f"{df['Assignments_Submitted'].mean() - 3:.1f}")
with col4:
    st.metric(label="Average Test Score", value=f"{df['Test_Score'].mean():.1f}",
              delta=f"{df['Test_Score'].mean() - 60:.1f}")

# Display tabs for different sections with custom styling
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Data Preview", "📊 Histograms", "📦 Box Plots", "💡 Insights", "🔍 Filtered Analysis"])

with tab1:
    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #4e8df5">
            <h3 style="color:#4e8df5; text-align:center">📋 Student Data</h3>
        </div>
        """, unsafe_allow_html=True)

        # Apply custom styling to the dataframe
        st.dataframe(
            df.style.background_gradient(
                cmap='Blues',
                subset=['Hours_Studied', 'Attendance (%)', 'Assignments_Submitted', 'Test_Score']
            ).format({
                'Hours_Studied': '{:.1f}',
                'Attendance (%)': '{:.1f}%',
                'Test_Score': '{:.1f}'
            }),
            use_container_width=True,
            height=400
        )

    with col2:
        st.markdown("""
        <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #4e8df5">
            <h3 style="color:#4e8df5; text-align:center">📊 Summary Statistics</h3>
        </div>
        """, unsafe_allow_html=True)

        # Display summary statistics with better styling
        st.dataframe(
            df[numerical_cols].describe().style.background_gradient(cmap='Blues').format("{:.2f}"),
            use_container_width=True,
            height=400
        )



with tab2:
    st.markdown("""
    <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #4e8df5">
        <h3 style="color:#4e8df5; text-align:center">📊 Data Distribution Analysis</h3>
        <p style="color:white; text-align:center">
            Histograms showing the distribution of key performance metrics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create columns for better layout
    cols = st.columns(2)

    # Use a more readable display name for plots
    display_names = {
        'Hours_Studied': 'Hours Studied',
        'Attendance (%)': 'Attendance Percentage',
        'Assignments_Submitted': 'Assignments Submitted',
        'Test_Score': 'Test Score'
    }

    # Custom colors for better visualization
    colors = ['#4e8df5', '#8c54ff', '#5ab4bd', '#ff6b6b']

    for i, col in enumerate(numerical_cols):
        with cols[i % 2]:
            # Create a card-like container for each plot
            st.markdown(f"""
            <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid {colors[i]}; margin-bottom:15px">
                <h4 style="color:{colors[i]}; text-align:center">{display_names.get(col, col)}</h4>
            </div>
            """, unsafe_allow_html=True)

            # Create enhanced histogram with progress indicator
            with st.spinner(f'Creating histogram for {display_names.get(col, col)}...'):
                fig, ax = plt.subplots(figsize=(7, 5), facecolor='#1e2130')
                ax.set_facecolor('#1e2130')

                # Create histogram with custom styling
                sns.histplot(df[col], kde=True, bins=10, ax=ax, color=colors[i])

                # Add tooltips with student names at each data point
                if col == 'Test_Score':
                    top_student = df.loc[df[col].idxmax()]
                    top_val = top_student[col]
                    top_name = top_student['Name']
                    ax.annotate(f"{top_name}: {top_val}",
                                xy=(top_val, 0.2),
                                xytext=(top_val, 1),
                                arrowprops=dict(facecolor='white', shrink=0.05, width=2, headwidth=8),
                                fontsize=9, color='white', ha='center')

                # Add mean line
                mean_val = df[col].mean()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {mean_val:.2f}')

                # Add styling to the plot
                ax.set_title(f'Distribution of {display_names.get(col, col)}', color='white', fontsize=14)
                ax.set_xlabel(display_names.get(col, col), color='white', fontsize=12)
                ax.set_ylabel('Frequency', color='white', fontsize=12)
                ax.tick_params(colors='white')
                ax.legend(facecolor='#1e2130', edgecolor=colors[i], loc='upper right', framealpha=0.7)

                # Change spine colors
                for spine in ax.spines.values():
                    spine.set_color('white')

                # Add grid for better readability
                ax.grid(True, linestyle='--', alpha=0.3)

                # Add text annotations in a simpler format
                stats_text = f"Mean: {df[col].mean():.2f}  |  Median: {df[col].median():.2f}"
                props = dict(boxstyle='round,pad=0.5', facecolor='#1e2130', alpha=0.7, edgecolor=colors[i])
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props, color='white')

                fig.tight_layout()
                st.pyplot(fig)

with tab3:
    st.markdown("""
    <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #4e8df5">
        <h3 style="color:#4e8df5; text-align:center">📦 Outlier Analysis</h3>
        <p style="color:white; text-align:center">
            Box plots showing the distribution and potential outliers in student performance metrics
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create columns for better layout
    cols = st.columns(2)

    # Reuse the colors and display names from the previous tab
    for i, col in enumerate(numerical_cols):
        with cols[i % 2]:
            # Create a card-like container for each plot
            st.markdown(f"""
            <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid {colors[i]}; margin-bottom:15px">
                <h4 style="color:{colors[i]}; text-align:center">{display_names.get(col, col)} - Outlier Analysis</h4>
            </div>
            """, unsafe_allow_html=True)

            # Create enhanced boxplot with progress indicator
            with st.spinner(f'Creating boxplot for {display_names.get(col, col)}...'):
                fig, ax = plt.subplots(figsize=(7, 5), facecolor='#1e2130')
                ax.set_facecolor('#1e2130')

                # Create boxplot with custom styling
                sns.boxplot(y=df[col], ax=ax, color=colors[i], width=0.5)

                # Add individual data points for better visualization
                sns.stripplot(y=df[col], ax=ax, color='white', alpha=0.5, size=4, jitter=True)

                # Add annotations for notable students
                if col == 'Test_Score':
                    # Annotate top performer
                    top_student = df.loc[df[col].idxmax()]
                    ax.annotate(f"Top: {top_student['Name']} ({top_student[col]})",
                                xy=(0, top_student[col]),
                                xytext=(0.3, top_student[col]),
                                arrowprops=dict(facecolor='#06d6a0', shrink=0.05, width=2, headwidth=8),
                                fontsize=9, color='white', ha='center')
                elif col == 'Hours_Studied':
                    # Annotate most studious
                    studious = df.loc[df[col].idxmax()]
                    ax.annotate(f"Most studious: {studious['Name']}",
                                xy=(0, studious[col]),
                                xytext=(0.3, studious[col]),
                                arrowprops=dict(facecolor='#4e8df5', shrink=0.05, width=2, headwidth=8),
                                fontsize=9, color='white', ha='center')
                elif col == 'Attendance (%)':
                    # Annotate best attendance
                    best_att = df.loc[df[col].idxmax()]
                    ax.annotate(f"Best attendance: {best_att['Name']}",
                                xy=(0, best_att[col]),
                                xytext=(0.3, best_att[col]),
                                arrowprops=dict(facecolor='#8c54ff', shrink=0.05, width=2, headwidth=8),
                                fontsize=9, color='white', ha='center')

                # Add styling to the plot
                ax.set_title(f'Outlier Detection for {display_names.get(col, col)}', color='white', fontsize=14)
                ax.set_ylabel(display_names.get(col, col), color='white', fontsize=12)
                ax.tick_params(colors='white')

                # Change spine colors
                for spine in ax.spines.values():
                    spine.set_color('white')

                # Add grid for better readability
                ax.grid(True, linestyle='--', alpha=0.3)

                # Calculate and display outlier information
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

                # Add text annotations in a simpler format
                stats_text = f"Median: {df[col].median():.2f}  |  IQR: {IQR:.2f}  |  Outliers: {len(outliers)}"
                props = dict(boxstyle='round,pad=0.5', facecolor='#1e2130', alpha=0.7, edgecolor=colors[i])
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props, color='white')

                fig.tight_layout()
                st.pyplot(fig)

                # If there are outliers, show them in a table
                if not outliers.empty:
                    with st.expander(f"View {len(outliers)} outliers in {display_names.get(col, col)}"):
                        st.dataframe(
                            outliers.style.background_gradient(cmap='Reds', subset=[col]),
                            use_container_width=True
                        )



# Add insights tab content
with tab4:
    st.markdown("""
    <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #4e8df5">
        <h3 style="color:#4e8df5; text-align:center">💡 Key Insights</h3>
        <p style="color:white; text-align:center">
            Discover meaningful patterns and insights from the student performance data
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Calculate key insights from the data
    top_performer = df.loc[df['Test_Score'].idxmax()]
    lowest_performer = df.loc[df['Test_Score'].idxmin()]
    highest_attendance = df.loc[df['Attendance (%)'].idxmax()]
    most_studious = df.loc[df['Hours_Studied'].idxmax()]

    # Create a correlation matrix for insights
    corr = df[numerical_cols].corr()

    # Create two columns for the insights cards
    col1, col2 = st.columns(2)

    # Top performers card
    with col1:
        st.markdown("""
        <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #4e8df5; margin-bottom:20px">
            <h4 style="color:#4e8df5; text-align:center">🏆 Top Performers</h4>
        </div>
        """, unsafe_allow_html=True)

        # Create a styled container for top performers
        st.markdown(f"""
        <div style="background-color:#1e2130; padding:15px; border-radius:10px; border:1px solid #06d6a0; margin-bottom:15px">
            <h5 style="color:#06d6a0; text-align:center">Highest Test Score</h5>
            <div style="display:flex; justify-content:space-between; color:white; margin-bottom:10px">
                <span><b>Student:</b> {top_performer['Name']}</span>
                <span><b>Score:</b> {top_performer['Test_Score']}</span>
            </div>
            <div style="background-color:#0e1117; padding:10px; border-radius:5px; color:white">
                <p><b>Study Hours:</b> {top_performer['Hours_Studied']} hours</p>
                <p><b>Attendance:</b> {top_performer['Attendance (%)']}%</p>
                <p><b>Assignments:</b> {top_performer['Assignments_Submitted']} submitted</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Most studious student
        st.markdown(f"""
        <div style="background-color:#1e2130; padding:15px; border-radius:10px; border:1px solid #4e8df5; margin-bottom:15px">
            <h5 style="color:#4e8df5; text-align:center">Most Studious</h5>
            <div style="display:flex; justify-content:space-between; color:white; margin-bottom:10px">
                <span><b>Student:</b> {most_studious['Name']}</span>
                <span><b>Hours:</b> {most_studious['Hours_Studied']}</span>
            </div>
            <div style="background-color:#0e1117; padding:10px; border-radius:5px; color:white">
                <p><b>Test Score:</b> {most_studious['Test_Score']}</p>
                <p><b>Attendance:</b> {most_studious['Attendance (%)']}%</p>
                <p><b>Assignments:</b> {most_studious['Assignments_Submitted']} submitted</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Highest attendance
        st.markdown(f"""
        <div style="background-color:#1e2130; padding:15px; border-radius:10px; border:1px solid #8c54ff; margin-bottom:15px">
            <h5 style="color:#8c54ff; text-align:center">Best Attendance</h5>
            <div style="display:flex; justify-content:space-between; color:white; margin-bottom:10px">
                <span><b>Student:</b> {highest_attendance['Name']}</span>
                <span><b>Attendance:</b> {highest_attendance['Attendance (%)']}%</span>
            </div>
            <div style="background-color:#0e1117; padding:10px; border-radius:5px; color:white">
                <p><b>Test Score:</b> {highest_attendance['Test_Score']}</p>
                <p><b>Study Hours:</b> {highest_attendance['Hours_Studied']} hours</p>
                <p><b>Assignments:</b> {highest_attendance['Assignments_Submitted']} submitted</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Correlation insights and areas of improvement
    with col2:
        st.markdown("""
        <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #ff6b6b; margin-bottom:20px">
            <h4 style="color:#ff6b6b; text-align:center">📈 Performance Factors</h4>
        </div>
        """, unsafe_allow_html=True)

        # Create a styled container for correlation insights
        hours_score_corr = corr.loc['Hours_Studied', 'Test_Score']
        attendance_score_corr = corr.loc['Attendance (%)', 'Test_Score']
        assignments_score_corr = corr.loc['Assignments_Submitted', 'Test_Score']

        # Format the correlations as percentages
        hours_impact = f"{abs(hours_score_corr) * 100:.1f}%"
        attendance_impact = f"{abs(attendance_score_corr) * 100:.1f}%"
        assignments_impact = f"{abs(assignments_score_corr) * 100:.1f}%"

        st.markdown(f"""
        <div style="background-color:#1e2130; padding:15px; border-radius:10px; border:1px solid #ff6b6b; margin-bottom:15px">
            <h5 style="color:#ff6b6b; text-align:center">Impact on Test Scores</h5>
            <div style="background-color:#0e1117; padding:10px; border-radius:5px; color:white; margin-bottom:10px">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px">
                    <span><b>Study Hours:</b></span>
                    <span>{hours_impact} correlation</span>
                </div>
                <div style="width:100%; background-color:#1e2130; height:10px; border-radius:5px">
                    <div style="width:{abs(hours_score_corr) * 100}%; background-color:#4e8df5; height:10px; border-radius:5px"></div>
                </div>
            </div>
            <div style="background-color:#0e1117; padding:10px; border-radius:5px; color:white; margin-bottom:10px">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px">
                    <span><b>Attendance:</b></span>
                    <span>{attendance_impact} correlation</span>
                </div>
                <div style="width:100%; background-color:#1e2130; height:10px; border-radius:5px">
                    <div style="width:{abs(attendance_score_corr) * 100}%; background-color:#8c54ff; height:10px; border-radius:5px"></div>
                </div>
            </div>
            <div style="background-color:#0e1117; padding:10px; border-radius:5px; color:white">
                <div style="display:flex; justify-content:space-between; margin-bottom:5px">
                    <span><b>Assignments:</b></span>
                    <span>{assignments_impact} correlation</span>
                </div>
                <div style="width:100%; background-color:#1e2130; height:10px; border-radius:5px">
                    <div style="width:{abs(assignments_score_corr) * 100}%; background-color:#06d6a0; height:10px; border-radius:5px"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Areas for improvement - students with low scores
        low_performers = df[df['Test_Score'] < 40].sort_values('Test_Score')

        if not low_performers.empty:
            # Create a container for low performers
            st.markdown("""
            <div style="background-color:#1e2130; padding:15px; border-radius:10px; border:1px solid #ffd166; margin-bottom:15px">
                <h5 style="color:#ffd166; text-align:center">Areas for Improvement</h5>
                <div style="background-color:#0e1117; padding:10px; border-radius:5px; color:white">
            """, unsafe_allow_html=True)

            # Add each low performer individually to avoid string concatenation issues
            for _, student in low_performers.iterrows():
                needs_improvement = []
                if student['Hours_Studied'] < 3:
                    needs_improvement.append("Study Hours")
                if student['Attendance (%)'] < 70:
                    needs_improvement.append("Attendance")
                if student['Assignments_Submitted'] < 3:
                    needs_improvement.append("Assignment completion")

                improvement_text = ", ".join(needs_improvement)

                st.markdown(f"""
                <div style="border-bottom:1px solid #1e2130; padding:5px 0">
                    <div style="display:flex; justify-content:space-between">
                        <span><b>{student['Name']}</b></span>
                        <span>Score: {student['Test_Score']}</span>
                    </div>
                    <div style="font-size:0.9em; color:#aaa">
                        Needs improvement in: {improvement_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Close the container
            st.markdown("""
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Add outlier insights
    st.markdown("""
    <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #5ab4bd; margin-top:20px; margin-bottom:20px">
        <h4 style="color:#5ab4bd; text-align:center">🔍 Outlier Analysis</h4>
    </div>
    """, unsafe_allow_html=True)

    # Create columns for outlier insights
    col1, col2 = st.columns(2)

    # Function to identify outliers
    def get_outliers(column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    # Check for outliers in each column
    for i, col in enumerate(numerical_cols):
        outliers = get_outliers(col)
        col_idx = i % 2

        if col_idx == 0:
            container = col1
        else:
            container = col2

        with container:
            if not outliers.empty:
                # Create a container for outliers
                outlier_html = f"""
                <div style="background-color:#1e2130; padding:15px; border-radius:10px; border:1px solid {colors[i]}; margin-bottom:15px">
                    <h5 style="color:{colors[i]}; text-align:center">{display_names.get(col, col)} Outliers</h5>
                    <div style="background-color:#0e1117; padding:10px; border-radius:5px; color:white">
                """

                # Add each outlier student to the HTML
                for _, student in outliers.iterrows():
                    is_high = student[col] > df[col].mean()
                    icon = "⬆️" if is_high else "⬇️"
                    status = "exceptionally high" if is_high else "exceptionally low"

                    outlier_html += f"""
                    <div style="margin-bottom:10px">
                        <div style="display:flex; justify-content:space-between">
                            <span><b>{student['Name']}</b> {icon}</span>
                            <span>{student[col]}</span>
                        </div>
                        <div style="font-size:0.9em; color:#aaa">
                            Has {status} {display_names.get(col, col).lower()}
                        </div>
                    </div>
                    """

                # Close the HTML tags and display the complete HTML
                outlier_html += """
                    </div>
                </div>
                """

                st.markdown(outlier_html, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color:#1e2130; padding:15px; border-radius:10px; border:1px solid {colors[i]}; margin-bottom:15px">
                    <h5 style="color:{colors[i]}; text-align:center">{display_names.get(col, col)}</h5>
                    <div style="background-color:#0e1117; padding:10px; border-radius:5px; color:white; text-align:center">
                        No outliers detected in this metric
                    </div>
                </div>
                """, unsafe_allow_html=True)

# Add filtered analysis tab content
with tab5:
    st.markdown("""
    <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #4e8df5">
        <h3 style="color:#4e8df5; text-align:center">🔍 Filtered Data Analysis</h3>
        <p style="color:white; text-align:center">
            Apply filters to visualize specific subsets of student data
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create a layout with sidebar for filters and main area for visualizations
    filter_col, viz_col = st.columns([1, 2])

    with filter_col:
        st.markdown("""
        <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #8c54ff; margin-bottom:15px">
            <h4 style="color:#8c54ff; text-align:center">Filter Options</h4>
        </div>
        """, unsafe_allow_html=True)

        # Student name filter (multiselect)
        selected_students = st.multiselect(
            "Select Students",
            options=df['Name'].tolist(),
            default=[],
            help="Select one or more students to filter the data"
        )

        # Study hours range filter
        hours_min = float(df['Hours_Studied'].min())
        hours_max = float(df['Hours_Studied'].max())
        hours_range = st.slider(
            "Study Hours Range",
            min_value=hours_min,
            max_value=hours_max,
            value=(hours_min, hours_max),
            step=0.5,
            help="Filter students by their study hours"
        )

        # Attendance percentage range filter
        attendance_min = float(df['Attendance (%)'].min())
        attendance_max = float(df['Attendance (%)'].max())
        attendance_range = st.slider(
            "Attendance Percentage Range",
            min_value=attendance_min,
            max_value=attendance_max,
            value=(attendance_min, attendance_max),
            step=5.0,
            help="Filter students by their attendance percentage"
        )

        # Assignments submitted range filter
        assignments_min = int(df['Assignments_Submitted'].min())
        assignments_max = int(df['Assignments_Submitted'].max())
        assignments_range = st.slider(
            "Assignments Submitted Range",
            min_value=assignments_min,
            max_value=assignments_max,
            value=(assignments_min, assignments_max),
            step=1,
            help="Filter students by the number of assignments they submitted"
        )

        # Test score range filter
        score_min = float(df['Test_Score'].min())
        score_max = float(df['Test_Score'].max())
        score_range = st.slider(
            "Test Score Range",
            min_value=score_min,
            max_value=score_max,
            value=(score_min, score_max),
            step=5.0,
            help="Filter students by their test scores"
        )

        # Apply filters button
        st.markdown("<br>", unsafe_allow_html=True)
        filter_button = st.button("Apply Filters", type="primary", use_container_width=True)
        reset_button = st.button("Reset Filters", type="secondary", use_container_width=True)

        # Visualization type selector
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #ff6b6b; margin-bottom:15px">
            <h4 style="color:#ff6b6b; text-align:center">Visualization Options</h4>
        </div>
        """, unsafe_allow_html=True)

        viz_type = st.radio(
            "Select Visualization Type",
            options=["Scatter Plot", "Bar Chart", "Data Table"],
            horizontal=True
        )

        # For scatter plot, allow selecting x and y axes
        if viz_type == "Scatter Plot":
            x_axis = st.selectbox(
                "X-Axis",
                options=numerical_cols,
                index=0
            )
            y_axis = st.selectbox(
                "Y-Axis",
                options=numerical_cols,
                index=3  # Default to Test_Score
            )

    # Main visualization area
    with viz_col:
        # Apply filters to the dataframe
        filtered_df = df.copy()

        # Reset filters if reset button is clicked
        if reset_button:
            selected_students = []
            hours_range = (hours_min, hours_max)
            attendance_range = (attendance_min, attendance_max)
            assignments_range = (assignments_min, assignments_max)
            score_range = (score_min, score_max)

        # Apply name filter if any names are selected
        if selected_students:
            filtered_df = filtered_df[filtered_df['Name'].isin(selected_students)]

        # Apply range filters
        filtered_df = filtered_df[
            (filtered_df['Hours_Studied'] >= hours_range[0]) &
            (filtered_df['Hours_Studied'] <= hours_range[1]) &
            (filtered_df['Attendance (%)'] >= attendance_range[0]) &
            (filtered_df['Attendance (%)'] <= attendance_range[1]) &
            (filtered_df['Assignments_Submitted'] >= assignments_range[0]) &
            (filtered_df['Assignments_Submitted'] <= assignments_range[1]) &
            (filtered_df['Test_Score'] >= score_range[0]) &
            (filtered_df['Test_Score'] <= score_range[1])
        ]

        # Display filter summary
        st.markdown(f"""
        <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #4e8df5; margin-bottom:15px">
            <h4 style="color:#4e8df5; text-align:center">Filtered Results</h4>
            <p style="color:white; text-align:center">
                Showing {len(filtered_df)} out of {len(df)} students
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Check if filtered dataframe is empty
        if filtered_df.empty:
            st.warning("No students match the selected filters. Please adjust your filter criteria.")
        else:
            # Display the selected visualization
            if viz_type == "Scatter Plot":
                st.markdown(f"""
                <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #8c54ff; margin-bottom:15px">
                    <h4 style="color:#8c54ff; text-align:center">Scatter Plot: {display_names.get(x_axis, x_axis)} vs {display_names.get(y_axis, y_axis)}</h4>
                </div>
                """, unsafe_allow_html=True)

                # Create scatter plot
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e2130')
                ax.set_facecolor('#1e2130')

                # Plot each point with student name as hover text
                scatter = ax.scatter(
                    filtered_df[x_axis],
                    filtered_df[y_axis],
                    c=filtered_df['Test_Score'],  # Color by test score
                    cmap='viridis',
                    s=100,  # Point size
                    alpha=0.8
                )

                # Add a color bar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Test Score', color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='white')

                # Add student names as annotations
                for i, row in filtered_df.iterrows():
                    ax.annotate(
                        row['Name'],
                        (row[x_axis], row[y_axis]),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=8,
                        color='white'
                    )

                # Add styling to the plot
                ax.set_title(f'{display_names.get(x_axis, x_axis)} vs {display_names.get(y_axis, y_axis)}', color='white', fontsize=14)
                ax.set_xlabel(display_names.get(x_axis, x_axis), color='white', fontsize=12)
                ax.set_ylabel(display_names.get(y_axis, y_axis), color='white', fontsize=12)
                ax.tick_params(colors='white')

                # Add grid for better readability
                ax.grid(True, linestyle='--', alpha=0.3)

                # Change spine colors
                for spine in ax.spines.values():
                    spine.set_color('white')

                # Add trend line
                if len(filtered_df) > 1:  # Need at least 2 points for a trend line
                    z = np.polyfit(filtered_df[x_axis], filtered_df[y_axis], 1)
                    p = np.poly1d(z)
                    ax.plot(filtered_df[x_axis], p(filtered_df[x_axis]), "r--", alpha=0.8,
                            label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
                    ax.legend(facecolor='#1e2130', edgecolor='#8c54ff', loc='upper left', framealpha=0.7)

                fig.tight_layout()
                st.pyplot(fig)

                # Add correlation information
                if len(filtered_df) > 1:
                    corr_value = filtered_df[x_axis].corr(filtered_df[y_axis])
                    corr_strength = "strong" if abs(corr_value) > 0.7 else "moderate" if abs(corr_value) > 0.3 else "weak"
                    corr_direction = "positive" if corr_value > 0 else "negative"

                    st.markdown(f"""
                    <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #4e8df5; margin-top:15px">
                        <h5 style="color:#4e8df5; text-align:center">Correlation Analysis</h5>
                        <p style="color:white; text-align:center">
                            The correlation between {display_names.get(x_axis, x_axis)} and {display_names.get(y_axis, y_axis)} is {corr_value:.2f},
                            indicating a {corr_strength} {corr_direction} relationship.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

            elif viz_type == "Bar Chart":
                st.markdown("""
                <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #ff6b6b; margin-bottom:15px">
                    <h4 style="color:#ff6b6b; text-align:center">Bar Chart Comparison</h4>
                </div>
                """, unsafe_allow_html=True)

                # Select metric for bar chart
                metric = st.selectbox(
                    "Select Metric to Compare",
                    options=numerical_cols,
                    index=3  # Default to Test_Score
                )

                # Sort by selected metric
                sorted_df = filtered_df.sort_values(by=metric, ascending=False)

                # Create bar chart
                fig, ax = plt.subplots(figsize=(10, 6), facecolor='#1e2130')
                ax.set_facecolor('#1e2130')

                # Create bars with custom colors
                bars = ax.bar(
                    sorted_df['Name'],
                    sorted_df[metric],
                    color=plt.cm.viridis(sorted_df[metric]/sorted_df[metric].max()),
                    alpha=0.8
                )

                # Add styling to the plot
                ax.set_title(f'Comparison of {display_names.get(metric, metric)} by Student', color='white', fontsize=14)
                ax.set_xlabel('Student Name', color='white', fontsize=12)
                ax.set_ylabel(display_names.get(metric, metric), color='white', fontsize=12)
                ax.tick_params(colors='white')

                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45, ha='right')

                # Add grid for better readability
                ax.grid(True, linestyle='--', alpha=0.3, axis='y')

                # Change spine colors
                for spine in ax.spines.values():
                    spine.set_color('white')

                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.1,
                        f'{height:.1f}',
                        ha='center',
                        va='bottom',
                        color='white',
                        fontsize=8
                    )

                fig.tight_layout()
                st.pyplot(fig)

                # Add average line
                avg_value = sorted_df[metric].mean()
                ax.axhline(y=avg_value, color='red', linestyle='--', alpha=0.8)
                ax.text(
                    len(sorted_df) - 1,
                    avg_value + 0.5,
                    f'Average: {avg_value:.1f}',
                    color='white',
                    ha='right'
                )

                fig.tight_layout()
                st.pyplot(fig)

            elif viz_type == "Data Table":
                st.markdown("""
                <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #06d6a0; margin-bottom:15px">
                    <h4 style="color:#06d6a0; text-align:center">Filtered Data Table</h4>
                </div>
                """, unsafe_allow_html=True)

                # Sort options
                sort_col = st.selectbox(
                    "Sort By",
                    options=["Name"] + numerical_cols,
                    index=0
                )

                sort_order = st.radio(
                    "Sort Order",
                    options=["Ascending", "Descending"],
                    horizontal=True
                )

                # Sort the dataframe
                sorted_df = filtered_df.sort_values(
                    by=sort_col,
                    ascending=(sort_order == "Ascending")
                )

                # Display the sorted dataframe with styling
                st.dataframe(
                    sorted_df.style.background_gradient(
                        cmap='Blues',
                        subset=['Hours_Studied', 'Attendance (%)', 'Assignments_Submitted', 'Test_Score']
                    ).format({
                        'Hours_Studied': '{:.1f}',
                        'Attendance (%)': '{:.1f}%',
                        'Test_Score': '{:.1f}'
                    }),
                    use_container_width=True,
                    height=400
                )

                # Add summary statistics for the filtered data
                st.markdown("""
                <div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #06d6a0; margin-top:15px">
                    <h5 style="color:#06d6a0; text-align:center">Summary Statistics for Filtered Data</h5>
                </div>
                """, unsafe_allow_html=True)

                st.dataframe(
                    sorted_df[numerical_cols].describe().style.background_gradient(cmap='Blues').format("{:.2f}"),
                    use_container_width=True
                )

# Add a simple footer
st.markdown("---")
st.markdown("""
<div style="background-color:#1e2130; padding:10px; border-radius:10px; border:1px solid #4e8df5; margin-top:20px">
    <p style="color:white; text-align:center; font-size:0.9em">
        Student Performance Analysis Dashboard
    </p>
</div>
""", unsafe_allow_html=True)

