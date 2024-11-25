import streamlit as st
from model import process_driving_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.title("Driving Behavior Analysis Dashboard")

    # File uploader for CSV input
    uploaded_file = st.file_uploader("Upload a trip data CSV file", type=["csv"])

    if uploaded_file is not None:
        st.success("File successfully uploaded!")
        try:
            # Convert the uploaded file to a DataFrame
            trip_data = pd.read_csv(uploaded_file)

            # Check if the uploaded file has the required columns
            required_columns = ['Gyroscope_X', 'Gyroscope_Y', 'Gyroscope_Z', 
                                'Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z', 'Timestamp']
            if not all(column in trip_data.columns for column in required_columns):
                raise ValueError(f"Uploaded file must contain the following columns: {', '.join(required_columns)}")

            # Process the data
            left_template_file = "L1.csv"  # Ensure this file exists
            right_template_file = "R1.csv"  # Ensure this file exists
            result = process_driving_data(trip_data, left_template_file, right_template_file)

            # Display Event Counts
            st.header("Event Counts")
            event_counts_df = pd.DataFrame(result['event_counts'].items(), columns=["Event Type", "Count"])
            st.table(event_counts_df)

            # Display Driving Score
            st.header("Driving Score")
            st.metric("Driving Score", result['driving_score'])

            # Visualization 1: Event Count Bar Chart
            st.subheader("Event Count Bar Chart")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=event_counts_df, x="Event Type", y="Count", ax=ax, palette="viridis")
            ax.set_title("Frequency of Driving Events")
            ax.set_xlabel("Event Type")
            ax.set_ylabel("Count")
            st.pyplot(fig)

            # Visualization 2: Driving Score Progress Bar
            st.subheader("Driving Score Gauge")
            st.progress(result['driving_score'] / 100)

            # # Visualization 3: Harsh Events Timeline
            # st.subheader("Harsh Events Timeline")
            # harsh_events = ["Harsh Left", "Harsh Right", "Harsh Acceleration", "Harsh Braking"]
            # timeline_data = [
            #     (event, index)
            #     for index, event, _ in result.get("harsh_event_log", [])  # Ensure to modify the function to log this
            #     if event in harsh_events
            # ]
            # if timeline_data:
            #     timeline_df = pd.DataFrame(timeline_data, columns=["Event Type", "Window Index"])
            #     fig, ax = plt.subplots(figsize=(10, 5))
            #     sns.lineplot(data=timeline_df, x="Window Index", y="Event Type", ax=ax, marker="o")
            #     ax.set_title("Harsh Events Over Time")
            #     ax.set_xlabel("Window Index")
            #     ax.set_ylabel("Event Type")
            #     st.pyplot(fig)
            # else:
            #     st.write("No harsh events detected.")

            # Visualization 4: Sensor Data Line Graphs
            st.subheader("Sensor Data Over Time")
            
            # Plot Accelerometer_X vs Timestamp
            st.write("**Accelerometer_X vs Timestamp**")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=trip_data, x="Timestamp", y="Accelerometer_X", ax=ax, color="blue")
            ax.set_title("Accelerometer_X vs Timestamp")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Accelerometer_X")
            st.pyplot(fig)

            # Plot Accelerometer_Y vs Timestamp
            st.write("**Accelerometer_Y vs Timestamp**")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=trip_data, x="Timestamp", y="Accelerometer_Y", ax=ax, color="green")
            ax.set_title("Accelerometer_Y vs Timestamp")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Accelerometer_Y")
            st.pyplot(fig)

            # Plot Gyroscope_Z vs Timestamp
            st.write("**Gyroscope_Z vs Timestamp**")
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.lineplot(data=trip_data, x="Timestamp", y="Gyroscope_Z", ax=ax, color="red")
            ax.set_title("Gyroscope_Z vs Timestamp")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Gyroscope_Z")
            st.pyplot(fig)

        except ValueError as ve:
            st.error(f"Input Error: {ve}")
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()
