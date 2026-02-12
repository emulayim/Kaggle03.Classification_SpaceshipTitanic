import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import numpy as np

# --- Page Config ---
st.set_page_config(
    page_title="Spaceship Titanic Predictor",
    page_icon="🚀",
    layout="wide"
)

# --- Helper Functions ---
def resolve_model_path(filename):
    """
    Resolves the model path for Local vs Hugging Face.
    """
    possible_paths = [
        os.path.join("models", filename),
        os.path.join("src", filename),
        filename,
        os.path.join("..", "models", filename)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def custom_preprocessing(df):
    """
    Mirrors the feature engineering steps from the notebook.
    """
    df = df.copy()
    
    # 1. Cabin Parsleme (Deck/Num/Side)
    if "Cabin" in df.columns:
        df["Cabin"] = df["Cabin"].fillna("Unknown/-1/Unknown")
        # Ensure string type before split
        df["Cabin"] = df["Cabin"].astype(str)
        
        # Safe split
        split_data = df["Cabin"].str.split("/", expand=True)
        if split_data.shape[1] == 3:
            df[["Deck", "Num", "Side"]] = split_data
        else:
            # Fallback if format is bad
            df["Deck"] = "Unknown"
            df["Num"] = -1
            df["Side"] = "Unknown"
        
        # Convert Num
        df["Num"] = pd.to_numeric(df["Num"], errors='coerce')
        
        # Drop original
        df = df.drop(columns=["Cabin"])
    else:
        # Create dummy columns if Cabin is missing (for manual input case)
        df["Deck"] = "Unknown"
        df["Num"] = 0
        df["Side"] = "Unknown"

    # 2. TotalSpend
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    # Ensure columns exist, add if missing (crucial for manual input or partial CSV)
    for col in spend_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    temp_df = df[spend_cols].fillna(0)
    df['TotalSpend'] = temp_df.sum(axis=1)
    
    # 3. Name (Drop)
    if "Name" in df.columns:
        df = df.drop(columns=["Name"])
        
    return df

@st.cache_resource
def load_model():
    model_path = resolve_model_path("best_model.pkl")
    if model_path:
        return joblib.load(model_path)
    return None

# --- Main App ---
def main():
    st.title("🚀 Spaceship Titanic Prediction")
    st.markdown("""
    **[TR]** Yolcuların başka bir boyuta geçip geçmediğini tahmin eden yapay zeka modeli.
    **[EN]** AI model predicting if passengers were transported to an alternate dimension.
    """)

    model = load_model()

    if model is None:
        st.error("🚨 Model file (`best_model.pkl`) not found! Please run the notebook to train the model first.")
        return

    # Tabs
    tab1, tab2 = st.tabs(["📁 Batch Prediction (CSV)", "🧑‍🚀 Manual Input (Single)"])

    # --- TAB 1: CSV Upload ---
    with tab1:
        st.subheader("Upload Test Data (CSV)")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file:
            try:
                raw_df = pd.read_csv(uploaded_file)
                st.write("First 5 rows of uploaded data:")
                st.dataframe(raw_df.head())

                if st.button("Predict Batch"):
                    # Preprocess
                    processed_df = custom_preprocessing(raw_df)
                    
                    # Prepare X
                    if "PassengerId" in processed_df.columns:
                        X = processed_df.drop(columns=["PassengerId"])
                    else:
                        X = processed_df
                    
                    # Predict
                    try:
                        predictions = model.predict(X)
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X)[:, 1]
                        else:
                            probs = None
                        
                        # Results
                        results = raw_df[["PassengerId"]].copy() if "PassengerId" in raw_df.columns else pd.DataFrame()
                        if results.empty:
                            results["Index"] = raw_df.index

                        results["Transported"] = predictions
                        results["Transported"] = results["Transported"].astype(bool)
                        
                        if probs is not None:
                            results["Probability"] = probs

                        st.success("✅ Prediction Complete!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = px.pie(results, names="Transported", title="Transported Distribution", hole=0.4)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if probs is not None:
                            with col2:
                                fig2 = px.histogram(results, x="Probability", title="Probability Distribution")
                                st.plotly_chart(fig2, use_container_width=True)

                        # Download
                        submission_df = results[["PassengerId", "Transported"]] if "PassengerId" in results.columns else results
                        csv = submission_df.to_csv(index=False).encode('utf-8')
                        
                        st.download_button(
                            label="📥 Download Submission CSV",
                            data=csv,
                            file_name="submission.csv",
                            mime="text/csv",
                        )
                        
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                        with st.expander("Technical Error Details"):
                            st.write(e)
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # --- TAB 2: Manual Input ---
    with tab2:
        st.subheader("Single Passenger Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            home_planet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
            cryo_sleep = st.selectbox("CryoSleep", [False, True])
            destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
            age = st.number_input("Age", 0, 100, 25)

        with col2:
            vip = st.selectbox("VIP", [False, True])
            # Cabin Input (Simplified)
            deck = st.selectbox("Deck", ["A", "B", "C", "D", "E", "F", "G", "T"])
            num = st.number_input("Cabin Num", 0, 2000, 100)
            side = st.selectbox("Side", ["P", "S"])
            # Reconstruct Cabin string for compatibility or pass directly
            cabin_str = f"{deck}/{num}/{side}"

        with col3:
            room_service = st.number_input("RoomService ($)", 0.0, 10000.0, 0.0)
            food_court = st.number_input("FoodCourt ($)", 0.0, 10000.0, 0.0)
            shopping_mall = st.number_input("ShoppingMall ($)", 0.0, 10000.0, 0.0)
            spa = st.number_input("Spa ($)", 0.0, 10000.0, 0.0)
            vr_deck = st.number_input("VRDeck ($)", 0.0, 10000.0, 0.0)

        if st.button("Predict Passenger"):
            # Create DataFrame
            input_data = pd.DataFrame({
                "HomePlanet": [home_planet],
                "CryoSleep": [cryo_sleep],
                "Cabin": [cabin_str],
                "Destination": [destination],
                "Age": [age],
                "VIP": [vip],
                "RoomService": [room_service],
                "FoodCourt": [food_court],
                "ShoppingMall": [shopping_mall],
                "Spa": [spa],
                "VRDeck": [vr_deck],
                # Add dummy cols that might be expected
                "PassengerId": ["Dummy_ID"],
                "Name": ["Dummy Name"]
            })
            
            # Preprocess
            processed_input = custom_preprocessing(input_data)
            
            # Drop ID
            if "PassengerId" in processed_input.columns:
                X_single = processed_input.drop(columns=["PassengerId"])
            else:
                X_single = processed_input

            # Predict
            try:
                pred = model.predict(X_single)[0]
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X_single)[0][1]
                else:
                    prob = 0.0
                
                st.divider()
                c1, c2 = st.columns(2)
                
                c1.metric("Transport Probability", f"{prob:.2%}")
                
                if pred: # True
                    c2.error("Result: TRANSPORTED 🌌")
                else:
                    c2.success("Result: SAFE (Not Transported) ✅")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.warning("Model pipeline mismatch. Ensure the notebook training columns match these inputs.")

if __name__ == "__main__":
    main()