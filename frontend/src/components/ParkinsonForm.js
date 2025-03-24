import React, { useState } from "react";

const ParkinsonForm = () => {
  const initialData = {
    MDVP_Fo_Hz: "",
    MDVP_Fhi_Hz: "",
    MDVP_Flo_Hz: "",
    MDVP_Jitter_percent: "",
    MDVP_Jitter_Abs: "",
    MDVP_RAP: "",
    MDVP_PPQ: "",
    Jitter_DDP: "",
    MDVP_Shimmer: "",
    MDVP_Shimmer_dB: "",
    Shimmer_APQ3: "",
    Shimmer_APQ5: "",
    MDVP_APQ: "",
    Shimmer_DDA: "",
    NHR: "",
    HNR: "",
    RPDE: "",
    DFA: "",
    spread1: "",
    spread2: "",
    D2: "",
    PPE: ""
  };

  const [formData, setFormData] = useState(initialData);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict/parkinsons", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          features: Object.values(formData).map(Number),
        }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error:", error);
      setResult({ error: "Failed to connect to the server." });
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData(initialData);
    setResult(null);
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.heading}>Parkinson's Disease Prediction</h2>
      <form onSubmit={handleSubmit} style={styles.form}>
        {Object.keys(formData).map((key, index) => (
          <div key={index} style={styles.inputGroup}>
            <label style={styles.label}>
              {key.replace(/_/g, " ")}:
              <input
                type="number"
                name={key}
                value={formData[key]}
                onChange={handleChange}
                required
                style={styles.input}
              />
            </label>
          </div>
        ))}
        <div style={styles.buttonGroup}>
          <button type="submit" style={styles.button} disabled={loading}>
            {loading ? "Predicting..." : "Predict"}
          </button>
          <button
            type="button"
            style={styles.resetButton}
            onClick={handleReset}
          >
            Reset
          </button>
        </div>
      </form>

      {result && (
        <div style={result.result === "Parkinson's Detected" ? styles.resultNegative : styles.resultPositive}>
          {result.error ? (
            <p style={styles.error}>{result.error}</p>
          ) : (
            <>
              <h3>{result.result}</h3>
              <p>Confidence: {(result.confidence * 100).toFixed(2)}%</p>
            </>
          )}
        </div>
      )}
    </div>
  );
};

const styles = {
  container: {
    width: "60%",
    margin: "20px auto",
    padding: "20px",
    backgroundColor: "#f8f9fa",
    borderRadius: "10px",
    boxShadow: "0px 0px 15px rgba(0,0,0,0.1)",
  },
  heading: {
    color: "#343a40",
    fontSize: "28px",
    textAlign: "center",
    marginBottom: "20px",
  },
  form: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
  },
  inputGroup: {
    width: "100%",
    marginBottom: "15px",
  },
  label: {
    fontSize: "14px",
    fontWeight: "bold",
    display: "block",
    marginBottom: "5px",
  },
  input: {
    padding: "10px",
    fontSize: "14px",
    borderRadius: "5px",
    border: "1px solid #ced4da",
    width: "100%",
  },
  buttonGroup: {
    display: "flex",
    justifyContent: "center",
    gap: "15px",
    marginTop: "15px",
  },
  button: {
    padding: "10px 20px",
    fontSize: "16px",
    backgroundColor: "#007bff",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
  resetButton: {
    padding: "10px 20px",
    fontSize: "16px",
    backgroundColor: "#6c757d",
    color: "white",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
  resultPositive: {
    marginTop: "20px",
    padding: "15px",
    backgroundColor: "#d4edda",
    borderRadius: "5px",
    textAlign: "center",
    border: "1px solid #c3e6cb",
  },
  resultNegative: {
    marginTop: "20px",
    padding: "15px",
    backgroundColor: "#f8d7da",
    borderRadius: "5px",
    textAlign: "center",
    border: "1px solid #f5c6cb",
  },
  error: {
    color: "#721c24",
  },
};

export default ParkinsonForm;



