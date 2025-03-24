import React, { useState, useEffect } from "react";
import axios from "axios";

const CancerForm = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [patientInfo, setPatientInfo] = useState({
    patientId: "",
    age: "",
    gender: "",
    smoker: "unknown",
    familyHistory: "unknown",
  });

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    processFile(file);
  };

  // Process the selected file
  const processFile = (file) => {
    if (file) {
      // Check file type
      if (!file.type.startsWith("image/")) {
        setError("Invalid file type. Please upload an image (PNG or JPEG).");
        return;
      }

      // Check file size (limit to 10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError("File is too large. Please upload an image smaller than 10MB.");
        return;
      }

      setSelectedImage(file);
      setPrediction(null);
      setError("");

      // Generate image preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Handle drag events
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  // Handle drop event
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  // Handle patient info changes
  const handlePatientInfoChange = (e) => {
    const { name, value } = e.target;
    setPatientInfo({
      ...patientInfo,
      [name]: value,
    });
  };

  // Handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();

    if (!selectedImage) {
      setError("Please upload a CT scan image before submitting.");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedImage);

    // Add patient information to the form data
    Object.keys(patientInfo).forEach((key) => {
      formData.append(key, patientInfo[key]);
    });

    try {
      setLoading(true);
      setError("");

      // Send image to the backend API for prediction
      const response = await axios.post(
        "http://127.0.0.1:5000/predict/cancer",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setPrediction(response.data);
    } catch (err) {
      setError(
        err.response?.data?.message ||
          "Error occurred while processing the image. Please try again."
      );
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Clear form data
  const handleClear = () => {
    setSelectedImage(null);
    setImagePreview(null);
    setPrediction(null);
    setError("");
    setPatientInfo({
      patientId: "",
      age: "",
      gender: "",
      smoker: "unknown",
      familyHistory: "unknown",
    });
  };

  // ✅ Correctly placed useEffect inside the component
  useEffect(() => {
    const styleSheet = document.createElement("style");
    styleSheet.innerHTML = `
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }

      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
      }

      .form-container:hover {
        transform: translateY(-5px);
        box-shadow: 0px 10px 25px rgba(0, 0, 0, 0.1);
      }

      .predict-button:hover:not(:disabled) {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
      }

      .clear-button:hover {
        background-color: #f8fafc;
        color: #3498db;
        border-color: #3498db;
      }

      input:focus, select:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
      }

      .result-container:hover {
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.08);
      }

      .imagePreviewContainer:hover .previewOverlay {
        opacity: 1;
      }
    `;
    document.head.appendChild(styleSheet);

    return () => {
      document.head.removeChild(styleSheet);
    };
  }, []);

  return (
    <div style={styles.container} className="form-container">
      <h2 style={styles.heading}>Lung Cancer Diagnosis Assistant</h2>
      <p style={styles.subtitle}>Upload a CT scan image for analysis</p>

      <form onSubmit={handleSubmit} style={styles.form}>
        {/* Patient Information Section */}
        <div style={styles.patientInfoContainer}>
          <h3 style={styles.sectionTitle}>Patient Information</h3>

          <div style={styles.formRow}>
            <div style={styles.formGroup}>
              <label style={styles.label}>Patient ID:</label>
              <input
                type="text"
                name="patientId"
                value={patientInfo.patientId}
                onChange={handlePatientInfoChange}
                style={styles.input}
                placeholder="Enter patient ID"
              />
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Age:</label>
              <input
                type="number"
                name="age"
                value={patientInfo.age}
                onChange={handlePatientInfoChange}
                style={styles.input}
                placeholder="Age"
                min="0"
                max="120"
              />
            </div>
          </div>

          <div style={styles.formRow}>
            <div style={styles.formGroup}>
              <label style={styles.label}>Gender:</label>
              <select
                name="gender"
                value={patientInfo.gender}
                onChange={handlePatientInfoChange}
                style={styles.select}
              >
                <option value="">Select gender</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>

            <div style={styles.formGroup}>
              <label style={styles.label}>Smoker:</label>
              <select
                name="smoker"
                value={patientInfo.smoker}
                onChange={handlePatientInfoChange}
                style={styles.select}
              >
                <option value="unknown">Unknown</option>
                <option value="yes">Yes</option>
                <option value="former">Former</option>
                <option value="no">No</option>
              </select>
            </div>
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>Family History of Cancer:</label>
            <select
              name="familyHistory"
              value={patientInfo.familyHistory}
              onChange={handlePatientInfoChange}
              style={styles.select}
            >
              <option value="unknown">Unknown</option>
              <option value="yes">Yes</option>
              <option value="no">No</option>
            </select>
          </div>
        </div>

        {/* Image Upload Section */}
        <div
          style={{
            ...styles.dropzone,
            ...(dragActive ? styles.dropzoneActive : {}),
            ...(imagePreview ? styles.dropzoneWithPreview : {}),
          }}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="image/png, image/jpeg"
            onChange={handleFileChange}
            style={styles.fileInput}
            id="file-upload"
          />

          {imagePreview ? (
            <div style={styles.imagePreviewContainer}>
              <img
                src={imagePreview}
                alt="CT Scan Preview"
                style={styles.imagePreview}
              />
              <div style={styles.previewOverlay}>
                <span>Click or drag to replace</span>
              </div>
            </div>
          ) : (
            <div style={styles.dropzoneContent}>
              <p>Drag and drop a CT scan image here, or click to select</p>
              <p style={styles.fileTypeInfo}>Accepts PNG, JPEG (max 10MB)</p>
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && <p style={styles.error}>{error}</p>}

        {/* Button Group */}
        <div style={styles.buttonGroup}>
          <button
            type="button"
            onClick={handleClear}
            style={styles.clearButton}
            className="clear-button"
            disabled={loading}
          >
            Clear Form
          </button>

          <button
            type="submit"
            disabled={loading || !selectedImage}
            style={{
              ...styles.button,
              ...((!selectedImage || loading) ? styles.buttonDisabled : {}),
            }}
            className="predict-button"
          >
            {loading ? (
              <div style={styles.loadingSpinner}>
                <div style={styles.spinner}></div>
                <span>Processing...</span>
              </div>
            ) : (
              "Analyze CT Scan"
            )}
          </button>
        </div>
      </form>

      {/* Prediction Results */}
      {prediction && (
        <div style={styles.resultContainer} className="result-container">
          <h3 style={styles.resultTitle}>Analysis Results</h3>

          <div style={styles.resultContent}>
            <div style={styles.resultItem}>
              <span style={styles.resultLabel}>Diagnosis:</span>
              <span
                style={{
                  ...styles.resultValue,
                  color:
                    prediction.result === "Malignant" ? "#d32f2f" : "#2e7d32",
                }}
              >
                {prediction.result}
              </span>
            </div>

            <div style={styles.resultItem}>
              <span style={styles.resultLabel}>Confidence:</span>
              <div style={styles.confidenceBar}>
                <div
                  style={{
                    ...styles.confidenceFill,
                    width: `${Math.round(prediction.confidence * 100)}%`,
                    backgroundColor:
                      prediction.result === "Malignant" ? "#d32f2f" : "#2e7d32",
                  }}
                ></div>
                <span style={styles.confidenceText}>
                  {Math.round(prediction.confidence * 100)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Inline CSS for styling
const styles = {
  container: {
    maxWidth: "800px",
    margin: "40px auto",
    padding: "30px",
    borderRadius: "12px",
    boxShadow: "0px 4px 20px rgba(0, 0, 0, 0.1)",
    backgroundColor: "#fff",
  },
  heading: {
    color: "#2c3e50",
    marginBottom: "5px",
    fontSize: "28px",
    textAlign: "center",
  },
  subtitle: {
    color: "#7f8c8d",
    marginBottom: "20px",
    textAlign: "center",
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "25px",
  },
  patientInfoContainer: {
    padding: "15px",
    backgroundColor: "#f8fafc",
    borderRadius: "8px",
  },
  formRow: {
    display: "flex",
    gap: "15px",
    marginBottom: "15px",
  },
  formGroup: {
    flex: "1",
    display: "flex",
    flexDirection: "column",
  },
  label: {
    fontSize: "14px",
    fontWeight: "500",
    marginBottom: "5px",
    color: "#34495e",
  },
  input: {
    padding: "10px 12px",
    borderRadius: "6px",
    border: "1px solid #dcdfe6",
    fontSize: "14px",
  },
  select: {
    padding: "10px 12px",
    borderRadius: "6px",
    border: "1px solid #dcdfe6",
    fontSize: "14px",
    backgroundColor: "white",
    cursor: "pointer",
  },
  buttonGroup: {
    display: "flex",
    justifyContent: "space-between",
    marginTop: "10px",
    gap: "15px",
  },
  button: {
    flex: "2",
    backgroundColor: "#3498db",
    color: "white",
    padding: "12px 20px",
    border: "none",
    borderRadius: "6px",
    cursor: "pointer",
    fontSize: "16px",
    fontWeight: "500",
  },
  buttonDisabled: {
    backgroundColor: "#a0aec0",
    cursor: "not-allowed",
  },
  clearButton: {
    flex: "1",
    backgroundColor: "transparent",
    color: "#64748b",
    padding: "12px 20px",
    border: "1px solid #cbd5e1",
    borderRadius: "6px",
    cursor: "pointer",
    fontSize: "16px",
  },
};

export default CancerForm;




