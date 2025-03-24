import React, { useState } from "react";
import axios from "axios";
import { 
  TextField, Button, Typography, CircularProgress, Box, Container, Paper,
  Alert, Divider, Stepper, Step, StepLabel, Grid
} from "@mui/material";
import { useNavigate } from "react-router-dom";

function DiabetesForm() {
  const [features, setFeatures] = useState(Array(8).fill(""));
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeStep, setActiveStep] = useState(0);
  const navigate = useNavigate();

  // Input field metadata with guidelines, units and normal ranges
  const inputFields = [
    { 
      label: "Pregnancies", 
      placeholder: "Number of pregnancies",
      helperText: "Enter the number of times pregnant",
      min: 0, 
      max: 17,
      unit: ""
    },
    { 
      label: "Glucose", 
      placeholder: "Plasma glucose concentration",
      helperText: "After 2 hours in an oral glucose tolerance test",
      min: 0, 
      max: 200,
      unit: "mg/dL"
    },
    { 
      label: "Blood Pressure", 
      placeholder: "Diastolic blood pressure",
      helperText: "Measured in mm Hg",
      min: 0, 
      max: 122,
      unit: "mm Hg"
    },
    { 
      label: "Skin Thickness", 
      placeholder: "Triceps skin fold thickness",
      helperText: "Measured in mm",
      min: 0, 
      max: 99,
      unit: "mm"
    },
    { 
      label: "Insulin", 
      placeholder: "2-Hour serum insulin",
      helperText: "Measured in mu U/ml",
      min: 0, 
      max: 846,
      unit: "mu U/ml"
    },
    { 
      label: "BMI", 
      placeholder: "Body mass index",
      helperText: "Weight in kg/(height in m)²",
      min: 0, 
      max: 67.1,
      unit: "kg/m²"
    },
    { 
      label: "Diabetes Pedigree Function", 
      placeholder: "Diabetes pedigree function",
      helperText: "Scores likelihood based on family history",
      min: 0.078, 
      max: 2.42,
      unit: ""
    },
    { 
      label: "Age", 
      placeholder: "Age in years",
      helperText: "Patient's age",
      min: 21, 
      max: 81,
      unit: "years"
    }
  ];

  // Form validation
  const [validationErrors, setValidationErrors] = useState(Array(8).fill(""));

  const validateField = (index, value) => {
    const field = inputFields[index];
    let error = "";
    
    if (value === "") {
      error = "This field is required";
    } else if (isNaN(value)) {
      error = "Please enter a valid number";
    } else if (Number(value) < field.min) {
      error = `Value should be at least ${field.min}`;
    } else if (Number(value) > field.max) {
      error = `Value should be at most ${field.max}`;
    }
    
    return error;
  };

  const handleChange = (index, value) => {
    const newFeatures = [...features];
    newFeatures[index] = value;
    setFeatures(newFeatures);
    
    const newValidationErrors = [...validationErrors];
    newValidationErrors[index] = validateField(index, value);
    setValidationErrors(newValidationErrors);
  };

  const isFormValid = () => {
    return features.every((feature, index) => feature !== "" && !validationErrors[index]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict/diabetes", { features });
      setResult(response.data.prediction);
      setActiveStep(1); // Move to results step
    } catch (err) {
      setError("Failed to fetch prediction. Please try again.");
      console.error("Error details:", err);
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFeatures(Array(8).fill(""));
    setResult(null);
    setError("");
    setValidationErrors(Array(8).fill(""));
    setActiveStep(0);
  };

  return (
    <Container maxWidth="md">
      <Paper elevation={3} sx={{ padding: 4, marginY: 5, borderRadius: 2 }}>
        <Box display="flex" alignItems="center" mb={3}>
          <Button 
            variant="outlined" 
            size="small" 
            onClick={() => navigate("/")}
            sx={{ mr: 2 }}
          >
            Back
          </Button>
          <Typography variant="h4" sx={{ fontWeight: 500 }}>
            Diabetes Risk Assessment
          </Typography>
        </Box>

        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          <Step>
            <StepLabel>Enter Patient Data</StepLabel>
          </Step>
          <Step>
            <StepLabel>View Results</StepLabel>
          </Step>
        </Stepper>

        {activeStep === 0 ? (
          <>
            <Box sx={{ mb: 3 }}>
              <Alert severity="info">
                Please enter the patient's clinical data accurately. All fields are required for a valid prediction.
              </Alert>
            </Box>

            <form onSubmit={handleSubmit}>
              <Grid container spacing={3}>
                {inputFields.map((field, index) => (
                  <Grid item xs={12} sm={6} key={index}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <TextField
                        label={field.label}
                        placeholder={field.placeholder}
                        type="number"
                        variant="outlined"
                        fullWidth
                        value={features[index]}
                        onChange={(e) => handleChange(index, e.target.value)}
                        required
                        error={!!validationErrors[index]}
                        helperText={validationErrors[index] || field.helperText}
                        InputProps={{
                          endAdornment: field.unit && (
                            <Typography variant="body2" color="text.secondary" sx={{ ml: 1 }}>
                              {field.unit}
                            </Typography>
                          )
                        }}
                      />
                      <Button 
                        size="small" 
                        sx={{ ml: 1, minWidth: 30 }}
                        title={`Typical range: ${field.min} - ${field.max} ${field.unit}`}
                      >
                        ?
                      </Button>
                    </Box>
                  </Grid>
                ))}
              </Grid>

              <Box sx={{ mt: 4 }}>
                <Button 
                  type="submit" 
                  variant="contained" 
                  color="primary" 
                  size="large"
                  fullWidth 
                  disabled={loading || !isFormValid()}
                >
                  {loading ? <CircularProgress size={24} /> : "Predict Risk"}
                </Button>
              </Box>
            </form>

            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </>
        ) : (
          <Box sx={{ textAlign: "center" }}>
            <Box 
              sx={{ 
                p: 4, 
                borderRadius: 2, 
                bgcolor: result === 1 ? 'error.light' : 'success.light',
                mb: 3
              }}
            >
              <Typography variant="h4" gutterBottom>
                {result === 1 ? "High Risk" : "Low Risk"}
              </Typography>
              <Typography variant="h6">
                The patient is predicted to be {result === 1 ? "at risk for diabetes" : "not at risk for diabetes"}
              </Typography>
            </Box>

            <Divider sx={{ my: 3 }} />

            <Typography variant="body1" sx={{ mb: 3, textAlign: 'left' }}>
              {result === 1 ? 
                "This patient shows risk factors consistent with diabetes. Consider additional testing and lifestyle interventions." :
                "This patient's data suggests low diabetes risk at this time. Regular monitoring is still recommended, especially if there is family history of diabetes."
              }
            </Typography>

            <Typography variant="body2" color="text.secondary" sx={{ mb: 4, textAlign: 'left' }}>
              Note: This is a machine learning-based prediction and should not replace professional medical diagnosis. 
              Please consult with a healthcare provider for proper medical advice.
            </Typography>

            <Button 
              variant="outlined" 
              color="primary" 
              onClick={resetForm}
            >
              New Assessment
            </Button>
          </Box>
        )}
      </Paper>
    </Container>
  );
}

export default DiabetesForm;

