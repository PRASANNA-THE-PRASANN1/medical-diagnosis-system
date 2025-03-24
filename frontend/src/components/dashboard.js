import React from "react";
import { useNavigate } from "react-router-dom";
import { 
  Card, CardContent, CardMedia, Typography, Grid, Button, Container, 
  Paper 
} from "@mui/material";
import { FaHeartbeat, FaLungs, FaBrain} from "react-icons/fa";

const diseaseModels = [
  { 
    name: "Diabetes Prediction", 
    path: "/diabetes", 
    icon: <FaHeartbeat size={40} color="red" />,
    description: "Predict diabetes risk based on clinical data"
  },
  { 
    name: "Cancer Prediction", 
    path: "/cancer", 
    icon: <FaLungs size={40} color="blue" />,
    description: "Analyze risk factors for lung cancer"
  },
  { 
    name: "Parkinson's Prediction", 
    path: "/parkinson", 
    icon: <FaBrain size={40} color="purple" />,
    description: "Assess neurological indicators for Parkinson's disease"
  },
];

function Dashboard() {
  const navigate = useNavigate();

  return (
    <Container maxWidth="md" sx={{ my: 4 }}>
      <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
        <Typography variant="h3" gutterBottom sx={{ fontWeight: 600 }}>
          Medical Diagnosis System
        </Typography>
        <Typography variant="subtitle1" color="textSecondary" gutterBottom>
          Select a prediction model to analyze patient data and get diagnostic insights
        </Typography>

        <Grid container spacing={4} justifyContent="center" sx={{ mt: 2 }}>
          {diseaseModels.map((model, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
              <Card
                sx={{ 
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  cursor: "pointer", 
                  transition: "0.3s",
                  "&:hover": { 
                    boxShadow: 6,
                    transform: 'translateY(-5px)' 
                  } 
                }}
              >
                <CardMedia
                  sx={{ 
                    pt: 2,
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center'
                  }}
                >
                  {model.icon}
                </CardMedia>
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h6" gutterBottom>
                    {model.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {model.description}
                  </Typography>
                  <Button 
                    variant="contained" 
                    color="primary" 
                    fullWidth
                    onClick={() => navigate(model.path)}
                  >
                    Start Analysis
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Paper>
    </Container>
  );
}

export default Dashboard;

// 3, 130, 70, 25, 100, 26.5, 0.7, 50