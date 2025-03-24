import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./components/dashboard";
import DiabetesForm from "./components/DiabetesForm";
import CancerForm from "./components/CancerForm";
import ParkinsonForm from "./components/ParkinsonForm";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/diabetes" element={<DiabetesForm />} />
        <Route path="/cancer" element={<CancerForm />} />
        <Route path="/parkinson" element={<ParkinsonForm />} />
      </Routes>
    </Router>
  );
}

export default App;


