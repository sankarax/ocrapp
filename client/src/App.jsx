import { useState } from 'react'
import reactLogo from './assets/react.svg'
import pytorchLogo from '/pytorch-icon.svg'
import './App.css'
import axios from "axios";
function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  
  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };
  
  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error uploading file:", error);
    }
  };

  return (
    <>
      <div>
        <a href="https://pytorch.org/docs/stable/index.html" target="_blank">
          <img src={pytorchLogo} className="logo" alt="Vite logo" />
        </a>
        <a href="https://react.dev" target="_blank">
          <img src={reactLogo} className="logo react" alt="React logo" />
        </a>
      </div>
      <h1>Handwriting Detector</h1>
      <div className="card">
      <input type="file" onChange={handleFileChange} style={{ margin: "10px 0" }} />
        <button onClick={handleUpload} style={{ padding: "10px 20px", cursor: "pointer" }}>
          Predict
        </button>
        <p>
        {prediction && <p style={{ marginTop: "20px", fontSize: "18px" }}>Prediction: {prediction}</p>}
        </p>
      </div>
      
    </>
  )
}

export default App
