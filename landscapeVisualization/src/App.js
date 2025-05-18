import React, { useState } from 'react';
import './App.css';
import VolumeVisualization from './components/VolumeVisualization';

function App() {
  const [apiUrl, setApiUrl] = useState('/grid15_sigma1.json');

  const handleApiChange = (event) => {
    setApiUrl(event.target.value);
  };

  return (
    <div className="App">
      <h1>3D Landscape Visualization</h1>
      <label htmlFor="api-select">Choose a dataset: </label>
      <select id="api-select" value={apiUrl} onChange={handleApiChange}>
        <option value="/grid15_sigma1.json">Grid 15 Sigma 1</option>
        <option value="/grid15_sigma2.json">Grid 15 Sigma 2</option>
        <option value="/grid15_sigma3.json">Grid 15 Sigma 3</option>
        <option value="/grid25_sigma1.json">Grid 25 Sigma 1</option>
        <option value="/grid25_sigma2.json">Grid 25 Sigma 2</option>
        <option value="/grid25_sigma3.json">Grid 25 Sigma 3</option>
        <option value="/grid41_sigma1.json">Grid 41 Sigma 1</option>
        <option value="/grid41_sigma2.json">Grid 41 Sigma 2</option>
        <option value="/grid41_sigma3.json">Grid 41 Sigma 3</option>
      </select>
      <VolumeVisualization jsonFilePath={apiUrl} />
    </div>
  );
}

export default App;