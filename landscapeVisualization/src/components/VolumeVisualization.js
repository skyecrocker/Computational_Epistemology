import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';

const VolumeVisualization = ({ jsonFilePath = '/volume_data.json' }) => {
  const mountRef = useRef(null);
  const [renderMode, setRenderMode] = useState('volume');
  const [slicePosition, setSlicePosition] = useState(2);
  const [sliceAxis, setSliceAxis] = useState('z');
  const [threshold, setThreshold] = useState(0.3);
  const [volumeData, setVolumeData] = useState(null);
  const [size, setSize] = useState(5);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Load NumPy data from JSON file
  useEffect(() => {
    fetch(jsonFilePath)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Failed to load data: ${response.statusText}`);
        }
        return response.json();
      })
      .then(result => {
        setVolumeData(result.data);
        setSize(result.dimensions[0]); // Assuming cubic data
        setSlicePosition(Math.floor(result.dimensions[0]/2)); // Set initial slice to middle
        setLoading(false);
      })
      .catch(err => {
        console.error("Error loading volume data:", err);
        setError(err.message);
        setLoading(false);
      });
  }, [jsonFilePath]);

  // Three.js visualization
  useEffect(() => {
    // Only set up the scene once data is loaded
    if (loading || error || !volumeData) return;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);
    
    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      (mountRef.current?.clientWidth || window.innerWidth) / 
      (mountRef.current?.clientHeight || window.innerHeight),
      0.1,
      1000
    );
    camera.position.set(size * 2, size * 2, size * 2);
    camera.lookAt(0, 0, 0);
    
    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(
      mountRef.current?.clientWidth || window.innerWidth, 
      mountRef.current?.clientHeight || window.innerHeight
    );
    
    if (mountRef.current) {
      // Clear any existing canvas
      while (mountRef.current.firstChild) {
        mountRef.current.removeChild(mountRef.current.firstChild);
      }
      mountRef.current.appendChild(renderer.domElement);
    }
    
    // Controls variables
    let isMouseDown = false;
    let previousMousePosition = { x: 0, y: 0 };
    
    // Grid spacing
    const spacing = 2;
    
    // Color mapping function
    const getColor = (value) => {
      // Color gradient from blue (low) to red (high)
      return new THREE.Color(value, 0.2, 1 - value);
    };
    
    // Group to hold visualization
    const visualizationGroup = new THREE.Group();
    scene.add(visualizationGroup);
    
    // Function to create visualization from volumeData
    const createVisualization = () => {
      const group = new THREE.Group();
      
      for (let x = 0; x < size; x++) {
        for (let y = 0; y < size; y++) {
          for (let z = 0; z < size; z++) {
            const value = volumeData[x][y][z];
            
            // Skip points below threshold in volume mode
            if (renderMode === 'volume' && value < threshold) continue;
            
            // Skip points not on the current slice in slice mode
            if (renderMode === 'slice') {
              if (sliceAxis === 'x' && x !== slicePosition) continue;
              if (sliceAxis === 'y' && y !== slicePosition) continue;
              if (sliceAxis === 'z' && z !== slicePosition) continue;
            }
            
            // Create cube geometry - size proportional to value
            const cubeSize = spacing * 0.8 * value;
            const geometry = new THREE.BoxGeometry(cubeSize, cubeSize, cubeSize);
            
            // Create material with color based on value
            const color = getColor(value);
            const material = new THREE.MeshLambertMaterial({
              color: color,
              opacity: Math.max(0.2, value),
              transparent: true
            });
            
            // Create and position cube
            const cube = new THREE.Mesh(geometry, material);
            cube.position.set(
              (x - size / 2) * spacing,
              (y - size / 2) * spacing,
              (z - size / 2) * spacing
            );
            
            group.add(cube);
          }
        }
      }
      
      return group;
    };
    
    // Create initial visualization
    let visualization = createVisualization();
    visualizationGroup.add(visualization);
    
    // Add grid helper
    const gridSize = size * spacing;
    const gridHelper = new THREE.GridHelper(gridSize, size);
    scene.add(gridHelper);
    
    // Add axes helper
    const axesHelper = new THREE.AxesHelper(gridSize / 2);
    scene.add(axesHelper);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.7);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);
    
    // Function to update visualization when controls change
    const updateVisualization = () => {
      visualizationGroup.remove(visualization);
      visualization = createVisualization();
      visualizationGroup.add(visualization);
    };
    
    // Mouse controls for rotation
    const onMouseDown = (event) => {
      isMouseDown = true;
      previousMousePosition = {
        x: event.clientX,
        y: event.clientY
      };
    };
    
    const onMouseUp = () => {
      isMouseDown = false;
    };
    
    const onMouseMove = (event) => {
      if (!isMouseDown) return;
      
      const deltaMove = {
        x: event.clientX - previousMousePosition.x,
        y: event.clientY - previousMousePosition.y
      };
      
      // Rotation speed
      const rotationSpeed = 0.005;
      
      // Compute rotation
      const radianY = deltaMove.x * rotationSpeed;
      const radianX = deltaMove.y * rotationSpeed;
      
      // Pivot point at center
      const pivot = new THREE.Vector3(0, 0, 0);
      
      // Rotate camera around pivot
      const cameraOffset = camera.position.clone().sub(pivot);
      cameraOffset.applyAxisAngle(new THREE.Vector3(0, 1, 0), radianY);
      cameraOffset.applyAxisAngle(new THREE.Vector3(1, 0, 0), radianX);
      camera.position.copy(pivot).add(cameraOffset);
      camera.lookAt(pivot);
      
      previousMousePosition = {
        x: event.clientX,
        y: event.clientY
      };
    };
    
    // Mouse wheel zoom
    const onWheel = (event) => {
      event.preventDefault();
      const zoomSpeed = 0.1;
      const direction = event.deltaY > 0 ? 1 : -1;
      
      // Direction vector from camera to center
      const vector = new THREE.Vector3(0, 0, 0).sub(camera.position).normalize();
      
      // Move camera along vector
      camera.position.addScaledVector(vector, direction * zoomSpeed * camera.position.length());
    };
    
    // Add event listeners
    renderer.domElement.addEventListener('mousedown', onMouseDown);
    renderer.domElement.addEventListener('mouseup', onMouseUp);
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('wheel', onWheel);
    
    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    };
    
    animate();
    
    // Handle window resize
    const handleResize = () => {
      const width = mountRef.current?.clientWidth || window.innerWidth;
      const height = mountRef.current?.clientHeight || window.innerHeight;
      
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    
    window.addEventListener('resize', handleResize);
    
    // Update visualization when controls change
    updateVisualization();
    
    // Cleanup
    return () => {
      window.removeEventListener('resize', handleResize);
      renderer.domElement.removeEventListener('mousedown', onMouseDown);
      renderer.domElement.removeEventListener('mouseup', onMouseUp);
      renderer.domElement.removeEventListener('mousemove', onMouseMove);
      renderer.domElement.removeEventListener('wheel', onWheel);
      
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, [volumeData, renderMode, slicePosition, sliceAxis, threshold, size, loading, error]);

  if (loading) {
    return <div>Loading volume data...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div 
        ref={mountRef} 
        style={{ 
          width: '100%', 
          height: '500px',
          backgroundColor: '#f0f0f0',
          borderRadius: '8px'
        }}
      />
      
      {/* Controls panel */}
      <div style={{ 
        position: 'absolute', 
        top: '10px', 
        right: '10px', 
        background: 'rgba(255,255,255,0.8)',
        padding: '15px',
        borderRadius: '8px',
        boxShadow: '0 2px 10px rgba(0,0,0,0.1)',
        maxWidth: '300px'
      }}>
        <h3 style={{ margin: '0 0 10px 0' }}>NumPy Volume Controls</h3>
        
        <div style={{ marginBottom: '15px' }}>
          <label style={{ display: 'block', marginBottom: '5px' }}>Display Mode:</label>
          <select 
            value={renderMode}
            onChange={(e) => setRenderMode(e.target.value)}
            style={{ width: '100%', padding: '5px' }}
          >
            <option value="volume">Full Volume</option>
            <option value="slice">Slice View</option>
          </select>
        </div>
        
        {renderMode === 'slice' && (
          <>
            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px' }}>Slice Axis:</label>
              <select 
                value={sliceAxis}
                onChange={(e) => setSliceAxis(e.target.value)}
                style={{ width: '100%', padding: '5px' }}
              >
                <option value="x">X Axis</option>
                <option value="y">Y Axis</option>
                <option value="z">Z Axis</option>
              </select>
            </div>
            
            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', marginBottom: '5px' }}>
                Slice Position: {slicePosition}
              </label>
              <input 
                type="range" 
                min="0" 
                max={size - 1} 
                value={slicePosition}
                onChange={(e) => setSlicePosition(parseInt(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
          </>
        )}
        
        {renderMode === 'volume' && (
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', marginBottom: '5px' }}>
              Value Threshold: {threshold.toFixed(2)}
            </label>
            <input 
              type="range" 
              min="0" 
              max="1" 
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>
        )}
        
        <div style={{ marginTop: '20px' }}>
          <div style={{ fontSize: '14px', marginBottom: '10px' }}>Color Legend:</div>
          <div style={{ 
            height: '20px', 
            width: '100%', 
            background: 'linear-gradient(to right, blue, purple, red)',
            marginBottom: '5px',
            borderRadius: '4px'
          }}></div>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            fontSize: '12px'
          }}>
            <span>0.0</span>
            <span>0.5</span>
            <span>1.0</span>
          </div>
        </div>
        
        <div style={{ marginTop: '20px', fontSize: '14px' }}>
          <p style={{ margin: '0 0 5px 0' }}><strong>Navigation:</strong></p>
          <p style={{ margin: '0 0 5px 0' }}>- Rotate: Click and drag</p>
          <p style={{ margin: '0 0 5px 0' }}>- Zoom: Scroll wheel</p>
        </div>
      </div>
    </div>
  );
};

export default VolumeVisualization;