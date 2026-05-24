import { useRef, useMemo, useState, useEffect } from 'react'
import { useFrame, useLoader } from '@react-three/fiber'
import * as THREE from 'three'

// Custom loader for our campus model
function CampusModel({ onSelectLocation }) {
  const groupRef = useRef()
  const [modelData, setModelData] = useState(null)

  // Load the campus model from JSON
  useEffect(() => {
    fetch('/models/campus_model.json')
      .then(res => res.json())
      .then(data => {
        setModelData(data)
        console.log('Campus model loaded:', data)
      })
      .catch(err => console.error('Error loading model:', err))
  }, [])
  
  useFrame((state) => {
    if (groupRef.current) {
      // Subtle rotation
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.05) * 0.02
    }
  })
  
  if (!modelData || !modelData.pointCloud) {
    return null
  }
  
  // Flatten the positions and colors arrays
  const positionsFlat = modelData.pointCloud.positions.flat()
  const colorsFlat = modelData.pointCloud.colors.flat()
  
  return (
    <group ref={groupRef}>
      <points>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={positionsFlat.length / 3}
            array={new Float32Array(positionsFlat)}
            itemSize={3}
          />
          <bufferAttribute
            attach="attributes-color"
            count={colorsFlat.length / 3}
            array={new Float32Array(colorsFlat)}
            itemSize={3}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.15}
          vertexColors
          transparent
          opacity={0.9}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          sizeAttenuation={true}
          color="#00ffcc"
        />
      </points>
    </group>
  )
}

export default function CampusPointCloud({ locations, onSelectLocation }) {
  return (
    <group>
      {/* Load actual campus model from DXF */}
      <CampusModel onSelectLocation={onSelectLocation} />
      
      {/* Location Markers for CIT Campus */}
      {locations.map((loc, index) => (
        <LocationMarker
          key={loc.id}
          position={loc.position}
          location={loc}
          onSelect={onSelectLocation}
        />
      ))}
      
      {/* Ground Grid */}
      <gridHelper args={[100, 20, 0x00ffcc, 0x003333]} position={[0, -5, 0]} />
    </group>
  )
}

function LocationMarker({ position, location, onSelect }) {
  const markerRef = useRef()
  
  useFrame((state) => {
    if (markerRef.current) {
      markerRef.current.position.y = Math.sin(state.clock.elapsedTime * 3) * 0.5 + 2
      markerRef.current.rotation.y += 0.02
    }
  })
  
  return (
    <group 
      position={position}
      onClick={(e) => {
        e.stopPropagation()
        onSelect(location)
      }}
    >
      {/* Marker Cone */}
      <mesh ref={markerRef} rotation={[0, 0, 0]}>
        <coneGeometry args={[1, 3, 4]} />
        <meshStandardMaterial
          color={location.type === 'building' ? '#00ffcc' : '#00ccff'}
          emissive={location.type === 'building' ? '#00ffcc' : '#00ccff'}
          emissiveIntensity={2}
          transparent
          opacity={0.9}
        />
      </mesh>
      
      {/* Glow Sphere */}
      <mesh>
        <sphereGeometry args={[0.5, 16, 16]} />
        <meshBasicMaterial
          color="#ffffff"
          transparent
          opacity={0.7}
        />
      </mesh>
      
      {/* Vertical line to ground */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, 0, 0, 3, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#00ffcc" transparent opacity={0.5} />
      </line>
    </group>
  )
}
