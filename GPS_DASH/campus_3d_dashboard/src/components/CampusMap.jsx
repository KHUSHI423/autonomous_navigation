import { useRef, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'
import CampusModel from './CampusModel'

// Campus location markers - adjusted for actual campus bounds
const LOCATIONS = [
  { id: 1, name: 'Main Gate', position: [-20, 0, -40], type: 'entrance' },
  { id: 2, name: 'Engineering Block', position: [-15, 0, -20], type: 'building' },
  { id: 3, name: 'Library Block', position: [20, 0, 15], type: 'building' },
  { id: 4, name: 'Auditorium', position: [-10, 0, 10], type: 'building' },
  { id: 5, name: 'Cafeteria', position: [25, 0, -15], type: 'facility' },
  { id: 6, name: 'Sports Complex', position: [30, 0, 25], type: 'facility' },
  { id: 7, name: 'Admin Office', position: [0, 0, 0], type: 'building' },
]

export default function CampusMap({ onSelectLocation }) {
  return (
    <group>
      {/* Load actual campus point cloud from DXF */}
      <CampusModel onSelectLocation={onSelectLocation} />

      {/* Location markers */}
      {LOCATIONS.map((loc) => (
        <LocationMarker
          key={loc.id}
          position={loc.position}
          location={loc}
          onSelect={onSelectLocation}
        />
      ))}
    </group>
  )
}

function LocationMarker({ position, location, onSelect }) {
  const markerRef = useRef()
  const [hovered, setHovered] = useState(false)

  useFrame((state) => {
    if (markerRef.current) {
      // Gentle floating animation
      markerRef.current.position.y = Math.sin(state.clock.elapsedTime * 2) * 0.3 + 3
      markerRef.current.rotation.y += 0.01
    }
  })

  return (
    <group
      position={position}
      onClick={(e) => {
        e.stopPropagation()
        onSelect(location)
      }}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      {/* Cone marker pointing up */}
      <mesh ref={markerRef}>
        <coneGeometry args={[1, 3, 4]} />
        <meshStandardMaterial
          color={hovered ? '#ffffff' : (location.type === 'building' ? '#00ffcc' : '#00ccff')}
          emissive={location.type === 'building' ? '#00ffcc' : '#00ccff'}
          emissiveIntensity={hovered ? 2 : 1}
          transparent
          opacity={0.9}
        />
      </mesh>

      {/* Small sphere at base */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.4, 12, 12]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.5} />
      </mesh>

      {/* Vertical line to ground */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, 0, 0, -3, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#00ffcc" transparent opacity={0.4} />
      </line>
    </group>
  )
}
