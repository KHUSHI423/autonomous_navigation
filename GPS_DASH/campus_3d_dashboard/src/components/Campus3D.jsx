import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

// Campus buildings - simplified layout matching the reference image
const BUILDINGS = [
  // Large square buildings (center-left) - main blocks
  { x: -25, z: -10, w: 20, d: 18, h: 5 },
  { x: 0, z: -10, w: 20, d: 18, h: 5 },
  
  // Right side buildings
  { x: 30, z: 0, w: 16, d: 14, h: 4 },
  { x: 30, z: 25, w: 16, d: 14, h: 4 },
  { x: 55, z: 0, w: 12, d: 12, h: 3 },
  { x: 55, z: 25, w: 12, d: 12, h: 3 },
  
  // Top buildings
  { x: -20, z: 45, w: 25, d: 12, h: 3 },
  { x: 25, z: 50, w: 18, d: 10, h: 2.5 },
  
  // Circular building (bottom left)
  { x: -35, z: -30, w: 10, d: 10, h: 3, circular: true },
  
  // Additional buildings
  { x: -45, z: 10, w: 14, d: 10, h: 3 },
  { x: 10, z: -35, w: 15, d: 8, h: 2.5 },
]

// Grass/field areas
const GRASS_AREAS = [
  { x: -10, z: 20, w: 25, d: 18 },
  { x: 35, z: 45, w: 12, d: 8 },
]

// Roads
const ROADS = [
  { x: 15, z: -5, w: 8, d: 60 },
  { x: -10, z: 5, w: 50, d: 6 },
]

export default function Campus3D() {
  return (
    <group>
      {/* Ground plane - light gray */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.1, 0]} receiveShadow>
        <planeGeometry args={[150, 150]} />
        <meshStandardMaterial color="#E8E8E8" />
      </mesh>

      {/* Grass areas - light green */}
      {GRASS_AREAS.map((grass, i) => (
        <mesh
          key={`grass-${i}`}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[grass.x, 0, grass.z]}
          receiveShadow
        >
          <planeGeometry args={[grass.w, grass.d]} />
          <meshStandardMaterial color="#90EE90" />
        </mesh>
      ))}

      {/* Roads - darker gray */}
      {ROADS.map((road, i) => (
        <mesh
          key={`road-${i}`}
          rotation={[-Math.PI / 2, 0, 0]}
          position={[road.x, 0.01, road.z]}
          receiveShadow
        >
          <planeGeometry args={[road.w, road.d]} />
          <meshStandardMaterial color="#C0C0C0" />
        </mesh>
      ))}

      {/* Buildings */}
      {BUILDINGS.map((building, i) => (
        <Building key={i} {...building} />
      ))}
    </group>
  )
}

function Building({ x, z, w, d, h, circular = false }) {
  const meshRef = useRef()

  useFrame((state) => {
    if (meshRef.current) {
      // Subtle glow pulse
      meshRef.current.material.emissiveIntensity = 0.1 + Math.sin(state.clock.elapsedTime * 2) * 0.05
    }
  })

  if (circular) {
    return (
      <group position={[x, h / 2, z]}>
        <mesh ref={meshRef} castShadow receiveShadow>
          <cylinderGeometry args={[w / 2, w / 2, h, 32]} />
          <meshStandardMaterial color="#F0F0F0" emissive="#F0F0F0" />
        </mesh>
        {/* Roof edge */}
        <mesh position={[0, h / 2, 0]}>
          <cylinderGeometry args={[w / 2 + 0.3, w / 2 + 0.3, 0.3, 32]} />
          <meshStandardMaterial color="#D0D0D0" />
        </mesh>
      </group>
    )
  }

  return (
    <group position={[x, h / 2, z]}>
      <mesh ref={meshRef} castShadow receiveShadow>
        <boxGeometry args={[w, h, d]} />
        <meshStandardMaterial color="#F5F5F5" emissive="#F5F5F5" emissiveIntensity={0.1} />
      </mesh>
      {/* Roof edge highlight */}
      <mesh position={[0, h / 2, 0]}>
        <boxGeometry args={[w + 0.5, 0.3, d + 0.5]} />
        <meshStandardMaterial color="#E0E0E0" />
      </mesh>
    </group>
  )
}
