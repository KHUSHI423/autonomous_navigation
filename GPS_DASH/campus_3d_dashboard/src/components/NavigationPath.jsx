import { useRef, useMemo, useState } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

// Navigation path from X to Y
const PATH_POINTS = [
  { x: -35, y: 0.3, z: -30 },  // X Start
  { x: -20, y: 0.3, z: -30 },
  { x: -5, y: 0.3, z: -30 },
  { x: 10, y: 0.3, z: -30 },
  { x: 15, y: 0.3, z: -20 },   // Turn
  { x: 15, y: 0.3, z: -5 },
  { x: 15, y: 0.3, z: 10 },    // Y Destination
]

export default function NavigationPath({ onLocationUpdate }) {
  const lineRef = useRef()
  const particleRef = useRef()
  const [progress, setProgress] = useState(0)

  const { lineGeometry } = useMemo(() => {
    const positions = new Float32Array(PATH_POINTS.flatMap(p => [p.x, p.y, p.z]))
    return { lineGeometry: positions }
  }, [])

  useFrame((state, delta) => {
    const time = state.clock.elapsedTime
    
    // Smooth animation - 8 seconds for full path
    const speed = 0.125 // 1/8 = 0.125 per second
    const newProgress = (time * speed) % 1
    setProgress(newProgress)
    
    // Update parent with progress percentage
    if (onLocationUpdate) {
      onLocationUpdate(Math.round(newProgress * 100))
    }

    // Animate path visibility (pulsing effect)
    if (lineRef.current) {
      lineRef.current.material.opacity = 0.6 + Math.sin(time * 4) * 0.2
    }

    // Update particle position
    if (particleRef.current) {
      const index = Math.floor(newProgress * (PATH_POINTS.length - 1))
      const nextIndex = Math.min(index + 1, PATH_POINTS.length - 1)
      const alpha = (newProgress * (PATH_POINTS.length - 1)) % 1

      const x = PATH_POINTS[index].x * (1 - alpha) + PATH_POINTS[nextIndex].x * alpha
      const y = PATH_POINTS[index].y * (1 - alpha) + PATH_POINTS[nextIndex].y * alpha
      const z = PATH_POINTS[index].z * (1 - alpha) + PATH_POINTS[nextIndex].z * alpha

      particleRef.current.position.set(x, y + 0.8, z)
    }
  })

  return (
    <group>
      {/* Navigation Path - Bright Blue */}
      <line ref={lineRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={PATH_POINTS.length}
            array={lineGeometry}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color="#0066FF"
          transparent
          opacity={0.8}
          linewidth={4}
        />
      </line>

      {/* Navigation Particle (arrow/dot) */}
      <mesh ref={particleRef} rotation={[-Math.PI / 2, 0, 0]}>
        <coneGeometry args={[1, 2, 4]} />
        <meshStandardMaterial
          color="#00FF00"
          emissive="#00FF00"
          emissiveIntensity={2}
        />
      </mesh>

      {/* Start Point - X */}
      <StartMarker position={PATH_POINTS[0]} />

      {/* End Point - Y */}
      <EndMarker position={PATH_POINTS[PATH_POINTS.length - 1]} />

      {/* Path waypoints indicators */}
      {PATH_POINTS.slice(1, -1).map((point, i) => (
        <Waypoint key={i} position={point} />
      ))}
    </group>
  )
}

// X Start Marker
function StartMarker({ position }) {
  return (
    <group position={[position.x, position.y + 2, position.z]}>
      {/* Yellow X */}
      <mesh rotation={[0, 0, Math.PI / 4]}>
        <boxGeometry args={[2, 0.4, 0.4]} />
        <meshStandardMaterial color="#FFD700" emissive="#FFD700" emissiveIntensity={1} />
      </mesh>
      <mesh rotation={[0, 0, -Math.PI / 4]}>
        <boxGeometry args={[2, 0.4, 0.4]} />
        <meshStandardMaterial color="#FFD700" emissive="#FFD700" emissiveIntensity={1} />
      </mesh>
      {/* Pole */}
      <mesh position={[0, -1.5, 0]}>
        <cylinderGeometry args={[0.15, 0.15, 3, 8]} />
        <meshStandardMaterial color="#666666" />
      </mesh>
      {/* Base */}
      <mesh position={[0, -3, 0]}>
        <cylinderGeometry args={[0.5, 0.5, 0.2, 8]} />
        <meshStandardMaterial color="#666666" />
      </mesh>
    </group>
  )
}

// Y End Marker
function EndMarker({ position }) {
  return (
    <group position={[position.x, position.y + 2, position.z]}>
      {/* Yellow Y */}
      <mesh position={[0, 0.5, 0]}>
        <boxGeometry args={[0.4, 1.2, 0.4]} />
        <meshStandardMaterial color="#FFD700" emissive="#FFD700" emissiveIntensity={1} />
      </mesh>
      <mesh rotation={[0, 0, -0.5]} position={[-0.5, 1.2, 0]}>
        <boxGeometry args={[1, 0.4, 0.4]} />
        <meshStandardMaterial color="#FFD700" emissive="#FFD700" emissiveIntensity={1} />
      </mesh>
      <mesh rotation={[0, 0, 0.5]} position={[0.5, 1.2, 0]}>
        <boxGeometry args={[1, 0.4, 0.4]} />
        <meshStandardMaterial color="#FFD700" emissive="#FFD700" emissiveIntensity={1} />
      </mesh>
      {/* Pole */}
      <mesh position={[0, -1.5, 0]}>
        <cylinderGeometry args={[0.15, 0.15, 3, 8]} />
        <meshStandardMaterial color="#666666" />
      </mesh>
      {/* Base */}
      <mesh position={[0, -3, 0]}>
        <cylinderGeometry args={[0.5, 0.5, 0.2, 8]} />
        <meshStandardMaterial color="#666666" />
      </mesh>
    </group>
  )
}

// Waypoint markers along the path
function Waypoint({ position }) {
  return (
    <mesh position={[position.x, position.y + 0.5, position.z]}>
      <sphereGeometry args={[0.3, 8, 8]} />
      <meshStandardMaterial
        color="#0066FF"
        emissive="#0066FF"
        emissiveIntensity={0.5}
      />
    </mesh>
  )
}
