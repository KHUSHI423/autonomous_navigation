import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

// Path from X to Y matching the reference image
const PATH_POINTS = [
  { x: -35, y: 0.5, z: -30 },  // Start: X (near circular building)
  { x: -20, y: 0.5, z: -30 },  // Move east
  { x: -5, y: 0.5, z: -30 },   // Continue east
  { x: 10, y: 0.5, z: -30 },   // Continue east
  { x: 15, y: 0.5, z: -20 },   // Turn north
  { x: 15, y: 0.5, z: -5 },    // Continue north
  { x: 15, y: 0.5, z: 10 },    // End: Y (near buildings)
]

export default function PathSimulation() {
  const lineRef = useRef()
  const particleRef = useRef()
  const glowRef = useRef()

  const { lineGeometry } = useMemo(() => {
    const positions = new Float32Array(PATH_POINTS.flatMap(p => [p.x, p.y, p.z]))
    return { lineGeometry: positions }
  }, [])

  useFrame((state) => {
    const time = state.clock.elapsedTime
    
    // Loop animation every 8 seconds
    const loopProgress = (time % 8) / 8

    // Animate line opacity
    if (lineRef.current) {
      lineRef.current.material.opacity = 0.5 + Math.sin(time * 3) * 0.2
    }

    // Update particle position along path
    if (particleRef.current && glowRef.current) {
      const index = Math.floor(loopProgress * (PATH_POINTS.length - 1))
      const nextIndex = Math.min(index + 1, PATH_POINTS.length - 1)
      const alpha = (loopProgress * (PATH_POINTS.length - 1)) % 1

      const x = PATH_POINTS[index].x * (1 - alpha) + PATH_POINTS[nextIndex].x * alpha
      const y = PATH_POINTS[index].y * (1 - alpha) + PATH_POINTS[nextIndex].y * alpha
      const z = PATH_POINTS[index].z * (1 - alpha) + PATH_POINTS[nextIndex].z * alpha

      particleRef.current.position.set(x, y + 0.5, z)
      glowRef.current.position.set(x, y + 0.5, z)
      
      // Add slight bounce
      particleRef.current.position.y += Math.sin(time * 6) * 0.15
    }
  })

  return (
    <group>
      {/* Path Line - Blue like in the reference image */}
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
          color="#00BFFF"
          transparent
          opacity={0.7}
          linewidth={3}
        />
      </line>

      {/* Animated Particle (orange dot) */}
      <mesh ref={particleRef}>
        <sphereGeometry args={[0.7, 16, 16]} />
        <meshStandardMaterial
          color="#FF6600"
          emissive="#FF6600"
          emissiveIntensity={2}
        />
      </mesh>

      {/* Particle Glow */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[1.3, 16, 16]} />
        <meshBasicMaterial
          color="#FF6600"
          transparent
          opacity={0.3}
        />
      </mesh>

      {/* X Marker (Start) - Yellow */}
      <XMarker position={PATH_POINTS[0]} />

      {/* Y Marker (End) - Yellow */}
      <YMarker position={PATH_POINTS[PATH_POINTS.length - 1]} />
    </group>
  )
}

// X Marker - Yellow X shape
function XMarker({ position }) {
  return (
    <group position={[position.x, position.y + 1.5, position.z]}>
      {/* First diagonal of X */}
      <mesh rotation={[0, 0, Math.PI / 4]}>
        <boxGeometry args={[2.5, 0.35, 0.35]} />
        <meshStandardMaterial
          color="#FFFF00"
          emissive="#FFFF00"
          emissiveIntensity={1.5}
        />
      </mesh>
      {/* Second diagonal of X */}
      <mesh rotation={[0, 0, -Math.PI / 4]}>
        <boxGeometry args={[2.5, 0.35, 0.35]} />
        <meshStandardMaterial
          color="#FFFF00"
          emissive="#FFFF00"
          emissiveIntensity={1.5}
        />
      </mesh>
      {/* Vertical line to ground */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, 0, 0, -1.5, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#FFFF00" transparent opacity={0.4} />
      </line>
    </group>
  )
}

// Y Marker - Yellow Y shape
function YMarker({ position }) {
  return (
    <group position={[position.x, position.y + 1.5, position.z]}>
      {/* Vertical stem of Y */}
      <mesh position={[0, 0.5, 0]}>
        <boxGeometry args={[0.35, 1.2, 0.35]} />
        <meshStandardMaterial
          color="#FFFF00"
          emissive="#FFFF00"
          emissiveIntensity={1.5}
        />
      </mesh>
      {/* Left diagonal of Y */}
      <mesh rotation={[0, 0, -0.5]} position={[-0.45, 1.1, 0]}>
        <boxGeometry args={[1, 0.35, 0.35]} />
        <meshStandardMaterial
          color="#FFFF00"
          emissive="#FFFF00"
          emissiveIntensity={1.5}
        />
      </mesh>
      {/* Right diagonal of Y */}
      <mesh rotation={[0, 0, 0.5]} position={[0.45, 1.1, 0]}>
        <boxGeometry args={[1, 0.35, 0.35]} />
        <meshStandardMaterial
          color="#FFFF00"
          emissive="#FFFF00"
          emissiveIntensity={1.5}
        />
      </mesh>
      {/* Vertical line to ground */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, 0, 0, -1.5, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#FFFF00" transparent opacity={0.4} />
      </line>
    </group>
  )
}
