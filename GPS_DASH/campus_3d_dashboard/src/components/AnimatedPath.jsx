import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export default function AnimatedPath({ points }) {
  const lineRef = useRef()
  const particleRef = useRef()

  const { lineGeometry } = useMemo(() => {
    const positions = new Float32Array(points.flatMap(p => [p.x, p.y, p.z]))
    return { lineGeometry: positions }
  }, [points])

  useFrame((state) => {
    const time = state.clock.elapsedTime
    
    // Loop animation every 10 seconds
    const loopProgress = (time % 10) / 10

    // Pulse the line
    if (lineRef.current) {
      lineRef.current.material.opacity = 0.4 + Math.sin(time * 2) * 0.2
    }

    // Update particle position
    if (particleRef.current) {
      const index = Math.floor(loopProgress * (points.length - 1))
      const nextIndex = Math.min(index + 1, points.length - 1)
      const alpha = (loopProgress * (points.length - 1)) % 1

      const x = points[index].x * (1 - alpha) + points[nextIndex].x * alpha
      const y = points[index].y * (1 - alpha) + points[nextIndex].y * alpha
      const z = points[index].z * (1 - alpha) + points[nextIndex].z * alpha

      particleRef.current.position.set(x, y + 0.5, z)
    }
  })

  return (
    <group>
      {/* Path Line - Blue */}
      <line ref={lineRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={points.length}
            array={lineGeometry}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial
          color="#00BFFF"
          transparent
          opacity={0.7}
          linewidth={2}
        />
      </line>

      {/* Moving Particle */}
      <mesh ref={particleRef}>
        <sphereGeometry args={[0.6, 16, 16]} />
        <meshStandardMaterial
          color="#ff6600"
          emissive="#ff6600"
          emissiveIntensity={2}
        />
      </mesh>

      {/* X Marker (Start) */}
      <XMarker position={points[0]} />

      {/* Y Marker (End) */}
      <YMarker position={points[points.length - 1]} />
    </group>
  )
}

// X Marker - Yellow
function XMarker({ position }) {
  return (
    <group position={[position.x, position.y + 1, position.z]}>
      {/* X shape */}
      <mesh rotation={[0, 0, Math.PI / 4]}>
        <boxGeometry args={[2, 0.25, 0.25]} />
        <meshStandardMaterial color="#FFFF00" emissive="#FFFF00" emissiveIntensity={1.5} />
      </mesh>
      <mesh rotation={[0, 0, -Math.PI / 4]}>
        <boxGeometry args={[2, 0.25, 0.25]} />
        <meshStandardMaterial color="#FFFF00" emissive="#FFFF00" emissiveIntensity={1.5} />
      </mesh>
      {/* Line to ground */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, 0, 0, -1, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#FFFF00" transparent opacity={0.4} />
      </line>
    </group>
  )
}

// Y Marker - Yellow
function YMarker({ position }) {
  return (
    <group position={[position.x, position.y + 1, position.z]}>
      {/* Y shape */}
      <mesh position={[0, 0.4, 0]}>
        <boxGeometry args={[0.25, 1, 0.25]} />
        <meshStandardMaterial color="#FFFF00" emissive="#FFFF00" emissiveIntensity={1.5} />
      </mesh>
      <mesh rotation={[0, 0, -0.5]} position={[-0.35, 0.85, 0]}>
        <boxGeometry args={[0.7, 0.25, 0.25]} />
        <meshStandardMaterial color="#FFFF00" emissive="#FFFF00" emissiveIntensity={1.5} />
      </mesh>
      <mesh rotation={[0, 0, 0.5]} position={[0.35, 0.85, 0]}>
        <boxGeometry args={[0.7, 0.25, 0.25]} />
        <meshStandardMaterial color="#FFFF00" emissive="#FFFF00" emissiveIntensity={1.5} />
      </mesh>
      {/* Line to ground */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, 0, 0, -1, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#FFFF00" transparent opacity={0.4} />
      </line>
    </group>
  )
}
