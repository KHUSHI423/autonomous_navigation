import { useRef, useMemo, useState, useEffect } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export default function CampusModel({ onSelectLocation }) {
  const groupRef = useRef()
  const [modelData, setModelData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/models/campus_model.json')
      .then(res => {
        if (!res.ok) throw new Error('Failed to load model')
        return res.json()
      })
      .then(data => {
        setModelData(data)
        setLoading(false)
        console.log('Campus model loaded:', data.metadata)
      })
      .catch(err => {
        console.error('Error loading model:', err)
        setLoading(false)
      })
  }, [])

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.02) * 0.01
    }
  })

  // Process point cloud data
  const { positions, colors } = useMemo(() => {
    if (!modelData || !modelData.pointCloud) return { positions: null, colors: null }

    const rawPositions = modelData.pointCloud.positions
    const rawColors = modelData.pointCloud.colors

    // Calculate bounds for normalization
    let minZ = Infinity, maxZ = -Infinity
    rawPositions.forEach(p => {
      minZ = Math.min(minZ, p[2])
      maxZ = Math.max(maxZ, p[2])
    })
    const zRange = maxZ - minZ || 1

    // Flatten positions
    const flatPositions = rawPositions.flat()

    // Generate colors based on elevation (Z value)
    const flatColors = []
    rawPositions.forEach((p, i) => {
      const z = p[2]
      const normalizedZ = (z - minZ) / zRange

      // Color based on elevation
      let r, g, b
      if (normalizedZ < 0.2) {
        // Low elevation - green (ground/grass)
        r = 0.2
        g = 0.7
        b = 0.3
      } else if (normalizedZ < 0.5) {
        // Mid elevation - light green/yellow (paths)
        r = 0.6
        g = 0.6
        b = 0.4
      } else if (normalizedZ < 0.7) {
        // Higher - gray (buildings)
        r = 0.7
        g = 0.7
        b = 0.7
      } else {
        // Highest - white (rooftops)
        r = 0.9
        g = 0.9
        b = 0.9
      }
      flatColors.push(r, g, b)
    })

    return { positions: flatPositions, colors: flatColors }
  }, [modelData])

  if (loading || !positions) {
    return null
  }

  return (
    <group ref={groupRef}>
      {/* 
        The DXF data is in XY plane (top-down view).
        We rotate -90 deg on X axis to make it vertical for 3D viewing.
        This makes: X->X, Y->Z (up), Z->Y (depth)
      */}
      <group rotation={[-Math.PI / 2, 0, 0]}>
        <points>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={positions.length / 3}
              array={new Float32Array(positions)}
              itemSize={3}
            />
            <bufferAttribute
              attach="attributes-color"
              count={colors.length / 3}
              array={new Float32Array(colors)}
              itemSize={3}
            />
          </bufferGeometry>
          <pointsMaterial
            size={0.5}
            vertexColors
            transparent
            opacity={0.9}
            sizeAttenuation={true}
          />
        </points>
      </group>
    </group>
  )
}
