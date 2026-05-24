import { useState, useEffect, Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import Campus3D from './components/Campus3D'
import NavigationPath from './components/NavigationPath'

function App() {
  const [navigationActive, setNavigationActive] = useState(false)
  const [currentLocation, setCurrentLocation] = useState(null)

  const startNavigation = () => {
    setNavigationActive(true)
  }

  const stopNavigation = () => {
    setNavigationActive(false)
    setCurrentLocation(null)
  }

  return (
    <div className="relative w-full h-screen bg-gray-100 overflow-hidden">
      {/* 3D Canvas */}
      <Canvas shadows camera={{ position: [0, -120, 100], fov: 30 }}>
        <PerspectiveCamera makeDefault position={[0, -120, 100]} fov={30} />
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={50}
          maxDistance={200}
          maxPolarAngle={Math.PI / 2.2}
          target={[0, 0, 0]}
        />

        {/* Lighting */}
        <ambientLight intensity={0.7} />
        <directionalLight
          position={[50, 50, 30]}
          intensity={1}
          castShadow
        />

        {/* Campus Map */}
        <Suspense fallback={null}>
          <Campus3D />
          {navigationActive && (
            <NavigationPath onLocationUpdate={setCurrentLocation} />
          )}
        </Suspense>
      </Canvas>

      {/* Navigation Header */}
      <div className="absolute top-0 left-0 right-0 bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-4 shadow-lg">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-4">
            <div className="bg-white/20 p-2 rounded-lg">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </div>
            <div>
              <h1 className="text-white font-bold text-lg">Campus Navigation</h1>
              <p className="text-blue-100 text-xs">CIT Campus</p>
            </div>
          </div>
          
          {navigationActive ? (
            <button
              onClick={stopNavigation}
              className="bg-red-500 hover:bg-red-600 text-white px-6 py-2 rounded-lg font-medium transition-colors"
            >
              End Navigation
            </button>
          ) : (
            <button
              onClick={startNavigation}
              className="bg-green-500 hover:bg-green-600 text-white px-6 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Start Navigation
            </button>
          )}
        </div>
      </div>

      {/* Navigation Instructions Panel */}
      {navigationActive && (
        <div className="absolute top-24 left-4 bg-white rounded-xl shadow-xl p-4 w-64">
          <div className="flex items-center gap-3 mb-3">
            <div className="bg-blue-100 p-2 rounded-lg">
              <svg className="w-6 h-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
              </svg>
            </div>
            <div>
              <div className="text-sm font-semibold text-gray-800">Route Guidance</div>
              <div className="text-xs text-gray-500">Follow the blue path</div>
            </div>
          </div>
          
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-sm">
              <div className="w-6 h-6 bg-yellow-400 rounded flex items-center justify-center text-white font-bold text-xs">X</div>
              <span className="text-gray-600">Start Point</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-0.5 h-4 bg-blue-400 ml-3"></div>
              <span className="text-gray-500 text-xs">Follow path</span>
            </div>
            <div className="flex items-center gap-2 text-sm">
              <div className="w-6 h-6 bg-yellow-400 rounded flex items-center justify-center text-white font-bold text-xs">Y</div>
              <span className="text-gray-600">Destination</span>
            </div>
          </div>
        </div>
      )}

      {/* Current Location Info */}
      {navigationActive && currentLocation && (
        <div className="absolute top-24 right-4 bg-white rounded-xl shadow-xl p-4 w-48">
          <div className="text-xs text-gray-500 mb-1">Distance Covered</div>
          <div className="text-2xl font-bold text-blue-600">{currentLocation}%</div>
          <div className="mt-2 bg-gray-200 rounded-full h-2 overflow-hidden">
            <div 
              className="bg-gradient-to-r from-blue-500 to-green-500 h-full transition-all duration-300"
              style={{ width: `${currentLocation}%` }}
            />
          </div>
        </div>
      )}

      {/* Bottom Navigation Bar */}
      <div className="absolute bottom-0 left-0 right-0 bg-white shadow-lg px-6 py-4">
        <div className="flex justify-between items-center">
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${navigationActive ? 'bg-green-500 animate-pulse' : 'bg-gray-300'}`}></div>
              <span className="text-sm text-gray-600">
                {navigationActive ? 'Navigating...' : 'Ready'}
              </span>
            </div>
            <div className="h-4 w-px bg-gray-300"></div>
            <div className="text-sm text-gray-600">
              {navigationActive ? (
                <span>Route: <span className="font-semibold text-blue-600">X → Y</span></span>
              ) : (
                <span>Click "Start Navigation" to begin</span>
              )}
            </div>
          </div>
          
          <div className="flex items-center gap-4 text-xs text-gray-500">
            <span>🖱️ Drag to rotate</span>
            <span>🔍 Scroll to zoom</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
